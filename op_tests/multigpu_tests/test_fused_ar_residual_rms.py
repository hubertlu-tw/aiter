# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
import aiter
import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import itertools
from aiter import dtypes

from aiter.dist.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
    get_tp_group,
    graph_capture,
    destroy_model_parallel,
    destroy_distributed_environment,
)
from aiter.dist.utils import get_open_port, get_distributed_init_method, get_ip
from aiter.dist.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_allreduce_rmsnorm,
)
from aiter.test_common import (
    checkAllclose,
    perftest,
    benchmark,
)
from multiprocessing import set_start_method, Pool, freeze_support
import logging

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def fused_ar_residual_rmsnorm(tp_size, pp_size, rankID, x, residual, weight, eps, withGraph=False):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)
    residual = residual.to(device)
    weight = weight.to(device)
    # dist.barrier(device_ids=[i for i in range(tp_size)])

    # warmup and align all gpu
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc:
            with torch.cuda.graph(graph, stream=gc.stream):
                # Use the new fused_allreduce_residual_rmsnorm function
                from aiter.dist.communication_op import tensor_model_parallel_fused_allreduce_residual_rmsnorm
                out, residual_out = tensor_model_parallel_fused_allreduce_residual_rmsnorm(x, residual, weight, eps)
        out.fill_(0)
        residual_out.fill_(0)

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
        out = (out, residual_out, us)
    else:

        @perftest()
        def run_ca(x, residual):
            from aiter.dist.communication_op import tensor_model_parallel_fused_allreduce_residual_rmsnorm
            return tensor_model_parallel_fused_allreduce_residual_rmsnorm(x, residual, weight, eps)

        result, us = run_ca(x, residual)
        out, residual_out = result
        out = (out, residual_out, us)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


def split_ar_residual_rmsnorm(tp_size, pp_size, rankID, x, residual, weight, eps, withGraph=False):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)
    residual = residual.to(device)
    weight = weight.to(device)
    # dist.barrier(device_ids=[i for i in range(tp_size)])

    # warmup and align all gpu
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc:
            with torch.cuda.graph(graph, stream=gc.stream):
                ar_out = tensor_model_parallel_all_reduce(x)
                out = torch.empty_like(ar_out)
                residual_out = torch.empty_like(ar_out)
                aiter.rmsnorm2d_fwd_with_add(
                    out,
                    ar_out,
                    residual,
                    residual_out,
                    weight,
                    eps,
                    0,
                )
        out.fill_(0)
        residual_out.fill_(0)

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
        out = (out, residual_out, us)
    else:

        @perftest()
        def run_ca(x, residual):
            ar_out = tensor_model_parallel_all_reduce(x)
            out = torch.empty_like(ar_out)
            residual_out = torch.empty_like(ar_out)
            aiter.rmsnorm2d_fwd_with_add(
                out,
                ar_out,
                residual,
                residual_out,
                weight,
                eps,
                0,
            )
            return out, residual_out

        out, residual_out = run_ca(x, residual)
        out = (out, residual_out, 0)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


@benchmark()
def test_split_ar_residual_rmsnorm(tp_size, pp_size, shape, dtype, withGraph=False):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    cpu_rslt = []
    cpu_residual_rslt = []
    weight_list = []
    res_inp = []
    res_residual = []
    
    m = shape[0]
    n = shape[1]
    eps = 1e-6
    
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        residual = torch.randn(shape, dtype=dtype)
        res_inp.append(x)
        res_residual.append(residual)
        ref += x
        weight = torch.randn((n,), dtype=dtype)
        weight_list.append(weight)
        rets.append(
            pool.apply_async(
                split_ar_residual_rmsnorm, args=(tp_size, pp_size, i, x, residual, weight, eps, withGraph)
            )
        )
    pool.close()
    pool.join()
    
    for i in range(tp_size):
        # Apply flashinfer formula: (X_sum + R) * γ / sqrt(mean((X_sum + R)^2) + ε)
        residual_added = ref + res_residual[i]
        host_rslt = F.rms_norm(
            input=residual_added,
            normalized_shape=(ref.shape[-1],),
            weight=weight_list[i],
            eps=eps,
        )
        cpu_rslt.append(host_rslt)
        cpu_residual_rslt.append(residual_added)
    
    rets = [el.get() for el in rets]
    for i, result in enumerate(rets):
        print(f"DEBUG: result type: {type(result)}, result: {result}")
        # Handle the nested tuple structure: ((out, residual_out), us, 0)
        if isinstance(result, tuple) and len(result) == 3:
            if isinstance(result[0], tuple) and len(result[0]) == 2:
                # Nested structure: ((out, residual_out), us, 0)
                out_residual, us, _ = result
                out, residual_out = out_residual
            else:
                # Flat structure: (out, residual_out, us)
                out, residual_out, us = result
            
            print(f"DEBUG: out type: {type(out)}, out: {out}")
            print(f"DEBUG: residual_out type: {type(residual_out)}, residual_out: {residual_out}")
            msg = f"test_split_ar_residual_rmsnorm: {shape=} {dtype=} {withGraph=} {us:>8.2f}"
            checkAllclose(cpu_rslt[i], out.to(ref), msg=msg)
            checkAllclose(cpu_residual_rslt[i], residual_out.to(ref), msg=msg)
        else:
            # Handle the case where it's not a tuple (fallback)
            out, residual_out = result
            msg = f"test_split_ar_residual_rmsnorm: {shape=} {dtype=} {withGraph=}"
            checkAllclose(cpu_rslt[i], out.to(ref), msg=msg)
            checkAllclose(cpu_residual_rslt[i], residual_out.to(ref), msg=msg)


@benchmark()
def test_fused_ar_residual_rmsnorm(tp_size, pp_size, shape, dtype, withGraph=False):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    cpu_rslt = []
    cpu_residual_rslt = []
    weight_list = []
    res_inp = []
    res_residual = []
    
    m = shape[0]
    n = shape[1]
    eps = 1e-6
    
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        residual = torch.randn(shape, dtype=dtype)
        res_inp.append(x)
        res_residual.append(residual)
        ref += x
        weight = torch.randn((n,), dtype=dtype)
        weight_list.append(weight)
        rets.append(
            pool.apply_async(
                fused_ar_residual_rmsnorm, args=(tp_size, pp_size, i, x, residual, weight, eps, withGraph)
            )
        )
    pool.close()
    pool.join()

    for i in range(tp_size):
        # Apply flashinfer formula: (X_sum + R) * γ / sqrt(mean((X_sum + R)^2) + ε)
        residual_added = ref + res_residual[i]
        host_rslt = F.rms_norm(
            input=residual_added,
            normalized_shape=(ref.shape[-1],),
            weight=weight_list[i],
            eps=eps,
        )
        cpu_rslt.append(host_rslt)
        cpu_residual_rslt.append(residual_added)

    rets = [el.get() for el in rets]
    for i, result in enumerate(rets):
        print(f"DEBUG: Result type: {type(result)}, length: {len(result) if isinstance(result, tuple) else 'N/A'}")
        if isinstance(result, tuple) and len(result) == 3:
            out, residual_out, us = result
            msg = f"test_fused_ar_residual_rmsnorm: {shape=} {dtype=} {withGraph=} {us:>8.2f}"
            checkAllclose(cpu_rslt[i], out.to(ref), msg=msg)
            checkAllclose(cpu_residual_rslt[i], residual_out.to(ref), msg=msg)
        else:
            # Handle unexpected result format - assume it's a 2-element tuple with timing info
            if isinstance(result, tuple) and len(result) == 2:
                out, residual_out = result
                us = 0
                msg = f"test_fused_ar_residual_rmsnorm: {shape=} {dtype=} {withGraph=} {us:>8.2f}"
                checkAllclose(cpu_rslt[i], out.to(ref), msg=msg)
                checkAllclose(cpu_residual_rslt[i], residual_out.to(ref), msg=msg)
            else:
                # Fallback for any other format
                print(f"WARNING: Unexpected result format: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                if isinstance(result, tuple) and len(result) >= 2:
                    out, residual_out = result[0], result[1]
                    msg = f"test_fused_ar_residual_rmsnorm: {shape=} {dtype=} {withGraph=}"
                    checkAllclose(cpu_rslt[i], out.to(ref), msg=msg)
                    checkAllclose(cpu_residual_rslt[i], residual_out.to(ref), msg=msg)


l_dtype = ["bf16"]
l_shape = [
    (64, 7168)
]
l_tp = [8]
l_pp = [1]
l_graph = [True, False]

parser = argparse.ArgumentParser(description="config input of test")
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="data type",
)
parser.add_argument(
    "-s",
    "--shape",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="shape. e.g. -s 128,8192",
)

parser.add_argument(
    "-t",
    "--tp",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="tp num. e.g. -t 8",
)

parser.add_argument(
    "-p",
    "--pp",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="tp num. e.g. -p 1",
)

parser.add_argument(
    "-g",
    "--graphon",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="open cudagraph. e.g. -g 1",
)


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    if args.shape is not None:
        l_shape = [args.shape]
    if args.tp is not None:
        l_tp = [args.tp]
    if args.pp is not None:
        l_pp = [args.pp]
    if args.graphon is not None:
        print(args.graphon)
        l_graph = [args.graphon]
    for dtype, shape, tp, pp, graph_on in itertools.product(
        l_dtype, l_shape, l_tp, l_pp, l_graph
    ):
        test_split_ar_residual_rmsnorm(tp, pp, shape, dtype, withGraph=graph_on)
        test_fused_ar_residual_rmsnorm(tp, pp, shape, dtype, withGraph=graph_on)
