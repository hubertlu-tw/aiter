# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 range regression for QuickReduce.

With cast_bf2half the kernel reduces in fp16, which saturates at 65504 while
bf16 activations (e.g. gpt-oss) go well past it. An over-range element must
not turn into inf/NaN, and for the block-scaled codecs it must not poison its
32-value block. Covers the plain two-shot kernel and the add+RMSNorm epilogue.
"""

import argparse
import multiprocessing
import socket

import torch
import torch.distributed as dist

import aiter as ops
from aiter.dist.device_communicators.quick_all_reduce import qr_exchange_handles

# Ids match QuickReduceRegime.
CODECS = {"FP": 0, "FP8": 1, "INT6": 2, "INT4": 3, "INT3": 4}
NUMEL = 1 << 20
HIDDEN = 4096


def _open_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _cases(world_size):
    # (name, value on every rank, every 32nd element set to 80000)
    sum_above = 40_000.0 if world_size == 2 else 20_000.0
    return [
        ("low", 2**-8, False),
        ("ordinary", 100.0, False),
        ("sum_above_fp16", sum_above, False),
        ("negative_sum_above_fp16", -sum_above, False),
        ("input_above_fp16", 80_000.0, False),
        ("mixed_input_above_fp16", 1.0, True),
    ]


def _input(value, mixed, device):
    inp = torch.full((NUMEL,), value, dtype=torch.bfloat16, device=device)
    if mixed:
        inp[::32] = 80_000.0
    return inp


def _check(out, expected, rtol, context):
    assert torch.isfinite(out).all().item(), f"{context}: non-finite output"
    assert torch.count_nonzero(out).item() > 0, f"{context}: all-zero output"
    if rtol is not None:
        torch.testing.assert_close(
            out, expected, rtol=rtol, atol=0, msg=lambda m: f"{context}\n{m}"
        )


def _arch():
    return getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")


def _worker(rank, world_size, port, casts):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    codecs = {k: v for k, v in CODECS.items() if k != "INT3" or world_size == 2}
    # gfx1250 only implements BF16 INT4 with the fp16 cast (PullQ4); the other
    # two-shot paths are stubs there.
    is_gfx1250 = "gfx1250" in _arch()
    if is_gfx1250:
        codecs = {"INT4": CODECS["INT4"]}
        casts = tuple(c for c in casts if c)
    try:
        for codec, level in codecs.items():
            ptr = ops.init_custom_qr(rank, world_size, None, level)
            try:
                qr_exchange_handles(ptr, world_size, group)
                for cast in casts:
                    for case, value, mixed in _cases(world_size):
                        context = f"TP{world_size} {codec} cast={cast} {case}"
                        inp = _input(value, mixed, device)
                        expected = (inp.float() * world_size).to(torch.bfloat16)
                        out = torch.empty_like(inp)
                        dist.barrier(group)
                        ops.qr_all_reduce(ptr, inp, out, level, cast)
                        torch.cuda.synchronize()
                        # FP must stay exact across the whole range. Uniform in-range
                        # blocks quantize exactly with the integer codecs; FP8's 1/240
                        # block scale is inexact, so it gets a tolerance.
                        if codec == "FP":
                            rtol = 0.0
                        elif case in ("low", "ordinary"):
                            rtol = 2e-2 if codec == "FP8" else 0.0
                        else:
                            rtol = None
                        _check(out, expected, rtol, context)

                        # The add+RMSNorm epilogue has no INT3 or gfx1250 path.
                        if codec == "INT3" or is_gfx1250:
                            continue
                        residual = torch.zeros_like(inp).view(-1, HIDDEN)
                        weight = torch.ones(HIDDEN, dtype=torch.bfloat16, device=device)
                        norm_out = torch.empty_like(residual)
                        residual_out = torch.empty_like(residual)
                        dist.barrier(group)
                        ops.qr_all_reduce_rmsnorm(
                            ptr,
                            inp.view(-1, HIDDEN),
                            residual,
                            residual_out,
                            norm_out,
                            weight,
                            1e-6,
                            HIDDEN,
                            level,
                            cast,
                        )
                        torch.cuda.synchronize()
                        _check(
                            residual_out.view(-1), expected, rtol, f"{context} rmsnorm"
                        )
                        assert (
                            torch.isfinite(norm_out).all().item()
                        ), f"{context} rmsnorm: non-finite norm output"
            finally:
                ops.qr_destroy(ptr)
    finally:
        dist.destroy_process_group()


def test_quick_all_reduce_bf16_range(world_size, casts=(1, 0)):
    multiprocessing.set_start_method("spawn", force=True)
    port = _open_port()
    processes = [
        multiprocessing.Process(target=_worker, args=(rank, world_size, port, casts))
        for rank in range(world_size)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join(timeout=300)
    for rank, p in enumerate(processes):
        if p.is_alive():
            p.terminate()
            p.join()
            raise RuntimeError(f"TP{world_size} rank {rank} timed out")
        assert p.exitcode == 0, f"TP{world_size} rank {rank} failed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument(
        "--include-gfx1250",
        action="store_true",
        help="run on gfx1250, whose PullQ4 path does not guard the fp16 range yet",
    )
    args = parser.parse_args()
    if "gfx1250" in _arch() and not args.include_gfx1250:
        raise SystemExit("skip: gfx1250 PullQ4 has no fp16 range guard yet")
    for tp in args.tp:
        if tp <= torch.cuda.device_count():
            test_quick_all_reduce_bf16_range(tp)
