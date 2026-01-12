import pytest
import torch

from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.ops.triton.utils.types import get_fp8_dtypes


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # 1 - cosine-like similarity used in other aiter tests/benches
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum().clamp(min=1e-12)
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def _kv_cache_cast_to_fp8_packed(
    kv_bf16: torch.Tensor, fp8_dtype: torch.dtype
) -> torch.Tensor:
    """
    Pack kv cache into the format expected by deepgemm_fp8_paged_mqa_logits:
      kv_cache_u8: (num_pages, KVBlockSize, 1, hidden_dim + 4) uint8, where memory layout is:
        - first KVBlockSize * hidden_dim bytes: fp8 data (flattened)
        - next  KVBlockSize * 4 bytes: fp32 scales (flattened)
    """
    assert kv_bf16.dtype == torch.bfloat16
    assert kv_bf16.ndim == 4  # (num_pages, KVBlockSize, 1, hidden_dim)
    num_pages, KVBlockSize, num_heads, hidden_dim = kv_bf16.shape
    assert num_heads == 1

    fp8_max = torch.finfo(fp8_dtype).max
    x_amax = kv_bf16.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / fp8_max
    kv_fp8 = (kv_bf16 * (1.0 / sf)).to(fp8_dtype)

    # IMPORTANT: match the layout in op_tests/op_benchmarks/triton/bench_deepgemm_attention.py
    # (all fp8 data first, then all fp32 scales).
    packed_u8 = torch.empty(
        (num_pages, KVBlockSize * (hidden_dim + 4)),
        device=kv_bf16.device,
        dtype=torch.uint8,
    )
    packed_u8[:, : KVBlockSize * hidden_dim] = kv_fp8.view(
        num_pages, KVBlockSize * hidden_dim
    ).view(torch.uint8)
    packed_u8[:, KVBlockSize * hidden_dim :] = (
        sf.to(torch.float32).view(num_pages, KVBlockSize).view(torch.uint8)
    )
    return packed_u8.view(num_pages, KVBlockSize, 1, hidden_dim + 4)


def _ref_fp8_paged_mqa_logits_from_packed(
    q_fp8: torch.Tensor,
    kv_cache_packed: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    kv_block_size: int,
) -> torch.Tensor:
    """
    Reference implementation that uses the SAME packed fp8+scale KV cache format,
    so it mainly validates paging/indexing correctness.
    """
    batch_size, next_n, heads, hidden_dim = q_fp8.shape
    assert weights.shape == (batch_size * next_n, heads)
    assert context_lens.shape == (batch_size,)
    assert block_tables.ndim == 2
    _, fp8_dtype = get_fp8_dtypes()

    # Unpack KV (fp8 data + fp32 scale) using the same layout assumptions as the kernel wrapper.
    assert kv_cache_packed.dtype == torch.uint8
    num_pages = kv_cache_packed.shape[0]
    flat = kv_cache_packed.view(num_pages, kv_block_size * (hidden_dim + 4))
    kv_fp8 = (
        flat[:, : kv_block_size * hidden_dim]
        .view(fp8_dtype)
        .view(num_pages, kv_block_size, 1, hidden_dim)
        .to(torch.float32)
    )
    kv_scale = (
        flat[:, kv_block_size * hidden_dim :]
        .view(torch.float32)
        .view(num_pages, kv_block_size, 1, 1)
    )
    kv = kv_fp8 * kv_scale

    q = q_fp8.to(torch.float32)

    out = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q_fp8.device,
        dtype=torch.float32,
    )

    # Small, explicit reference (ok for unit-test sizes)
    for b in range(batch_size):
        ctx_len = int(context_lens[b].item())
        for j in range(next_n):
            row = b * next_n + j
            qpos = ctx_len - next_n + j
            if qpos < 0:
                continue
            for t in range(min(ctx_len, max_model_len)):
                if t > qpos:
                    continue
                page_rk = t // kv_block_size
                tok_off = t % kv_block_size
                page_id = int(block_tables[b, page_rk].item())
                k_t = kv[page_id, tok_off, 0]  # (hidden_dim,)

                # score per head: (heads,)
                s = (q[b, j] * k_t[None, :]).sum(dim=-1)  # (heads,)
                s = torch.relu(s) * weights[row]
                out[row, t] = s.sum()

    return out


@pytest.mark.parametrize("KVBlockSize", [64])
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("heads", [16])
@pytest.mark.parametrize("hidden_dim", [32])
@torch.inference_mode()
def test_deepgemm_fp8_paged_mqa_logits_kvblocksize_64(
    KVBlockSize: int, next_n: int, heads: int, hidden_dim: int
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")

    _, fp8_dtype = get_fp8_dtypes()

    torch.manual_seed(0)
    device = "cuda"
    batch_size = 2
    max_model_len = 128

    num_pages = (max_model_len + KVBlockSize - 1) // KVBlockSize

    # Inputs
    q_bf16 = torch.randn(
        (batch_size, next_n, heads, hidden_dim), device=device, dtype=torch.bfloat16
    )
    q_fp8 = q_bf16.to(fp8_dtype)

    kv_bf16 = torch.randn(
        (num_pages, KVBlockSize, 1, hidden_dim), device=device, dtype=torch.bfloat16
    )
    kv_cache_packed = _kv_cache_cast_to_fp8_packed(kv_bf16, fp8_dtype)

    weights = torch.randn(
        (batch_size * next_n, heads), device=device, dtype=torch.float32
    )

    # Pick context lens such that decode window is valid
    context_lens = torch.tensor([65, 103], device=device, dtype=torch.int32)

    # Page table: (batch, num_pages)
    block_tables = torch.arange(num_pages, device=device, dtype=torch.int32).repeat(
        batch_size, 1
    )

    out_logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )

    deepgemm_fp8_paged_mqa_logits(
        q_fp8,
        kv_cache_packed,
        weights,
        out_logits,
        context_lens,
        block_tables,
        max_model_len,
        Preshuffle=False,
        KVBlockSize=KVBlockSize,
        ChunkK=128,
        TotalCuCount=256,
        WavePerEU=2,
    )

    ref = _ref_fp8_paged_mqa_logits_from_packed(
        q_fp8=q_fp8,
        kv_cache_packed=kv_cache_packed,
        weights=weights,
        context_lens=context_lens,
        block_tables=block_tables,
        max_model_len=max_model_len,
        kv_block_size=KVBlockSize,
    )

    # Compare with a mask matching causal + context len constraint (same as bench)
    positions = torch.arange(max_model_len, device=device)[None, :].expand(
        batch_size * next_n, -1
    )
    row_indices = torch.arange(batch_size * next_n, device=device) // next_n
    next_n_offset = torch.arange(batch_size * next_n, device=device) % next_n
    mask = positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(
        1
    )

    # Ensure -inf masks are identical
    assert torch.equal(out_logits == float("-inf"), ref == float("-inf"))

    out_m = out_logits.masked_fill(~mask, 0)
    ref_m = ref.masked_fill(~mask, 0)
    diff = _calc_diff(out_m, ref_m).item()
    assert diff < 5e-3, f"{diff=}"
