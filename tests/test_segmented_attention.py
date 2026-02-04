import torch
import pytest
import time
try:
    from vllm_flash_attn import flash_attn_varlen_func
except ImportError:
    from flash_attn import flash_attn_varlen_func


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_segmented_attention_performance():
    device = "cuda"
    dtype = torch.float16
    
    # Configuration
    batch_size = 1 
    n_heads = 32
    n_heads_k = 32
    head_dim = 128
    block_size = 16 # Page size
    
    # Define segments - Increased sizes for benchmarking
    # Segment 1: Contiguous, length 4096
    seg1_len = 4096
    seg1_k = torch.randn(seg1_len, n_heads_k, head_dim, device=device, dtype=dtype)
    seg1_v = torch.randn(seg1_len, n_heads_k, head_dim, device=device, dtype=dtype)
    
    # Segment 2: Contiguous, length 2048
    seg2_len = 2048
    seg2_k = torch.randn(seg2_len, n_heads_k, head_dim, device=device, dtype=dtype)
    seg2_v = torch.randn(seg2_len, n_heads_k, head_dim, device=device, dtype=dtype)
    
    # Segment 3: Paged, length 1024 (simulating active generation)
    seg3_len = 1024
    num_blocks_seg3 = (seg3_len + block_size - 1) // block_size
    
    # Create Paged KV Cache for Segment 3
    total_blocks = 2000
    k_cache = torch.randn(total_blocks, block_size, n_heads_k, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(total_blocks, block_size, n_heads_k, head_dim, device=device, dtype=dtype)
    
    block_table = torch.arange(num_blocks_seg3, dtype=torch.int32, device=device).unsqueeze(0) # Batch size 1
    
    # Fill cache with "ground truth" data for Seg 3
    seg3_k_flat = torch.randn(seg3_len, n_heads_k, head_dim, device=device, dtype=dtype)
    seg3_v_flat = torch.randn(seg3_len, n_heads_k, head_dim, device=device, dtype=dtype)
    
    for i in range(num_blocks_seg3):
        start = i * block_size
        end = min(start + block_size, seg3_len)
        chunk_len = end - start
        k_cache[i, :chunk_len].copy_(seg3_k_flat[start:end])
        v_cache[i, :chunk_len].copy_(seg3_v_flat[start:end])
        
    # Total Sequence
    total_len = seg1_len + seg2_len + seg3_len
    
    # Query
    q = torch.randn(batch_size, total_len, n_heads, head_dim, device=device, dtype=dtype)
    q_unpad = q.view(-1, n_heads, head_dim)
    
    # Reference Input (Concatenated)
    k_ref = torch.cat([seg1_k, seg2_k, seg3_k_flat], dim=0)
    v_ref = torch.cat([seg1_v, seg2_v, seg3_v_flat], dim=0)
    
    cu_seqlens_q = torch.tensor([0, total_len], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, total_len], dtype=torch.int32, device=device)
    
    # Prepare Segment Args
    segment_lens = torch.tensor([seg1_len, seg2_len, seg3_len], dtype=torch.int32, device=device)
    segment_k_ptrs = torch.tensor([seg1_k.data_ptr(), seg2_k.data_ptr(), 0], dtype=torch.int64, device=device)
    segment_v_ptrs = torch.tensor([seg1_v.data_ptr(), seg2_v.data_ptr(), 0], dtype=torch.int64, device=device)
    
    out_seg = torch.empty_like(q_unpad)

    # Benchmarking
    iterations = 100
    warmup = 10
    
    print(f"\nRunning Benchmark: Total Len={total_len}, Heads={n_heads}, Dim={head_dim}")
    print(f"Segments: {seg1_len} (Contiguous) + {seg2_len} (Contiguous) + {seg3_len} (Paged)")
    
    # 1. Measure Reference (Copy + Attn)
    # We must include copy time because that's what we are optimizing away!
    torch.cuda.synchronize()
    start_ref = time.time()
    for i in range(iterations + warmup):
        if i == warmup:
            torch.cuda.synchronize()
            start_ref = time.time()
            
        # Simulate memory layout transformation (copying segments to contiguous buffer)
        # This is the overhead we want to avoid
        k_ref_bench = torch.cat([seg1_k, seg2_k, seg3_k_flat], dim=0)
        v_ref_bench = torch.cat([seg1_v, seg2_v, seg3_v_flat], dim=0)
        
        flash_attn_varlen_func(
            q_unpad, k_ref_bench, v_ref_bench,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=total_len,
            max_seqlen_k=total_len,
            dropout_p=0.0,
            softmax_scale=1.0
        )
    torch.cuda.synchronize()
    end_ref = time.time()
    avg_ref = (end_ref - start_ref) / iterations * 1000 # ms
    
    # 2. Measure Segmented Attention (No Copy)
    torch.cuda.synchronize()
    start_seg = time.time()
    for i in range(iterations + warmup):
        if i == warmup:
            torch.cuda.synchronize()
            start_seg = time.time()
            
        # No copy needed! Direct pointers.
        torch.ops._vllm_fa2_C.varlen_fwd(
            q_unpad,
            k_cache, v_cache,
            out_seg,
            cu_seqlens_q, cu_seqlens_k,
            None, None, block_table, None,
            total_len, total_len,
            0.0, 1.0, False, False, -1, -1, 0.0, False, 0, None,
            segment_lens, segment_k_ptrs, segment_v_ptrs
        )
    torch.cuda.synchronize()
    end_seg = time.time()
    avg_seg = (end_seg - start_seg) / iterations * 1000 # ms
    
    print(f"Reference Time (incl. copy): {avg_ref:.3f} ms")
    print(f"Segmented Time (no copy):    {avg_seg:.3f} ms")
    print(f"Speedup: {avg_ref / avg_seg:.2f}x")

if __name__ == "__main__":
    test_segmented_attention_performance()
