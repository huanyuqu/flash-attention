import torch
import pytest
import time


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_segmented_attention_performance():
    device = "cuda"
    dtype = torch.float16
    
    # Configuration
    # NOTE: Segmented Attention prototype currently assumes a single set of segment pointers,
    # effectively limiting it to Batch Size = 1 for this test structure.
    batch_size = 1
    n_heads = 32
    n_heads_k = 32
    head_dim = 128
    block_size = 16 
    
    # Define segments
    seg1_len = 4096
    seg2_len = 2048
    seg3_len = 1024
    total_len = seg1_len + seg2_len + seg3_len
    
    print(f"\nRunning Benchmark: Batch={batch_size}, Total Len={total_len}, Heads={n_heads}, Dim={head_dim}")
    print(f"Segments: {seg1_len} (Contiguous) + {seg2_len} (Contiguous) + {seg3_len} (Paged)")
    
    # Data Preparation
    # Segment 1 & 2 (Contiguous)
    seg1_k = torch.randn(batch_size * seg1_len, n_heads_k, head_dim, device=device, dtype=dtype)
    seg1_v = torch.randn(batch_size * seg1_len, n_heads_k, head_dim, device=device, dtype=dtype)
    seg2_k = torch.randn(batch_size * seg2_len, n_heads_k, head_dim, device=device, dtype=dtype)
    seg2_v = torch.randn(batch_size * seg2_len, n_heads_k, head_dim, device=device, dtype=dtype)
    
    # Segment 3 (Paged source data)
    seg3_k_flat = torch.randn(batch_size * seg3_len, n_heads_k, head_dim, device=device, dtype=dtype)
    seg3_v_flat = torch.randn(batch_size * seg3_len, n_heads_k, head_dim, device=device, dtype=dtype)
    
    # Paged KV Cache Setup
    num_blocks_seg3 = (seg3_len + block_size - 1) // block_size
    total_blocks_needed = (total_len + block_size - 1) // block_size
    # Total blocks for all batches
    total_blocks_cache = batch_size * (total_blocks_needed + 100)
    
    k_cache = torch.randn(total_blocks_cache, block_size, n_heads_k, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(total_blocks_cache, block_size, n_heads_k, head_dim, device=device, dtype=dtype)
    
    # Prepare Block Tables
    block_table_seg3 = torch.empty(batch_size, num_blocks_seg3, dtype=torch.int32, device=device)
    
    # Initialize Seg 3 data in cache and block table
    for b in range(batch_size):
        base_block_idx = b * total_blocks_needed 
        for i in range(num_blocks_seg3):
            global_block_idx = base_block_idx + i
            block_table_seg3[b, i] = global_block_idx
            
            # Copy data to populate initial state of cache (not counted in bench)
            src_start = b * seg3_len + i * block_size
            src_end = min(b * seg3_len + (i + 1) * block_size, (b + 1) * seg3_len)
            chunk_len = src_end - src_start
            if chunk_len > 0:
                k_cache[global_block_idx, :chunk_len] = seg3_k_flat[src_start:src_end]
                v_cache[global_block_idx, :chunk_len] = seg3_v_flat[src_start:src_end]

    # Query
    q = torch.randn(batch_size, total_len, n_heads, head_dim, device=device, dtype=dtype)
    q_unpad = q.view(-1, n_heads, head_dim)
    
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * total_len, total_len, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * total_len, total_len, dtype=torch.int32, device=device)
    
    # Outputs
    out_non_paged = torch.empty_like(q_unpad)
    out_paged = torch.empty_like(q_unpad)
    out_seg = torch.empty_like(q_unpad)
    
    # Segmented Args (Specific to Batch=1)
    segment_lens = torch.tensor([seg1_len, seg2_len, seg3_len], dtype=torch.int32, device=device)
    
    # Pointers
    # Note: seg1_k is [batch_size*seg1_len, ...], for batch=1 this is just seg1_len
    segment_k_ptrs = torch.tensor([seg1_k.data_ptr(), seg2_k.data_ptr(), 0], dtype=torch.int64, device=device)
    segment_v_ptrs = torch.tensor([seg1_v.data_ptr(), seg2_v.data_ptr(), 0], dtype=torch.int64, device=device)
    
    
    iterations = 100
    warmup = 10
    
    # =========================================================================
    # 1. Non-Paged Attention Kernel (Baseline)
    # =========================================================================
    
    # Prepare Data (Outside Loop)
    k_ref = torch.cat([seg1_k, seg2_k, seg3_k_flat], dim=0)
    v_ref = torch.cat([seg1_v, seg2_v, seg3_v_flat], dim=0)
    
    # Warmup
    for _ in range(warmup):
         torch.ops._vllm_fa2_C.varlen_fwd(
            q_unpad, k_ref, v_ref, out_non_paged, cu_seqlens_q, cu_seqlens_k, 
            None, None, None, None, total_len, total_len, 0.0, 1.0, False, False, -1, -1, 0.0, False, 0, None, None, None, None
        )

    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iterations):
        # Kernel Only
        torch.ops._vllm_fa2_C.varlen_fwd(
            q_unpad, k_ref, v_ref, out_non_paged, cu_seqlens_q, cu_seqlens_k, 
            None, None, None, None, total_len, total_len, 0.0, 1.0, False, False, -1, -1, 0.0, False, 0, None, None, None, None
        )
    torch.cuda.synchronize()
    t_end = time.time()
    
    time_kernel_non_paged = (t_end - t_start) / iterations * 1000

    # =========================================================================
    # 2. Paged Attention Kernel
    # =========================================================================
    # Pre-calc indices
    num_blocks_seg1 = seg1_len // block_size
    num_blocks_seg2 = seg2_len // block_size
    start_idx_seg1 = num_blocks_seg3
    start_idx_seg2 = start_idx_seg1 + num_blocks_seg1
    block_indices_seg1 = torch.arange(start_idx_seg1, start_idx_seg1 + num_blocks_seg1, device=device, dtype=torch.int32)
    block_indices_seg2 = torch.arange(start_idx_seg2, start_idx_seg2 + num_blocks_seg2, device=device, dtype=torch.int32)
    block_table_full = torch.cat([block_indices_seg1, block_indices_seg2, block_table_seg3.squeeze(0)]).unsqueeze(0)
    
    # Prepare Data (Outside Loop) - Populate Cache
    k_cache[block_indices_seg1] = seg1_k.view(num_blocks_seg1, block_size, n_heads_k, head_dim)
    v_cache[block_indices_seg1] = seg1_v.view(num_blocks_seg1, block_size, n_heads_k, head_dim)
    k_cache[block_indices_seg2] = seg2_k.view(num_blocks_seg2, block_size, n_heads_k, head_dim)
    v_cache[block_indices_seg2] = seg2_v.view(num_blocks_seg2, block_size, n_heads_k, head_dim)
    
    # Warmup
    for _ in range(warmup):
        torch.ops._vllm_fa2_C.varlen_fwd(
            q_unpad, k_cache, v_cache, out_paged, cu_seqlens_q, cu_seqlens_k, 
            None, None, block_table_full, None, total_len, total_len, 0.0, 1.0, False, False, -1, -1, 0.0, False, 0, None, None, None, None
        )

    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iterations):
        # Kernel Only
        torch.ops._vllm_fa2_C.varlen_fwd(
            q_unpad, k_cache, v_cache, out_paged, cu_seqlens_q, cu_seqlens_k, 
            None, None, block_table_full, None, total_len, total_len, 0.0, 1.0, False, False, -1, -1, 0.0, False, 0, None, None, None, None
        )
    torch.cuda.synchronize()
    t_end = time.time()
    
    time_kernel_paged = (t_end - t_start) / iterations * 1000

    # =========================================================================
    # 3. Segmented Attention Kernel
    # =========================================================================
    # Warmup
    for _ in range(warmup):
        torch.ops._vllm_fa2_C.varlen_fwd(
            q_unpad, k_cache, v_cache, out_seg, cu_seqlens_q, cu_seqlens_k, 
            None, None, block_table_seg3, None, total_len, total_len, 0.0, 1.0, False, False, -1, -1, 0.0, False, 0, None, 
            segment_lens, segment_k_ptrs, segment_v_ptrs
        )

    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iterations):
        # Kernel Only
        torch.ops._vllm_fa2_C.varlen_fwd(
            q_unpad, k_cache, v_cache, out_seg, cu_seqlens_q, cu_seqlens_k, 
            None, None, block_table_seg3, None, total_len, total_len, 0.0, 1.0, False, False, -1, -1, 0.0, False, 0, None, 
            segment_lens, segment_k_ptrs, segment_v_ptrs
        )
    torch.cuda.synchronize()
    t_end = time.time()
    
    time_kernel_seg = (t_end - t_start) / iterations * 1000

    # Print Results Table
    print("\n" + "="*60)
    print(f"{'Method':<20} | {'Kernel Time (ms)':<18} | {'Relative Speed'}")
    print("-" * 60)
    print(f"{'Non-Paged':<20} | {time_kernel_non_paged:<18.3f} | 1.00x (Baseline)")
    print(f"{'Paged':<20} | {time_kernel_paged:<18.3f} | {time_kernel_non_paged/time_kernel_paged:.2f}x")
    print(f"{'Segmented':<20} | {time_kernel_seg:<18.3f} | {time_kernel_non_paged/time_kernel_seg:.2f}x")
    print("="*60)
    print("\nAnalysis:")
    print(f"Segmented Attention Overhead: {(time_kernel_seg/time_kernel_non_paged - 1)*100:.1f}% vs Non-Paged")
    print(f"Paged Attention Overhead:     {(time_kernel_paged/time_kernel_non_paged - 1)*100:.1f}% vs Non-Paged")
    
if __name__ == "__main__":
    test_segmented_attention_performance()
