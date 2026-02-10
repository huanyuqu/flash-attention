import torch
import pytest
from vllm_flash_attn import flash_attn_varlen_func


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("segment_lengths", [
    [2048] * 9 + [16],
    [16384, 16],
])
@pytest.mark.parametrize("shuffle_blocks", [False, True])
@pytest.mark.parametrize("num_splits", [0, 1])
@pytest.mark.parametrize("modes", [
    ("prefill",),
    ("decoding",),
    ("prefill", "decoding"),
])
def test_segmented_attention(segment_lengths, shuffle_blocks, 
                             num_splits, modes, 
                             warmup=20, iterations=200):
    device = "cuda"
    dtype = torch.float16
    
    # Configuration
    batch_size = 1
    n_heads = 32
    n_heads_k = 32
    head_dim = 128
    block_size = 16 

    # Validate segment lengths
    for i, l in enumerate(segment_lengths):
        if l % block_size != 0:
            raise ValueError(f"Segment length at index {i} ({l}) is not a multiple of block_size ({block_size})")

    total_len = sum(segment_lengths)
    
    print(f"\nRunning Benchmark: Batch={batch_size}, Total Len={total_len}, Heads={n_heads}, Dim={head_dim}")
    print(f"Segments: {segment_lengths} (Last one is Paged, others Contiguous)")
    print(f"Shuffle Blocks: {shuffle_blocks}")
    if isinstance(modes, str):
        modes = (modes,)
    elif isinstance(modes, list):
        modes = tuple(modes)
    else:
        modes = tuple(modes)

    allowed_modes = {"prefill", "decoding"}
    unknown_modes = [m for m in modes if m not in allowed_modes]
    if unknown_modes:
        raise ValueError(f"Unknown modes: {unknown_modes}. Allowed: {sorted(allowed_modes)}")

    print(f"num_splits: {num_splits}")
    print(f"modes: {list(modes)}")
    
    # Data Preparation
    segments_k = []
    segments_v = []
    segment_ptrs_k_list = []
    segment_ptrs_v_list = []
    
    # Assume last segment is paged
    num_segments = len(segment_lengths)
    paged_segment_idx = num_segments - 1

    for i, seg_len in enumerate(segment_lengths):
        k = torch.randn(batch_size * seg_len, n_heads_k, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size * seg_len, n_heads_k, head_dim, device=device, dtype=dtype)
        segments_k.append(k)
        segments_v.append(v)
        
        if i == paged_segment_idx:
            segment_ptrs_k_list.append(0)
            segment_ptrs_v_list.append(0)
        else:
            segment_ptrs_k_list.append(k.data_ptr())
            segment_ptrs_v_list.append(v.data_ptr())

    # New segmented-attention metadata format (even for batch_size=1):
    # segment_num: [B]
    # segment_lens / segment_{k,v}_ptrs: [B, max_num_segments]
    segment_num_tensor = torch.tensor([num_segments], dtype=torch.int32, device=device)
    max_num_segments = num_segments
    segment_lens_tensor = torch.tensor(segment_lengths, dtype=torch.int32, device=device).view(1, max_num_segments)
    segment_k_ptrs = torch.tensor(segment_ptrs_k_list, dtype=torch.int64, device=device).view(1, max_num_segments)
    segment_v_ptrs = torch.tensor(segment_ptrs_v_list, dtype=torch.int64, device=device).view(1, max_num_segments)

    # Paged KV Cache Setup
    # Calculate total blocks needed for ALL segments (for Paged Attention Benchmark)
    # Note: We calculate exact blocks needed by summing per-segment requirements to avoid underestimation
    total_blocks_required = 0
    for l in segment_lengths:
        total_blocks_required += (l + block_size - 1) // block_size
        
    total_blocks_cache = batch_size * (total_blocks_required + 100)
    
    k_cache = torch.randn(total_blocks_cache, block_size, n_heads_k, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(total_blocks_cache, block_size, n_heads_k, head_dim, device=device, dtype=dtype)
    
    # Prepare Block Tables
    # For Segmented Attention, we only need block table for the paged segment (the last one)
    paged_len = segment_lengths[paged_segment_idx]
    num_blocks_paged = (paged_len + block_size - 1) // block_size
    block_table_paged = torch.empty(batch_size, num_blocks_paged, dtype=torch.int32, device=device)
    
    # For Paged Attention (Benchmark), we need block table for ALL segments
    block_indices_all_list = []
    
    # Simulate fragmentation: Use a random permutation of available blocks
    if shuffle_blocks:
        print(f"Simulating Fragmentation: Randomizing block allocation in cache (Total blocks: {total_blocks_cache})")
        all_available_blocks = torch.randperm(total_blocks_cache, device=device, dtype=torch.int32)
    else:
        print(f"Simulating Fragmentation: Sequential block allocation in cache")
        all_available_blocks = torch.arange(total_blocks_cache, device=device, dtype=torch.int32)
        
    block_alloc_cursor = 0
    
    # We populate the cache and block tables
    for i, seg_len in enumerate(segment_lengths):
        num_blocks = (seg_len + block_size - 1) // block_size
        
        segment_block_indices_per_batch = []
        
        for b in range(batch_size):
            # Allocate blocks randomly from the global pool
            indices = all_available_blocks[block_alloc_cursor : block_alloc_cursor + num_blocks]
            block_alloc_cursor += num_blocks
            
            segment_block_indices_per_batch.append(indices)
            
            # Populate cache
            src_k = segments_k[i]
            src_v = segments_v[i]
            
            # Slice for this batch
            b_src_start = b * seg_len
            b_src_end = (b + 1) * seg_len
            b_k = src_k[b_src_start:b_src_end]
            b_v = src_v[b_src_start:b_src_end]
            
            # Reshape to blocks
            b_k_blocks = b_k.view(num_blocks, block_size, n_heads_k, head_dim)
            b_v_blocks = b_v.view(num_blocks, block_size, n_heads_k, head_dim)
            
            k_cache[indices] = b_k_blocks
            v_cache[indices] = b_v_blocks
            
            if i == paged_segment_idx:
                block_table_paged[b] = indices

        # Stack indices for this segment [batch, num_blocks]
        seg_block_indices = torch.stack(segment_block_indices_per_batch)
        block_indices_all_list.append(seg_block_indices)

    # Full block table for Paged Attention Benchmark
    block_table_full = torch.cat(block_indices_all_list, dim=1)

    # Query
    q_prefill = torch.randn(batch_size, total_len, n_heads, head_dim, device=device, dtype=dtype)
    q_prefill_unpad = q_prefill.view(-1, n_heads, head_dim)
    q_decode = torch.randn(batch_size, 1, n_heads, head_dim, device=device, dtype=dtype)
    q_decode_unpad = q_decode.view(-1, n_heads, head_dim)

    cu_seqlens_q_prefill = torch.arange(0, (batch_size + 1) * total_len, total_len, dtype=torch.int32, device=device)
    cu_seqlens_q_decode = torch.arange(0, (batch_size + 1) * 1, 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * total_len, total_len, dtype=torch.int32, device=device)
    seqused_k = torch.tensor([total_len] * batch_size, dtype=torch.int32, device=device)
    
    # Outputs
    out_prefill_non_paged = torch.empty_like(q_prefill_unpad)
    out_prefill_non_paged_splitkv = torch.empty_like(q_prefill_unpad)
    out_prefill_paged = torch.empty_like(q_prefill_unpad)
    out_prefill_seg = torch.empty_like(q_prefill_unpad)

    out_decode_non_paged = torch.empty_like(q_decode_unpad)
    out_decode_non_paged_splitkv = torch.empty_like(q_decode_unpad)
    out_decode_paged = torch.empty_like(q_decode_unpad)
    out_decode_seg = torch.empty_like(q_decode_unpad)
    
    def time_cuda_ms_per_iter(fn, *, warmup_iters: int, iters: int) -> float:
        for _ in range(warmup_iters):
            fn()
        torch.cuda.synchronize()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(iters):
            fn()
        end_evt.record()
        end_evt.synchronize()

        total_ms = start_evt.elapsed_time(end_evt)
        return total_ms / iters
    
    # Prepare Data (Outside Loop)
    # Construct contiguous reference KV for batch=1
    k_ref = torch.cat(segments_k, dim=0)
    v_ref = torch.cat(segments_v, dim=0)

    def bench_4way(*, mode_name: str, q_unpad, max_seqlen_q: int, cu_seqlens_q, out_non_paged, out_non_paged_splitkv, out_paged, out_seg, report_speedup: bool):
        # 1) Non-Paged (standard kernel)
        def run_non_paged():
            flash_attn_varlen_func(
                q=q_unpad, k=k_ref, v=v_ref,
                max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=total_len, cu_seqlens_k=cu_seqlens_k,
                num_splits=num_splits,
                softmax_scale=1.0, causal=False,
                out=out_non_paged,
            )

        time_non_paged = time_cuda_ms_per_iter(run_non_paged, warmup_iters=warmup, iters=iterations)

        # 2) Non-Paged (SplitKV kernel)
        def run_non_paged_splitkv():
            flash_attn_varlen_func(
                q=q_unpad, k=k_ref, v=v_ref,
                max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=total_len, cu_seqlens_k=cu_seqlens_k,
                force_split_kernel=True,
                num_splits=num_splits,
                softmax_scale=1.0, causal=False,
                out=out_non_paged_splitkv,
            )

        time_non_paged_splitkv = time_cuda_ms_per_iter(run_non_paged_splitkv, warmup_iters=warmup, iters=iterations)

        # 3) Paged
        def run_paged():
            flash_attn_varlen_func(
                q=q_unpad, k=k_cache, v=v_cache,
                max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=total_len, seqused_k=seqused_k,
                block_table=block_table_full,
                num_splits=num_splits,
                softmax_scale=1.0, causal=False,
                out=out_paged,
            )

        time_paged = time_cuda_ms_per_iter(run_paged, warmup_iters=warmup, iters=iterations)

        # 4) Segmented (Hybrid)
        def run_seg():
            flash_attn_varlen_func(
                q=q_unpad, k=k_cache, v=v_cache,
                max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=total_len, seqused_k=seqused_k,
                block_table=block_table_paged,
                segment_num=segment_num_tensor,
                segment_lens=segment_lens_tensor,
                segment_k_ptrs=segment_k_ptrs,
                segment_v_ptrs=segment_v_ptrs,
                num_splits=num_splits,
                softmax_scale=1.0, causal=False,
                out=out_seg,
            )

        time_seg = time_cuda_ms_per_iter(run_seg, warmup_iters=warmup, iters=iterations)

        print("\n" + "=" * 60)
        print(f"Mode: {mode_name}")
        print(f"Shuffle Blocks: {shuffle_blocks}")
        print(f"num_splits: {num_splits}")
        print(f"{'Method':<20} | {'Time (ms)':<18} | {'Relative Speed'}")
        print("-" * 60)
        print(f"{'Non-Paged':<20} | {time_non_paged:<18.3f} | 1.00x (Baseline)")
        print(f"{'Non-Paged(SplitKV)':<20} | {time_non_paged_splitkv:<18.3f} | {time_non_paged / time_non_paged_splitkv:.2f}x")
        print(f"{'Paged':<20} | {time_paged:<18.3f} | {time_non_paged / time_paged:.2f}x")
        print(f"{'Seg(Hybrid)':<20} | {time_seg:<18.3f} | {time_non_paged / time_seg:.2f}x")
        print("=" * 60)
        print("\nAnalysis:")
        if report_speedup:
            print(f"Non-Paged(SplitKV) Speedup:    {(time_non_paged / time_non_paged_splitkv - 1) * 100:.1f}% vs Non-Paged")
            print(f"Segmented(Hybrid) Speedup:     {(time_non_paged / time_seg - 1) * 100:.1f}% vs Non-Paged")
            print(f"Paged Attention Speedup:       {(time_non_paged / time_paged - 1) * 100:.1f}% vs Non-Paged")
        else:
            print(f"Non-Paged(SplitKV) Overhead:   {(time_non_paged_splitkv / time_non_paged - 1) * 100:.1f}% vs Non-Paged")
            print(f"Segmented(Hybrid) Overhead:    {(time_seg / time_non_paged - 1) * 100:.1f}% vs Non-Paged")
            print(f"Paged Attention Overhead:      {(time_paged / time_non_paged - 1) * 100:.1f}% vs Non-Paged")

    if "prefill" in modes:
        bench_4way(
            mode_name="Prefill (seqlen_q = total_len)",
            q_unpad=q_prefill_unpad,
            max_seqlen_q=total_len,
            cu_seqlens_q=cu_seqlens_q_prefill,
            out_non_paged=out_prefill_non_paged,
            out_non_paged_splitkv=out_prefill_non_paged_splitkv,
            out_paged=out_prefill_paged,
            out_seg=out_prefill_seg,
            report_speedup=False,
        )

    if "decoding" in modes:
        bench_4way(
            mode_name="Decode-like (seqlen_q = 1)",
            q_unpad=q_decode_unpad,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q_decode,
            out_non_paged=out_decode_non_paged,
            out_non_paged_splitkv=out_decode_non_paged_splitkv,
            out_paged=out_decode_paged,
            out_seg=out_decode_seg,
            report_speedup=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("segment_lengths", [
    # segment_lengths: List[List[int]] with shape [B][num_segments_b]
    [[2048] * 9 + [16], [2048] * 10 + [16]],
])
@pytest.mark.parametrize("modes", [
    ("prefill",),
    ("decoding",),
    ("prefill", "decoding"),
])
def test_segmented_attention_multibatch(segment_lengths, modes, warmup=20, iterations=200):
    device = "cuda"
    dtype = torch.float16

    batch_size = len(segment_lengths)
    n_heads = 32
    n_heads_k = 32
    head_dim = 128
    block_size = 16

    # Per-request segment lengths. Last segment of each request is paged.
    segment_lens_list = segment_lengths
    total_lens = [sum(x) for x in segment_lens_list]
    max_total_len = max(total_lens)

    for seg_lens in segment_lens_list:
        for l in seg_lens:
            if l % block_size != 0:
                raise ValueError("All segment lengths must be multiples of block_size for this test")

    if isinstance(modes, str):
        modes = (modes,)
    elif isinstance(modes, list):
        modes = tuple(modes)
    else:
        modes = tuple(modes)

    allowed_modes = {"prefill", "decoding"}
    unknown_modes = [m for m in modes if m not in allowed_modes]
    if unknown_modes:
        raise ValueError(f"Unknown modes: {unknown_modes}. Allowed: {sorted(allowed_modes)}")

    # Build per-request dense Q/K/V (for baselines)
    q_dense_list = []
    k_dense_list = []
    v_dense_list = []

    # IMPORTANT: Keep per-segment tensors alive for contiguous segments.
    # segment_{k,v}_ptrs_2d stores raw data_ptr() values; if we don't hold Python references,
    # those tensors can be freed/reused and the pointers become dangling.
    keep_kv_tensors = []

    max_num_segments = max(len(x) for x in segment_lens_list)
    segment_num = torch.tensor([len(x) for x in segment_lens_list], dtype=torch.int32, device=device)
    segment_lens_2d = torch.zeros((batch_size, max_num_segments), dtype=torch.int32, device=device)
    segment_k_ptrs_2d = torch.zeros((batch_size, max_num_segments), dtype=torch.int64, device=device)
    segment_v_ptrs_2d = torch.zeros((batch_size, max_num_segments), dtype=torch.int64, device=device)

    # Prepare a paged cache for ALL segments (for paged baseline), and also a block table for
    # the last segment only (for segmented-hybrid path).
    num_blocks_last = [segment_lens_list[b][-1] // block_size for b in range(batch_size)]
    max_blocks_last = max(num_blocks_last)
    total_blocks_per_req = [sum(seg_len // block_size for seg_len in segment_lens_list[b]) for b in range(batch_size)]
    max_total_blocks = max(total_blocks_per_req)
    total_blocks_cache = sum(total_blocks_per_req) + 8

    k_cache = torch.empty((total_blocks_cache, block_size, n_heads_k, head_dim), device=device, dtype=dtype)
    v_cache = torch.empty((total_blocks_cache, block_size, n_heads_k, head_dim), device=device, dtype=dtype)
    block_table_full = torch.full((batch_size, max_total_blocks), 0, dtype=torch.int32, device=device)
    block_table_paged = torch.full((batch_size, max_blocks_last), 0, dtype=torch.int32, device=device)

    block_cursor = 0
    for b in range(batch_size):
        seg_lens = segment_lens_list[b]
        num_segments = len(seg_lens)
        paged_segment_idx = num_segments - 1

        bt_cursor = 0

        q_b = torch.randn((total_lens[b], n_heads, head_dim), device=device, dtype=dtype)
        q_dense_list.append(q_b)

        k_segs_b = []
        v_segs_b = []
        for si, seg_len in enumerate(seg_lens):
            k_seg = torch.randn((seg_len, n_heads_k, head_dim), device=device, dtype=dtype)
            v_seg = torch.randn((seg_len, n_heads_k, head_dim), device=device, dtype=dtype)
            k_segs_b.append(k_seg)
            v_segs_b.append(v_seg)

            keep_kv_tensors.append((k_seg, v_seg))

            segment_lens_2d[b, si] = seg_len

            nblk = seg_len // block_size
            blk_ids = torch.arange(block_cursor, block_cursor + nblk, device=device, dtype=torch.int32)
            k_cache[blk_ids] = k_seg.view(nblk, block_size, n_heads_k, head_dim)
            v_cache[blk_ids] = v_seg.view(nblk, block_size, n_heads_k, head_dim)
            block_cursor += nblk

            # For paged baseline (all segments): append to per-request full block table
            block_table_full[b, bt_cursor:bt_cursor + nblk] = blk_ids
            bt_cursor += nblk

            if si == paged_segment_idx:
                # For segmented hybrid: last segment is paged -> use block_table_paged and 0 sentinel ptrs
                block_table_paged[b, :nblk] = blk_ids
                segment_k_ptrs_2d[b, si] = 0
                segment_v_ptrs_2d[b, si] = 0
            else:
                # Earlier segments are contiguous -> use direct pointers
                segment_k_ptrs_2d[b, si] = k_seg.data_ptr()
                segment_v_ptrs_2d[b, si] = v_seg.data_ptr()

        k_dense_list.append(torch.cat(k_segs_b, dim=0))
        v_dense_list.append(torch.cat(v_segs_b, dim=0))

    q_ref = torch.cat(q_dense_list, dim=0)
    k_ref = torch.cat(k_dense_list, dim=0)
    v_ref = torch.cat(v_dense_list, dim=0)

    # cu_seqlens_* must have shape (batch_size + 1)
    lens = torch.tensor(total_lens, dtype=torch.int32, device=device)
    cu_seqlens = torch.empty((batch_size + 1,), dtype=torch.int32, device=device)
    cu_seqlens[0] = 0
    cu_seqlens[1:] = torch.cumsum(lens, dim=0)
    cu_seqlens_q = cu_seqlens
    cu_seqlens_k = cu_seqlens
    seqused_k = torch.tensor(total_lens, dtype=torch.int32, device=device)

    # Query for decode-like mode
    q_decode = torch.randn((batch_size, 1, n_heads, head_dim), device=device, dtype=dtype)
    q_decode_unpad = q_decode.view(-1, n_heads, head_dim)
    cu_seqlens_q_decode = torch.arange(0, (batch_size + 1) * 1, 1, dtype=torch.int32, device=device)

    # Pre-allocated outputs to reduce allocator noise
    out_prefill_non_paged = torch.empty_like(q_ref)
    out_prefill_non_paged_splitkv = torch.empty_like(q_ref)
    out_prefill_paged = torch.empty_like(q_ref)
    out_prefill_seg = torch.empty_like(q_ref)

    out_decode_non_paged = torch.empty_like(q_decode_unpad)
    out_decode_non_paged_splitkv = torch.empty_like(q_decode_unpad)
    out_decode_paged = torch.empty_like(q_decode_unpad)
    out_decode_seg = torch.empty_like(q_decode_unpad)

    def time_cuda_ms_per_iter(fn, *, warmup_iters: int, iters: int) -> float:
        for _ in range(warmup_iters):
            fn()
        torch.cuda.synchronize()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(iters):
            fn()
        end_evt.record()
        end_evt.synchronize()
        return start_evt.elapsed_time(end_evt) / iters

    def bench_4way(*, mode_name: str, q_unpad, max_seqlen_q: int, cu_seqlens_q, out_non_paged, out_non_paged_splitkv, out_paged, out_seg, report_speedup: bool):
        # 1) Non-Paged (standard kernel)
        def run_non_paged():
            flash_attn_varlen_func(
                q=q_unpad, k=k_ref, v=v_ref,
                max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=max_total_len, cu_seqlens_k=cu_seqlens_k,
                softmax_scale=1.0, causal=False,
                out=out_non_paged,
            )

        time_non_paged = time_cuda_ms_per_iter(run_non_paged, warmup_iters=warmup, iters=iterations)

        # 2) Non-Paged (SplitKV kernel)
        def run_non_paged_splitkv():
            flash_attn_varlen_func(
                q=q_unpad, k=k_ref, v=v_ref,
                max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=max_total_len, cu_seqlens_k=cu_seqlens_k,
                force_split_kernel=True,
                softmax_scale=1.0, causal=False,
                out=out_non_paged_splitkv,
            )

        time_non_paged_splitkv = time_cuda_ms_per_iter(run_non_paged_splitkv, warmup_iters=warmup, iters=iterations)

        # 3) Paged (all segments paged)
        def run_paged():
            flash_attn_varlen_func(
                q=q_unpad, k=k_cache, v=v_cache,
                max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=max_total_len, seqused_k=seqused_k,
                block_table=block_table_full,
                softmax_scale=1.0, causal=False,
                out=out_paged,
            )

        time_paged = time_cuda_ms_per_iter(run_paged, warmup_iters=warmup, iters=iterations)

        # 4) Segmented (Hybrid)
        def run_seg():
            flash_attn_varlen_func(
                q=q_unpad, k=k_cache, v=v_cache,
                max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=max_total_len, seqused_k=seqused_k,
                block_table=block_table_paged,
                segment_num=segment_num,
                segment_lens=segment_lens_2d,
                segment_k_ptrs=segment_k_ptrs_2d,
                segment_v_ptrs=segment_v_ptrs_2d,
                softmax_scale=1.0, causal=False,
                out=out_seg,
            )

        time_seg = time_cuda_ms_per_iter(run_seg, warmup_iters=warmup, iters=iterations)

        print("\n" + "=" * 60)
        print(f"Mode: {mode_name}")
        print(f"Batch: {batch_size}")
        for b in range(batch_size):
            print(f"Segments b{b}: {segment_lens_list[b]}")
        print(f"{'Method':<20} | {'Time (ms)':<18} | {'Relative Speed'}")
        print("-" * 60)
        print(f"{'Non-Paged':<20} | {time_non_paged:<18.3f} | 1.00x (Baseline)")
        print(f"{'Non-Paged(SplitKV)':<20} | {time_non_paged_splitkv:<18.3f} | {time_non_paged / time_non_paged_splitkv:.2f}x")
        print(f"{'Paged':<20} | {time_paged:<18.3f} | {time_non_paged / time_paged:.2f}x")
        print(f"{'Seg(Hybrid)':<20} | {time_seg:<18.3f} | {time_non_paged / time_seg:.2f}x")
        print("=" * 60)
        print("\nAnalysis:")
        if report_speedup:
            print(f"Non-Paged(SplitKV) Speedup:    {(time_non_paged / time_non_paged_splitkv - 1) * 100:.1f}% vs Non-Paged")
            print(f"Segmented(Hybrid) Speedup:     {(time_non_paged / time_seg - 1) * 100:.1f}% vs Non-Paged")
            print(f"Paged Attention Speedup:       {(time_non_paged / time_paged - 1) * 100:.1f}% vs Non-Paged")
        else:
            print(f"Non-Paged(SplitKV) Overhead:   {(time_non_paged_splitkv / time_non_paged - 1) * 100:.1f}% vs Non-Paged")
            print(f"Segmented(Hybrid) Overhead:    {(time_seg / time_non_paged - 1) * 100:.1f}% vs Non-Paged")
            print(f"Paged Attention Overhead:      {(time_paged / time_non_paged - 1) * 100:.1f}% vs Non-Paged")

    if "prefill" in modes:
        bench_4way(
            mode_name="Prefill (varlen)",
            q_unpad=q_ref,
            max_seqlen_q=max_total_len,
            cu_seqlens_q=cu_seqlens_q,
            out_non_paged=out_prefill_non_paged,
            out_non_paged_splitkv=out_prefill_non_paged_splitkv,
            out_paged=out_prefill_paged,
            out_seg=out_prefill_seg,
            report_speedup=False,
        )

    if "decoding" in modes:
        bench_4way(
            mode_name="Decode-like (seqlen_q = 1)",
            q_unpad=q_decode_unpad,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q_decode,
            out_non_paged=out_decode_non_paged,
            out_non_paged_splitkv=out_decode_non_paged_splitkv,
            out_paged=out_decode_paged,
            out_seg=out_decode_seg,
            report_speedup=True,
        )

if __name__ == "__main__":
    # test_segmented_attention(segment_lengths=[2048] * 9 + [16], 
    #                          shuffle_blocks=True, num_splits=1, 
    #                          modes=("decoding"))
    test_segmented_attention_multibatch(
        segment_lengths=[[2048] * 9 + [16]] * 5,
        modes=("decoding",),
    )
