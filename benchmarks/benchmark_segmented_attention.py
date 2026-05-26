"""Benchmark segmented attention vs. non-segmented baseline.

This script sweeps over total token length and contiguous segment length,
measures the forward latency of the segmented-hybrid path relative to a
fully contiguous baseline, and renders a heatmap of the speedup.

The benchmark keeps the same last-segment semantics as the segmented tests:
all segments except the final one are backed by direct pointers, and the
final segment is treated as paged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from vllm_flash_attn import flash_attn_varlen_func


DEFAULT_TOTAL_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
DEFAULT_SEGMENT_LENGTHS = [64, 128, 256, 512, 1024, 2048]


def round_up(value: int, multiple: int) -> int:
	return ((value + multiple - 1) // multiple) * multiple


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


def measure_kernel_ms(fn, *, warmup_iters: int, iters: int) -> float:
	return time_cuda_ms_per_iter(fn, warmup_iters=warmup_iters, iters=iters)


def build_segmented_workload(
	*,
	total_len: int,
	segment_len: int,
	block_size: int,
	batch_size: int,
	n_heads: int,
	n_heads_k: int,
	head_dim: int,
	dtype: torch.dtype,
	device: str,
	shuffle_blocks: bool,
	seed: int,
):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	if total_len < block_size:
		raise ValueError(f"total_len must be >= block_size, got {total_len}")
	if segment_len % block_size != 0:
		raise ValueError(
			f"segment_len must be a multiple of block_size ({block_size}), got {segment_len}"
		)
	if total_len % block_size != 0:
		raise ValueError(
			f"total_len must be a multiple of block_size ({block_size}), got {total_len}"
		)

	num_full_segments = max(0, total_len // segment_len - 1)
	tail_len = total_len - num_full_segments * segment_len
	if tail_len <= 0:
		raise RuntimeError(
			f"Invalid partition for total_len={total_len}, segment_len={segment_len}"
		)
	if tail_len % block_size != 0:
		tail_len = round_up(tail_len, block_size)
		if num_full_segments * segment_len + tail_len != total_len:
			raise RuntimeError(
				"Could not build a valid segmented layout with the requested lengths"
			)

	segment_lengths = [segment_len] * num_full_segments + [tail_len]
	num_segments = len(segment_lengths)
	paged_segment_idx = num_segments - 1

	segments_k = []
	segments_v = []
	segment_ptrs_k_list = []
	segment_ptrs_v_list = []

	for idx, seg_len in enumerate(segment_lengths):
		# Kernel maps processing to kBlockN (up to 128), which can over-read past seg_len
		# allocating slightly larger blocks prevents CUDA illegal memory access
		padded_seg_len = (seg_len + 127) // 128 * 128
		k = torch.randn(
			batch_size * padded_seg_len,
			n_heads_k,
			head_dim,
			device=device,
			dtype=dtype,
		)[:batch_size * seg_len]
		v = torch.randn(
			batch_size * padded_seg_len,
			n_heads_k,
			head_dim,
			device=device,
			dtype=dtype,
		)[:batch_size * seg_len]
		segments_k.append(k)
		segments_v.append(v)

		if idx == paged_segment_idx:
			segment_ptrs_k_list.append(0)
			segment_ptrs_v_list.append(0)
		else:
			segment_ptrs_k_list.append(k.data_ptr())
			segment_ptrs_v_list.append(v.data_ptr())

	segment_num_tensor = torch.tensor([num_segments], dtype=torch.int32, device=device)
	max_num_segments = num_segments
	segment_lens_tensor = torch.tensor(segment_lengths, dtype=torch.int32, device=device).view(
		1, max_num_segments
	)
	segment_k_ptrs = torch.tensor(segment_ptrs_k_list, dtype=torch.int64, device=device).view(
		1, max_num_segments
	)
	segment_v_ptrs = torch.tensor(segment_ptrs_v_list, dtype=torch.int64, device=device).view(
		1, max_num_segments
	)

	total_blocks_required = sum((seg_len + block_size - 1) // block_size for seg_len in segment_lengths)
	total_blocks_cache = batch_size * (total_blocks_required + 32)
	k_cache = torch.randn(
		total_blocks_cache,
		block_size,
		n_heads_k,
		head_dim,
		device=device,
		dtype=dtype,
	)
	v_cache = torch.randn(
		total_blocks_cache,
		block_size,
		n_heads_k,
		head_dim,
		device=device,
		dtype=dtype,
	)

	num_blocks_paged = (tail_len + block_size - 1) // block_size
	block_table_paged = torch.empty(batch_size, num_blocks_paged, dtype=torch.int32, device=device)

	if shuffle_blocks:
		available_blocks = torch.randperm(total_blocks_cache, device=device, dtype=torch.int32)
	else:
		available_blocks = torch.arange(total_blocks_cache, device=device, dtype=torch.int32)

	block_alloc_cursor = 0
	block_indices_all_list = []
	keep_alive_kv = []

	for seg_idx, seg_len in enumerate(segment_lengths):
		num_blocks = (seg_len + block_size - 1) // block_size
		segment_block_indices_per_batch = []

		for batch_idx in range(batch_size):
			indices = available_blocks[block_alloc_cursor : block_alloc_cursor + num_blocks]
			block_alloc_cursor += num_blocks
			segment_block_indices_per_batch.append(indices)

			src_k = segments_k[seg_idx]
			src_v = segments_v[seg_idx]
			start = batch_idx * seg_len
			end = (batch_idx + 1) * seg_len
			b_k = src_k[start:end]
			b_v = src_v[start:end]

			b_k_blocks = b_k.view(num_blocks, block_size, n_heads_k, head_dim)
			b_v_blocks = b_v.view(num_blocks, block_size, n_heads_k, head_dim)
			k_cache[indices] = b_k_blocks
			v_cache[indices] = b_v_blocks

			if seg_idx == paged_segment_idx:
				block_table_paged[batch_idx] = indices

		block_indices_all_list.append(torch.stack(segment_block_indices_per_batch))

	block_table_full = torch.cat(block_indices_all_list, dim=1)
	keep_alive_kv.extend(zip(segments_k, segments_v))

	q_contiguous = torch.randn(
		batch_size,
		total_len,
		n_heads,
		head_dim,
		device=device,
		dtype=dtype,
	)
	q_contiguous_unpad = q_contiguous.view(-1, n_heads, head_dim)

	cu_seqlens_q = torch.arange(
		0,
		(batch_size + 1) * total_len,
		total_len,
		dtype=torch.int32,
		device=device,
	)
	cu_seqlens_k = cu_seqlens_q
	seqused_k = torch.tensor([total_len] * batch_size, dtype=torch.int32, device=device)

	return {
		"segment_lengths": segment_lengths,
		"tail_len": tail_len,
		"num_full_segments": num_full_segments,
		"q_contiguous_unpad": q_contiguous_unpad,
		"k_ref": torch.cat(segments_k, dim=0),
		"v_ref": torch.cat(segments_v, dim=0),
		"k_cache": k_cache,
		"v_cache": v_cache,
		"block_table_full": block_table_full,
		"block_table_paged": block_table_paged,
		"segment_num_tensor": segment_num_tensor,
		"segment_lens_tensor": segment_lens_tensor,
		"segment_k_ptrs": segment_k_ptrs,
		"segment_v_ptrs": segment_v_ptrs,
		"cu_seqlens_q": cu_seqlens_q,
		"cu_seqlens_k": cu_seqlens_k,
		"seqused_k": seqused_k,
		"keep_alive_kv": keep_alive_kv,
	}


def benchmark_single_variant(
	*,
	total_len: int,
	segment_len: int,
	block_size: int,
	batch_size: int,
	n_heads: int,
	n_heads_k: int,
	head_dim: int,
	dtype: torch.dtype,
	device: str,
	warmup: int,
	iterations: int,
	shuffle_blocks: bool,
	mode: str,
	kernel: str,
	seed: int,
) -> dict[str, float | int | list[int]]:
	workload = build_segmented_workload(
		total_len=total_len,
		segment_len=segment_len,
		block_size=block_size,
		batch_size=batch_size,
		n_heads=n_heads,
		n_heads_k=n_heads_k,
		head_dim=head_dim,
		dtype=dtype,
		device=device,
		shuffle_blocks=shuffle_blocks,
		seed=seed,
	)

	q_contiguous_unpad = workload["q_contiguous_unpad"]
	k_cache = workload["k_cache"]
	v_cache = workload["v_cache"]
	block_table_full = workload["block_table_full"]
	block_table_paged = workload["block_table_paged"]
	segment_num_tensor = workload["segment_num_tensor"]
	segment_lens_tensor = workload["segment_lens_tensor"]
	segment_k_ptrs = workload["segment_k_ptrs"]
	segment_v_ptrs = workload["segment_v_ptrs"]
	cu_seqlens_q = workload["cu_seqlens_q"]
	cu_seqlens_k = workload["cu_seqlens_k"]
	seqused_k = workload["seqused_k"]

	if mode == "decoding":
		q = torch.randn(batch_size, 1, n_heads, head_dim, device=device, dtype=dtype)
		q_unpad = q.view(-1, n_heads, head_dim)
		cu_seqlens_q_mode = torch.arange(0, (batch_size + 1) * 1, 1, dtype=torch.int32, device=device)
		max_seqlen_q = 1
		out_shape = q_unpad.shape
	elif mode == "prefill":
		q_unpad = q_contiguous_unpad
		cu_seqlens_q_mode = cu_seqlens_q
		max_seqlen_q = total_len
		out_shape = q_unpad.shape
	else:
		raise ValueError(f"Unsupported mode: {mode}")

	out_paged = torch.empty(out_shape, device=device, dtype=dtype)
	out_segmented = torch.empty(out_shape, device=device, dtype=dtype)

	def run_paged():
		flash_attn_varlen_func(
			q=q_unpad,
			k=k_cache,
			v=v_cache,
			max_seqlen_q=max_seqlen_q,
			cu_seqlens_q=cu_seqlens_q_mode,
			max_seqlen_k=total_len,
			seqused_k=seqused_k,
			block_table=block_table_full,
			num_splits=0,
			softmax_scale=1.0,
			causal=False,
			out=out_paged,
		)

	def run_segmented():
		flash_attn_varlen_func(
			q=q_unpad,
			k=k_cache,
			v=v_cache,
			max_seqlen_q=max_seqlen_q,
			cu_seqlens_q=cu_seqlens_q_mode,
			max_seqlen_k=total_len,
			seqused_k=seqused_k,
			block_table=block_table_paged,
			segment_num=segment_num_tensor,
			segment_lens=segment_lens_tensor,
			segment_k_ptrs=segment_k_ptrs,
			segment_v_ptrs=segment_v_ptrs,
			num_splits=0,
			softmax_scale=1.0,
			causal=False,
			out=out_segmented,
		)

	if kernel == "paged":
		run_kernel = run_paged
		other_kernel = "segmented"
	elif kernel == "segmented":
		run_kernel = run_segmented
		other_kernel = "paged"
	else:
		raise ValueError(f"Unsupported kernel: {kernel}")

	measured_ms = measure_kernel_ms(run_kernel, warmup_iters=warmup, iters=iterations)

	return {
		"total_len": total_len,
		"segment_len": segment_len,
		"kernel": kernel,
		"other_kernel": other_kernel,
		"ms": measured_ms,
		"segment_layout": workload["segment_lengths"],
		"tail_len": workload["tail_len"],
		"num_full_segments": workload["num_full_segments"],
	}


def run_point_subprocess(
	*,
	total_len: int,
	segment_len: int,
	block_size: int,
	batch_size: int,
	n_heads: int,
	n_heads_k: int,
	head_dim: int,
	warmup: int,
	iterations: int,
	shuffle_blocks: bool,
	mode: str,
	kernel: str,
	seed: int,
) -> dict[str, float | int | list[int]]:
	with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
		result_path = Path(tmp_file.name)

	cmd = [
		sys.executable,
		str(Path(__file__).resolve()),
		"--worker",
		"--kernel",
		kernel,
		"--result-path",
		str(result_path),
		"--total-lengths",
		str(total_len),
		"--segment-lengths",
		str(segment_len),
		"--block-size",
		str(block_size),
		"--batch-size",
		str(batch_size),
		"--n-heads",
		str(n_heads),
		"--n-heads-k",
		str(n_heads_k),
		"--head-dim",
		str(head_dim),
		"--warmup",
		str(warmup),
		"--iterations",
		str(iterations),
		"--mode",
		mode,
		"--seed",
		str(seed),
	]
	if shuffle_blocks:
		cmd.append("--shuffle-blocks")

	completed = subprocess.run(cmd, capture_output=True, text=True)
	try:
		if completed.returncode != 0:
			raise RuntimeError(
				"Subprocess benchmark failed for "
				f"kernel={kernel}, total_len={total_len}, segment_len={segment_len}\n"
				f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
			)
		return json.loads(result_path.read_text(encoding="utf-8"))
	finally:
		result_path.unlink(missing_ok=True)


def benchmark_one_point(
	*,
	total_len: int,
	segment_len: int,
	block_size: int,
	batch_size: int,
	n_heads: int,
	n_heads_k: int,
	head_dim: int,
	dtype: torch.dtype,
	device: str,
	warmup: int,
	iterations: int,
	rounds: int,
	shuffle_blocks: bool,
	mode: str,
) -> dict[str, float | int | list[int]]:
	if rounds <= 0:
		raise ValueError("rounds must be positive")

	seed_base = total_len * 1009 + segment_len
	times = {"paged": [], "segmented": []}
	representative: dict[str, float | int | list[int]] | None = None

	for round_idx in range(rounds):
		seed = seed_base + round_idx * 10007
		order = ["paged", "segmented"]
		if round_idx % 2 == 1:
			order.reverse()

		for kernel in order:
			result = run_point_subprocess(
				total_len=total_len,
				segment_len=segment_len,
				block_size=block_size,
				batch_size=batch_size,
				n_heads=n_heads,
				n_heads_k=n_heads_k,
				head_dim=head_dim,
				warmup=warmup,
				iterations=iterations,
				shuffle_blocks=shuffle_blocks,
				mode=mode,
				kernel=kernel,
				seed=seed,
			)
			times[kernel].append(float(result["ms"]))
			if representative is None:
				representative = result

	if representative is None:
		raise RuntimeError("No measurements were collected")

	paged_ms = sum(times["paged"]) / len(times["paged"])
	segmented_ms = sum(times["segmented"]) / len(times["segmented"])
	speedup = paged_ms / segmented_ms
	slowdown_pct = (segmented_ms / paged_ms - 1.0) * 100.0

	return {
		"total_len": total_len,
		"segment_len": segment_len,
		"speedup": speedup,
		"slowdown_pct": slowdown_pct,
		"paged_ms": paged_ms,
		"segmented_ms": segmented_ms,
		"segment_layout": representative["segment_layout"],
		"tail_len": representative["tail_len"],
		"num_full_segments": representative["num_full_segments"],
		"rounds": rounds,
	}


def render_heatmap(
	*,
	matrix: list[list[float]],
	total_lengths: Sequence[int],
	segment_lengths: Sequence[int],
	output_path: Path,
	title: str,
	annotate: bool,
):
	fig, ax = plt.subplots(figsize=(1.2 * max(6, len(total_lengths)), 0.9 * max(5, len(segment_lengths))))
	image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
	ax.set_xticks(range(len(total_lengths)))
	ax.set_xticklabels([str(value) for value in total_lengths], rotation=45, ha="right", fontsize=15)
	ax.set_yticks(range(len(segment_lengths)))
	ax.set_yticklabels([str(value) for value in segment_lengths], fontsize=15)
	ax.set_xlabel("Total tokens per request", fontsize=15)
	ax.set_ylabel("Contiguous segment length", fontsize=15)
	ax.tick_params(axis="both", which="major", labelsize=15)

	colorbar = fig.colorbar(image, ax=ax)
	colorbar.set_label("Speedup vs paged (x)", fontsize=15)
	colorbar.ax.tick_params(labelsize=15)

	if annotate:
		threshold = (max(max(row) for row in matrix) + min(min(row) for row in matrix)) / 2.0
		for row_idx, row in enumerate(matrix):
			for col_idx, value in enumerate(row):
				text_color = "white" if value < threshold else "black"
				ax.text(
					col_idx,
					row_idx,
					f"{value:.2f}x",
					ha="center",
					va="center",
					color=text_color,
					fontsize=15,
				)

	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def run_benchmark(
	*,
	total_lengths: Sequence[int],
	segment_lengths: Sequence[int],
	output_path: Path,
	mode: str,
	batch_size: int,
	n_heads: int,
	n_heads_k: int,
	head_dim: int,
	block_size: int,
	warmup: int,
	iterations: int,
	rounds: int,
	shuffle_blocks: bool,
	annotate: bool,
):
	device = "cuda"
	dtype = torch.float16

	speedup_matrix: list[list[float]] = []
	slowdown_matrix: list[list[float]] = []
	results: list[dict[str, float | int | list[int]]] = []

	for segment_len in segment_lengths:
		speedup_row = []
		slowdown_row = []
		for total_len in total_lengths:
			result = benchmark_one_point(
				total_len=total_len,
				segment_len=segment_len,
				block_size=block_size,
				batch_size=batch_size,
				n_heads=n_heads,
				n_heads_k=n_heads_k,
				head_dim=head_dim,
				dtype=dtype,
				device=device,
				warmup=warmup,
				iterations=iterations,
				rounds=rounds,
				shuffle_blocks=shuffle_blocks,
				mode=mode,
			)
			results.append(result)
			speedup_row.append(float(result["speedup"]))
			slowdown_row.append(float(result["slowdown_pct"]))

			print(
				f"total_len={total_len:>6}, segment_len={segment_len:>4}, "
				f"layout={result['segment_layout']}, speedup={result['speedup']:.3f}x, "
				f"segmented={result['segmented_ms']:.3f} ms, paged={result['paged_ms']:.3f} ms"
			)

		speedup_matrix.append(speedup_row)
		slowdown_matrix.append(slowdown_row)

	title = f"Segmented attention speedup vs Paged ({mode})"
	render_heatmap(
		matrix=speedup_matrix,
		total_lengths=total_lengths,
		segment_lengths=segment_lengths,
		output_path=output_path,
		title=title,
		annotate=annotate,
	)

	summary_path = output_path.with_suffix(".json")
	summary = {
		"mode": mode,
		"batch_size": batch_size,
		"n_heads": n_heads,
		"n_heads_k": n_heads_k,
		"head_dim": head_dim,
		"block_size": block_size,
		"rounds": rounds,
		"total_lengths": list(total_lengths),
		"segment_lengths": list(segment_lengths),
		"speedup_matrix": speedup_matrix,
		"slowdown_matrix_pct": slowdown_matrix,
		"results": results,
	}
	summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	print(f"\nSaved heatmap to: {output_path}")
	print(f"Saved raw results to: {summary_path}")


def build_argparser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Benchmark segmented attention and render a heatmap vs. the Paged baseline.",
	)
	parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
	parser.add_argument("--kernel", choices=("paged", "segmented"), help=argparse.SUPPRESS)
	parser.add_argument("--result-path", type=Path, help=argparse.SUPPRESS)
	parser.add_argument(
		"--total-lengths",
		nargs="+",
		type=int,
		default=DEFAULT_TOTAL_LENGTHS,
		help="Total token lengths to sweep.",
	)
	parser.add_argument(
		"--segment-lengths",
		nargs="+",
		type=int,
		default=DEFAULT_SEGMENT_LENGTHS,
		help="Contiguous segment lengths to sweep.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("benchmarks/segmented_attention_heatmap.png"),
		help="Output PNG path for the heatmap.",
	)
	parser.add_argument(
		"--mode",
		choices=("prefill", "decoding"),
		default="decoding",
		help="Which query shape to benchmark.",
	)
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument("--n-heads", type=int, default=32)
	parser.add_argument("--n-heads-k", type=int, default=32)
	parser.add_argument("--head-dim", type=int, default=128)
	parser.add_argument("--block-size", type=int, default=16)
	parser.add_argument("--warmup", type=int, default=20)
	parser.add_argument("--iterations", type=int, default=200)
	parser.add_argument(
		"--rounds",
		type=int,
		default=1,
		help="Independent subprocess rounds per point.",
	)
	parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
	parser.add_argument(
		"--shuffle-blocks",
		action="store_true",
		default=True,
		help="Randomize block allocation to simulate fragmented caches.",
	)
	parser.add_argument(
		"--annotate",
		action="store_true",
		default=True,
		help="Draw the speedup value in each heatmap cell.",
	)
	return parser


def run_worker(args: argparse.Namespace) -> None:
	if args.result_path is None:
		raise SystemExit("--result-path is required in worker mode")
	if args.kernel is None:
		raise SystemExit("--kernel is required in worker mode")

	result = benchmark_single_variant(
		total_len=args.total_lengths[0],
		segment_len=args.segment_lengths[0],
		block_size=args.block_size,
		batch_size=args.batch_size,
		n_heads=args.n_heads,
		n_heads_k=args.n_heads_k,
		head_dim=args.head_dim,
		dtype=torch.float16,
		device="cuda",
		warmup=args.warmup,
		iterations=args.iterations,
		shuffle_blocks=args.shuffle_blocks,
		mode=args.mode,
		kernel=args.kernel,
		seed=args.seed,
	)
	args.result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
	parser = build_argparser()
	args = parser.parse_args(argv)

	if not torch.cuda.is_available():
		raise SystemExit("CUDA is not available")

	if args.worker:
		run_worker(args)
		return

	total_lengths = sorted(set(args.total_lengths))
	segment_lengths = sorted(set(args.segment_lengths))

	if not total_lengths:
		raise SystemExit("--total-lengths cannot be empty")
	if not segment_lengths:
		raise SystemExit("--segment-lengths cannot be empty")

	for value in total_lengths + segment_lengths:
		if value % args.block_size != 0:
			raise SystemExit(
				f"All lengths must be multiples of block_size={args.block_size}, got {value}"
			)

	run_benchmark(
		total_lengths=total_lengths,
		segment_lengths=segment_lengths,
		output_path=args.output,
		mode=args.mode,
		batch_size=args.batch_size,
		n_heads=args.n_heads,
		n_heads_k=args.n_heads_k,
		head_dim=args.head_dim,
		block_size=args.block_size,
		warmup=args.warmup,
		iterations=args.iterations,
		rounds=args.rounds,
		shuffle_blocks=args.shuffle_blocks,
		annotate=args.annotate,
	)


if __name__ == "__main__":
	main()
