"""Analyzes PyTorch Kineto Chrome trace files and explains performance in context."""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TraceEvent:
    name: str
    category: str
    ts: int  # microseconds
    dur: int  # microseconds
    pid: int
    tid: int
    args: dict = field(default_factory=dict)

    @property
    def end(self) -> int:
        return self.ts + self.dur

    @property
    def correlation(self) -> int | None:
        return self.args.get("correlation")

    @property
    def external_id(self) -> int | None:
        return self.args.get("External id")

    @property
    def stream(self) -> int | None:
        return self.args.get("stream")

    @property
    def is_kernel(self) -> bool:
        return self.category.lower() in ("kernel",)

    @property
    def is_cpu_op(self) -> bool:
        return self.category.lower() in ("cpu_op", "user_annotation")

    @property
    def is_runtime(self) -> bool:
        return self.category.lower() in ("runtime", "cuda_runtime", "cuda_driver")

    @property
    def is_memcpy(self) -> bool:
        return self.category.lower() in ("memcpy", "gpu_memcpy")

    @property
    def is_memset(self) -> bool:
        return self.category.lower() in ("gpu_memset",)


@dataclass
class GpuGap:
    start: int
    end: int
    stream: int
    before_kernel: str
    after_kernel: str
    cpu_events_during: list[TraceEvent]

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass
class TraceAnalysis:
    total_time_us: int
    gpu_busy_us: int
    gpu_idle_us: int
    gpu_utilization: float
    num_kernels: int
    num_cpu_ops: int
    num_memcpy: int
    kernel_stats: list[dict]
    cpu_op_stats: list[dict]
    gpu_gaps: list[GpuGap]
    streams: set[int]
    sync_events: list[TraceEvent]


def parse_trace(path: str | Path) -> list[TraceEvent]:
    p = Path(path)
    if p.suffix == ".gz" or p.name.endswith(".json.gz"):
        import gzip
        raw = gzip.open(p).read()
    else:
        raw = p.read_bytes()
    data = json.loads(raw)

    # Handle both formats: {"traceEvents": [...]} and [...]
    raw_events = data.get("traceEvents", data) if isinstance(data, dict) else data

    events = []
    for e in raw_events:
        if not isinstance(e, dict):
            continue
        # Only process complete events (ph=X) or duration events (ph=B/E)
        ph = e.get("ph", "")
        if ph not in ("X", "B", "E"):
            continue
        if ph == "X":
            try:
                events.append(TraceEvent(
                    name=e.get("name", ""),
                    category=e.get("cat", ""),
                    ts=int(float(e.get("ts", 0))),
                    dur=int(float(e.get("dur", 0))),
                    pid=int(e.get("pid", 0)) if str(e.get("pid", "0")).isdigit() else 0,
                    tid=int(e.get("tid", 0)) if str(e.get("tid", "0")).isdigit() else 0,
                    args=e.get("args", {}),
                ))
            except (ValueError, TypeError):
                continue
    return events


def analyze_trace(path: str | Path) -> TraceAnalysis:
    events = parse_trace(path)

    kernels = [e for e in events if e.is_kernel]
    cpu_ops = [e for e in events if e.is_cpu_op]
    runtime_calls = [e for e in events if e.is_runtime]
    memcpy_events = [e for e in events if e.is_memcpy]
    sync_events = [e for e in events if "Synchronize" in e.name or "cudaStreamSync" in e.name]

    # Build correlation map: runtime correlation -> cpu op
    ext_id_to_cpu_op = {}
    for op in cpu_ops:
        eid = op.external_id
        if eid is not None:
            ext_id_to_cpu_op[eid] = op

    # Overall timeline
    all_events = kernels + cpu_ops + runtime_calls + memcpy_events
    if not all_events:
        return TraceAnalysis(
            total_time_us=0, gpu_busy_us=0, gpu_idle_us=0, gpu_utilization=0.0,
            num_kernels=0, num_cpu_ops=0, num_memcpy=0,
            kernel_stats=[], cpu_op_stats=[], gpu_gaps=[], streams=set(),
            sync_events=[],
        )

    trace_start = min(e.ts for e in all_events)
    trace_end = max(e.end for e in all_events)
    total_time = trace_end - trace_start

    # GPU utilization (per-stream busy time, merged intervals)
    streams = set()
    gpu_intervals: list[tuple[int, int]] = []
    for e in kernels + memcpy_events:
        gpu_intervals.append((e.ts, e.end))
        if e.stream is not None:
            streams.add(e.stream)

    gpu_intervals.sort()
    merged = []
    for start, end in gpu_intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    gpu_busy = sum(end - start for start, end in merged)
    gpu_idle = total_time - gpu_busy
    gpu_util = gpu_busy / total_time if total_time > 0 else 0.0

    # Kernel stats (aggregate by name)
    kernel_by_name: dict[str, list[TraceEvent]] = {}
    for k in kernels:
        kernel_by_name.setdefault(k.name, []).append(k)

    kernel_stats = []
    for name, ks in sorted(kernel_by_name.items(), key=lambda x: -sum(k.dur for k in x[1])):
        durs = [k.dur for k in ks]
        kernel_stats.append({
            "name": name,
            "count": len(ks),
            "total_us": sum(durs),
            "mean_us": sum(durs) / len(durs),
            "min_us": min(durs),
            "max_us": max(durs),
            "pct_of_gpu": sum(durs) / gpu_busy * 100 if gpu_busy > 0 else 0,
            "grid": ks[0].args.get("grid"),
            "block": ks[0].args.get("block"),
            "registers": ks[0].args.get("registers per thread"),
            "shared_mem": ks[0].args.get("shared memory"),
        })

    # CPU op stats
    cpu_by_name: dict[str, list[TraceEvent]] = {}
    for op in cpu_ops:
        cpu_by_name.setdefault(op.name, []).append(op)

    cpu_op_stats = []
    for name, ops in sorted(cpu_by_name.items(), key=lambda x: -sum(o.dur for o in x[1])):
        durs = [o.dur for o in ops]
        cpu_op_stats.append({
            "name": name,
            "count": len(ops),
            "total_us": sum(durs),
            "mean_us": sum(durs) / len(durs),
        })

    # GPU gaps — find idle periods on each stream
    gpu_gaps = []
    kernels_by_stream: dict[int, list[TraceEvent]] = {}
    for k in kernels + memcpy_events:
        s = k.stream or 0
        kernels_by_stream.setdefault(s, []).append(k)

    for stream, stream_kernels in kernels_by_stream.items():
        stream_kernels.sort(key=lambda k: k.ts)
        for i in range(len(stream_kernels) - 1):
            gap_start = stream_kernels[i].end
            gap_end = stream_kernels[i + 1].ts
            gap_dur = gap_end - gap_start
            if gap_dur <= 0:
                continue

            # Find CPU events happening during this gap
            cpu_during = [
                e for e in cpu_ops + runtime_calls
                if e.ts < gap_end and e.end > gap_start
            ]

            gpu_gaps.append(GpuGap(
                start=gap_start,
                end=gap_end,
                stream=stream,
                before_kernel=stream_kernels[i].name,
                after_kernel=stream_kernels[i + 1].name,
                cpu_events_during=cpu_during,
            ))

    gpu_gaps.sort(key=lambda g: -g.duration)

    return TraceAnalysis(
        total_time_us=total_time,
        gpu_busy_us=gpu_busy,
        gpu_idle_us=gpu_idle,
        gpu_utilization=gpu_util,
        num_kernels=len(kernels),
        num_cpu_ops=len(cpu_ops),
        num_memcpy=len(memcpy_events),
        kernel_stats=kernel_stats,
        cpu_op_stats=cpu_op_stats,
        gpu_gaps=gpu_gaps,
        streams=streams,
        sync_events=sync_events,
    )


def format_analysis(analysis: TraceAnalysis) -> str:
    lines = []
    lines.append("# Trace Analysis")
    lines.append("")
    lines.append(f"Total trace time: {analysis.total_time_us} us")
    lines.append(f"GPU busy: {analysis.gpu_busy_us} us ({analysis.gpu_utilization:.1%})")
    lines.append(f"GPU idle: {analysis.gpu_idle_us} us ({1 - analysis.gpu_utilization:.1%})")
    lines.append(f"Kernels: {analysis.num_kernels}, CPU ops: {analysis.num_cpu_ops}, Memcpy: {analysis.num_memcpy}")
    lines.append(f"GPU streams used: {sorted(analysis.streams)}")

    if analysis.sync_events:
        total_sync = sum(e.dur for e in analysis.sync_events)
        lines.append(f"Sync overhead: {total_sync} us across {len(analysis.sync_events)} calls")

    # Top kernels
    lines.append("")
    lines.append("## Top Kernels by GPU Time")
    lines.append("")
    for i, k in enumerate(analysis.kernel_stats[:10], 1):
        lines.append(
            f"{i}. `{k['name']}` — {k['total_us']} us total "
            f"({k['count']}x, mean {k['mean_us']:.0f} us, {k['pct_of_gpu']:.1f}% of GPU time)"
        )
        details = []
        if k.get("grid"):
            details.append(f"grid={k['grid']}")
        if k.get("block"):
            details.append(f"block={k['block']}")
        if k.get("registers"):
            details.append(f"regs={k['registers']}")
        if k.get("shared_mem"):
            details.append(f"smem={k['shared_mem']}B")
        if details:
            lines.append(f"   {', '.join(details)}")

    # CPU ops
    if analysis.cpu_op_stats:
        lines.append("")
        lines.append("## CPU Operations")
        lines.append("")
        for op in analysis.cpu_op_stats[:10]:
            lines.append(f"- `{op['name']}` — {op['total_us']} us ({op['count']}x)")

    # GPU gaps (bottleneck analysis)
    if analysis.gpu_gaps:
        lines.append("")
        lines.append("## GPU Idle Gaps (Bottlenecks)")
        lines.append("")
        for i, gap in enumerate(analysis.gpu_gaps[:10], 1):
            lines.append(
                f"{i}. **{gap.duration} us idle** on stream {gap.stream} "
                f"between `{gap.before_kernel}` and `{gap.after_kernel}`"
            )

            if gap.cpu_events_during:
                runtime = [e for e in gap.cpu_events_during if e.is_runtime]
                cpu_ops = [e for e in gap.cpu_events_during if e.is_cpu_op]

                if cpu_ops:
                    cpu_names = ", ".join(f"`{e.name}`({e.dur}us)" for e in cpu_ops[:3])
                    lines.append(f"   CPU during gap: {cpu_names}")
                if runtime:
                    rt_names = ", ".join(f"`{e.name}`({e.dur}us)" for e in runtime[:3])
                    lines.append(f"   CUDA runtime: {rt_names}")

                # Diagnose the gap
                total_cpu = sum(e.dur for e in gap.cpu_events_during)
                if total_cpu < gap.duration * 0.3:
                    lines.append(f"   Diagnosis: GPU starved — CPU dispatch accounts for only {total_cpu}us of {gap.duration}us gap")
                elif any("Synchronize" in e.name for e in gap.cpu_events_during):
                    lines.append("   Diagnosis: Explicit sync — CPU waiting for GPU to finish")
                else:
                    lines.append(f"   Diagnosis: CPU overhead — dispatch + framework overhead filling the gap")
            else:
                lines.append("   Diagnosis: No CPU activity during gap — possible stream dependency or sync")

    return "\n".join(lines)


def format_timeline(events: list[TraceEvent], num_segments: int = 10) -> str:
    """Segment the trace into time windows and show GPU utilization per segment."""
    kernels = sorted([e for e in events if e.is_kernel], key=lambda e: e.ts)
    if not kernels:
        return "No GPU kernels found in trace."

    trace_start = kernels[0].ts
    trace_end = max(k.end for k in kernels)
    total = trace_end - trace_start
    if total <= 0:
        return "Trace too short to segment."

    seg_dur = total / num_segments

    lines = []
    lines.append("# Timeline Analysis (GPU utilization over time)")
    lines.append("")
    lines.append(f"Total GPU kernel span: {total} us, {num_segments} segments of ~{seg_dur:.0f} us each")
    lines.append("")

    for seg_idx in range(num_segments):
        seg_start = trace_start + seg_idx * seg_dur
        seg_end = seg_start + seg_dur

        # Find kernels overlapping this segment
        seg_kernels = []
        gpu_busy = 0
        for k in kernels:
            if k.end <= seg_start or k.ts >= seg_end:
                continue
            seg_kernels.append(k)
            overlap_start = max(k.ts, seg_start)
            overlap_end = min(k.end, seg_end)
            gpu_busy += overlap_end - overlap_start

        util = gpu_busy / seg_dur if seg_dur > 0 else 0

        # Build a visual bar
        bar_len = 30
        filled = int(util * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        # Count gaps in this segment
        gaps = 0
        total_gap_dur = 0
        for i in range(len(seg_kernels) - 1):
            gap = seg_kernels[i + 1].ts - seg_kernels[i].end
            if gap > 0:
                gaps += 1
                total_gap_dur += gap

        # Identify dominant kernel type
        kernel_names: dict[str, int] = {}
        for k in seg_kernels:
            kernel_names[k.name] = kernel_names.get(k.name, 0) + k.dur

        dominant = max(kernel_names.items(), key=lambda x: x[1])[0] if kernel_names else "none"
        # Shorten name
        if "nvjet" in dominant.lower():
            dominant_short = "GEMM (nvJet)"
        elif "cudnn" in dominant.lower() and "sdpa" in dominant.lower():
            dominant_short = "Attention (cuDNN SDPA)"
        elif "triton_red" in dominant.lower():
            dominant_short = "Triton reduction"
        elif "triton_poi" in dominant.lower():
            dominant_short = "Triton pointwise"
        elif "triton_per" in dominant.lower():
            dominant_short = "Triton persistent"
        else:
            dominant_short = dominant[:35]

        rel_start = seg_start - trace_start
        lines.append(
            f"  {rel_start:7.0f}-{rel_start + seg_dur:7.0f}us  "
            f"{bar} {util:5.1%}  "
            f"{len(seg_kernels):3d} kernels, {gaps} gaps ({total_gap_dur:.0f}us idle)  "
            f"dominant: {dominant_short}"
        )

    # Identify underutilized segments
    lines.append("")
    underutilized = []
    for seg_idx in range(num_segments):
        seg_start = trace_start + seg_idx * seg_dur
        seg_end = seg_start + seg_dur
        gpu_busy = 0
        for k in kernels:
            if k.end <= seg_start or k.ts >= seg_end:
                continue
            overlap_start = max(k.ts, seg_start)
            overlap_end = min(k.end, seg_end)
            gpu_busy += overlap_end - overlap_start
        util = gpu_busy / seg_dur
        if util < 0.6:
            underutilized.append((seg_idx, util))

    if underutilized:
        lines.append("## Underutilized Segments")
        lines.append("")
        for seg_idx, util in underutilized:
            rel_start = seg_idx * seg_dur
            lines.append(
                f"- Segment {seg_idx + 1} ({rel_start:.0f}-{rel_start + seg_dur:.0f}us): "
                f"only {util:.1%} GPU utilization — likely CPU overhead or kernel launch latency"
            )

    return "\n".join(lines)


def format_bottlenecks(analysis: TraceAnalysis) -> str:
    lines = []
    lines.append("# Performance Bottleneck Analysis")
    lines.append("")

    issues = []

    # 1. Low GPU utilization
    if analysis.gpu_utilization < 0.7:
        severity = "CRITICAL" if analysis.gpu_utilization < 0.4 else "WARNING"
        issues.append({
            "severity": severity,
            "issue": f"Low GPU utilization: {analysis.gpu_utilization:.1%}",
            "detail": (
                f"GPU is idle {analysis.gpu_idle_us}us out of {analysis.total_time_us}us. "
                "The GPU is waiting for work most of the time."
            ),
            "suggestion": (
                "Look at the GPU idle gaps below. Common causes: "
                "CPU-bound dispatch (Python/framework overhead), unnecessary synchronization, "
                "small kernels with high launch overhead."
            ),
        })

    # 2. Many small kernels
    tiny_kernels = [k for k in analysis.kernel_stats if k["mean_us"] < 10]
    if tiny_kernels:
        total_tiny = sum(k["total_us"] for k in tiny_kernels)
        issues.append({
            "severity": "WARNING",
            "issue": f"{len(tiny_kernels)} kernel types averaging <10us",
            "detail": (
                f"Small kernels: {', '.join(k['name'][:40] for k in tiny_kernels[:3])}. "
                f"Total GPU time in tiny kernels: {total_tiny}us. "
                "Kernel launch overhead (~5-10us) dominates for these."
            ),
            "suggestion": "Consider operator fusion (torch.compile, custom CUDA kernel, or CuTe DSL).",
        })

    # 3. Large GPU gaps
    big_gaps = [g for g in analysis.gpu_gaps if g.duration > 50]
    if big_gaps:
        total_gap = sum(g.duration for g in big_gaps)
        issues.append({
            "severity": "CRITICAL" if total_gap > analysis.total_time_us * 0.3 else "WARNING",
            "issue": f"{len(big_gaps)} large GPU idle gaps totaling {total_gap}us",
            "detail": "\n".join(
                f"  - {g.duration}us gap: `{g.before_kernel[:30]}` → `{g.after_kernel[:30]}` on stream {g.stream}"
                for g in big_gaps[:5]
            ),
            "suggestion": (
                "Reduce CPU overhead between kernel launches. "
                "Use CUDA graphs, torch.compile, or overlap CPU work with GPU execution."
            ),
        })

    # 4. Sync overhead
    if analysis.sync_events:
        total_sync = sum(e.dur for e in analysis.sync_events)
        if total_sync > analysis.total_time_us * 0.1:
            issues.append({
                "severity": "WARNING",
                "issue": f"Synchronization overhead: {total_sync}us ({total_sync / analysis.total_time_us:.1%} of total)",
                "detail": (
                    f"{len(analysis.sync_events)} sync calls. "
                    "CPU is blocked waiting for GPU during these calls."
                ),
                "suggestion": (
                    "Reduce explicit syncs. Use async operations where possible. "
                    "Move cudaDeviceSynchronize to the end of the pipeline."
                ),
            })

    # 5. Single stream
    if len(analysis.streams) <= 1 and analysis.num_kernels > 5:
        issues.append({
            "severity": "INFO",
            "issue": "All kernels on a single stream",
            "detail": "No kernel overlap possible when using a single stream.",
            "suggestion": "Consider using multiple CUDA streams for independent operations.",
        })

    if not issues:
        lines.append("No major bottlenecks detected. GPU utilization is healthy.")
    else:
        for issue in issues:
            lines.append(f"### [{issue['severity']}] {issue['issue']}")
            lines.append("")
            lines.append(issue["detail"])
            lines.append("")
            lines.append(f"**Suggestion:** {issue['suggestion']}")
            lines.append("")

    return "\n".join(lines)
