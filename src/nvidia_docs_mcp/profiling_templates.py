"""Intra-kernel profiling templates for CuTe DSL kernels.

Provides code templates and guidance for instrumenting GPU kernels
with cycle-level timing using PTX special registers (%clock64, %globaltimer).
"""

INTRA_KERNEL_OVERVIEW = """# Intra-Kernel Profiling with CuTe DSL

## How It Works

PTX provides two timing mechanisms accessible from inside a kernel:
- `%clock64` — per-SM cycle counter (high resolution, SM-local)
- `%globaltimer` — system-wide nanosecond timer (lower resolution, global)

CuTe DSL supports inline PTX via `llvm.inline_asm`, so you can read these
registers at any point in your kernel to measure how long each stage takes.

## Typical Stages to Profile in a GEMM Kernel

1. **Prologue** — TMA loads, shared memory setup, pipeline init
2. **Mainloop MMA** — wgmma.mma_async / tcgen05.mma iterations
3. **Mainloop Copy** — cp.async.bulk.tensor / TMA loads for next tile
4. **Epilogue** — accumulator store, type conversion, output write

## Important Notes

- `%clock64` counts SM clock cycles, NOT wall time. Divide by SM clock frequency to get time.
- Reading `%clock64` is cheap (~4 cycles) but can affect instruction scheduling.
- Only profile with a single warp/thread to avoid perturbation.
- Use `elect_one()` to have only one thread do the timing.
- Write timestamps to a global memory buffer for host readback.
"""

CLOCK_READ_TEMPLATE = """# Reading %clock64 in CuTe DSL

```python
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

@dsl_user_op
def read_clock64():
    \"\"\"Read the per-SM 64-bit cycle counter via inline PTX.\"\"\"
    result = llvm.inline_asm(
        llvm.i64,           # return type
        [],                 # no inputs
        "mov.u64 $0, %clock64;",
        "=l",               # output constraint: 64-bit register
        has_side_effects=True,
        is_align_stack=False,
    )
    return result

@dsl_user_op
def read_globaltimer():
    \"\"\"Read the system-wide nanosecond timer via inline PTX.\"\"\"
    result = llvm.inline_asm(
        llvm.i64,
        [],
        "mov.u64 $0, %globaltimer;",
        "=l",
        has_side_effects=True,
        is_align_stack=False,
    )
    return result
```
"""

GEMM_PROFILING_TEMPLATE = """# Profiling a CuTe DSL GEMM Kernel

This template shows how to add cycle-level timing to a Blackwell GEMM kernel.
The timing data is written to a `timestamps` tensor for host readback.

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int64, Int32
from cutlass.cute.arch import elect_one

# Define timing helpers (see read_clock64 above)
# ...

@cute.kernel(grid=grid, block=block, smem=smem_bytes)
def profiled_gemm(
    A: cute.Tensor, B: cute.Tensor, D: cute.Tensor,
    timestamps: cute.Tensor,   # shape: (num_tiles, 6) for 6 timing points
    # ... other args
):
    # Determine if this thread should record timestamps
    # Only one thread per CTA writes timing data
    with elect_one():
        tile_idx = cute.arch.block_idx()[0]

        # === TIMING POINT 0: Kernel start ===
        t0 = read_clock64()

        # ... prologue: allocate smem, init pipeline, TMA setup ...

        # === TIMING POINT 1: After prologue ===
        t1 = read_clock64()

    # ... mainloop: MMA + copy pipeline ...
    for k_tile in cutlass.range(num_k_tiles):
        # producer: TMA load
        # consumer: wgmma.mma_async

        pass  # your mainloop here

    with elect_one():
        # === TIMING POINT 2: After mainloop ===
        t2 = read_clock64()

    # ... epilogue: store results ...

    with elect_one():
        # === TIMING POINT 3: After epilogue ===
        t3 = read_clock64()

        # Write timestamps to global memory
        # timestamps[tile_idx, 0] = t0  (kernel start)
        # timestamps[tile_idx, 1] = t1  (after prologue)
        # timestamps[tile_idx, 2] = t2  (after mainloop)
        # timestamps[tile_idx, 3] = t3  (after epilogue)
```

## Host-Side Analysis

```python
import torch

# After kernel execution:
ts = timestamps.cpu().numpy()  # shape: (num_tiles, 4)

# Per-tile breakdown (in cycles)
prologue_cycles = ts[:, 1] - ts[:, 0]
mainloop_cycles = ts[:, 2] - ts[:, 1]
epilogue_cycles = ts[:, 3] - ts[:, 2]
total_cycles = ts[:, 3] - ts[:, 0]

print(f"Prologue: {prologue_cycles.mean():.0f} cycles ({prologue_cycles.mean()/total_cycles.mean()*100:.1f}%)")
print(f"Mainloop: {mainloop_cycles.mean():.0f} cycles ({mainloop_cycles.mean()/total_cycles.mean()*100:.1f}%)")
print(f"Epilogue: {epilogue_cycles.mean():.0f} cycles ({epilogue_cycles.mean()/total_cycles.mean()*100:.1f}%)")

# Convert to time (need SM clock freq, e.g., 1.98 GHz for H100)
sm_clock_ghz = 1.98  # H100 SM clock
prologue_us = prologue_cycles.mean() / (sm_clock_ghz * 1000)
mainloop_us = mainloop_cycles.mean() / (sm_clock_ghz * 1000)
epilogue_us = epilogue_cycles.mean() / (sm_clock_ghz * 1000)
print(f"Prologue: {prologue_us:.1f} us, Mainloop: {mainloop_us:.1f} us, Epilogue: {epilogue_us:.1f} us")
```
"""

PIPELINE_PROFILING_TEMPLATE = """# Profiling Pipeline Stages (MMA vs Copy Overlap)

For warp-specialized kernels, you want to measure how well MMA and copy overlap.
Profile the producer (TMA copy) and consumer (MMA) separately.

```python
@cute.kernel(grid=grid, block=block, smem=smem_bytes)
def profiled_warp_specialized_gemm(
    A, B, D, timestamps,
    # ...
):
    warp_id = cute.arch.warp_idx()

    if warp_id == 0:
        # Producer warp (TMA loads)
        for stage in cutlass.range(num_stages):
            with elect_one():
                t_copy_start = read_clock64()

            # TMA copy for this stage
            # ...

            with elect_one():
                t_copy_end = read_clock64()
                # Record copy timing
    else:
        # Consumer warps (MMA)
        for stage in cutlass.range(num_stages):
            with elect_one():
                t_mma_start = read_clock64()

            # wgmma.mma_async / tcgen05.mma
            # ...

            with elect_one():
                t_mma_end = read_clock64()
                # Record MMA timing
```

## What to Look For

- **MMA >> Copy**: Good overlap. Copy is hidden behind MMA.
- **Copy >> MMA**: Memory bound. Need more pipeline stages or better TMA utilization.
- **MMA ≈ Copy**: Balanced. Close to optimal for this problem size.
- **Large gaps between stages**: Pipeline bubble. Check barrier/mbarrier synchronization.
"""

MBARRIER_PROFILING_TEMPLATE = """# Profiling Barrier Wait Times

Barrier waits are often where performance is lost. Measure how long
threads wait at mbarrier_wait/mbarrier_try_wait.

```python
with elect_one():
    t_before_wait = read_clock64()

# Wait for producer to finish loading
cute.arch.mbarrier_wait(mbar_ptr, phase)

with elect_one():
    t_after_wait = read_clock64()
    wait_cycles = t_after_wait - t_before_wait
    # If wait_cycles is large, the producer is slower than the consumer
```

## Interpreting Results

- **wait_cycles ≈ 0**: No wait, producer is ahead of consumer. Good.
- **wait_cycles > 100**: Producer is behind. Possible causes:
  - TMA bandwidth saturation
  - Shared memory bank conflicts in the copy layout
  - Not enough pipeline stages to hide latency
  - Clock frequency throttling (thermal)
"""

PTX_TIMING_REGISTERS = """# PTX Special Registers for Timing

## %clock / %clock64
- Per-SM cycle counter
- Resolution: 1 SM clock cycle
- Wraps: %clock wraps at 2^32, %clock64 wraps at 2^64
- PTX: `mov.u64 reg, %clock64;`
- Architecture: Available on all SM architectures

## %globaltimer
- System-wide nanosecond timer
- Resolution: ~1 nanosecond (but actual precision varies)
- Synchronized across all SMs
- PTX: `mov.u64 reg, %globaltimer;`
- Architecture: sm_30+

## When to Use Which

| Register | Use case | Pros | Cons |
|---|---|---|---|
| %clock64 | Intra-SM timing (MMA, copy stages) | High resolution, low overhead | SM-local (can't compare across SMs) |
| %globaltimer | Cross-SM timing, wall clock | Global, comparable across SMs | Lower resolution, may have jitter |

## nanosleep (sm_70+)
- `nanosleep.u32 reg;` — sleep for approximately `reg` nanoseconds
- Useful for controlled delay injection when debugging pipeline stalls
"""

NSIGHT_COMPUTE_METRICS = """# Key Nsight Compute Metrics for Kernel Analysis

When using `nsight-python` with `@analyze.kernel`, these are the most useful metrics:

## Performance Metrics
- `gpu__time_duration.sum` — Total kernel execution time (default)
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — SM throughput utilization
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` — Memory throughput

## Occupancy
- `sm__warps_active.avg.pct_of_peak_sustained_active` — Achieved occupancy
- `sm__maximum_warps_per_active_cycle_pct` — Theoretical max occupancy
- `launch__occupancy_limit_blocks` — Block limit on occupancy
- `launch__occupancy_limit_registers` — Register limit on occupancy
- `launch__occupancy_limit_shared_mem` — Shared memory limit on occupancy
- `launch__occupancy_limit_warps` — Warp limit on occupancy

## Warp Stalls (why warps are waiting)
- `smsp__warps_issue_stalled_barrier.avg` — Stalled on barrier
- `smsp__warps_issue_stalled_membar.avg` — Stalled on memory barrier
- `smsp__warps_issue_stalled_long_scoreboard.avg` — Waiting for L2/DRAM
- `smsp__warps_issue_stalled_short_scoreboard.avg` — Waiting for L1/shared mem
- `smsp__warps_issue_stalled_math_pipe_throttle.avg` — Math pipe full
- `smsp__warps_issue_stalled_mio_throttle.avg` — Memory I/O throttle
- `smsp__warps_issue_stalled_wait.avg` — General wait
- `smsp__warps_issue_stalled_not_selected.avg` — Eligible but not selected

## Memory
- `dram__bytes_read.sum` — DRAM bytes read
- `dram__bytes_write.sum` — DRAM bytes written
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` — L1 global load bytes
- `lts__t_bytes.sum` — L2 total bytes
- `sm__sass_data_bytes_mem_shared.sum` — Shared memory bytes

## Tensor Core
- `sm__pipe_tensor_op_hmma_cycles_active.avg` — Tensor core active cycles (Hopper)
- `smsp__inst_executed_pipe_tensor_op_hmma.sum` — Tensor core instructions executed

## Usage with nsight-python

```python
from nsight import analyze

@analyze.kernel(
    metrics=[
        "gpu__time_duration.sum",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__warps_issue_stalled_long_scoreboard.avg",
        "smsp__warps_issue_stalled_barrier.avg",
        "dram__bytes_read.sum",
        "sm__pipe_tensor_op_hmma_cycles_active.avg",
    ],
    runs=10,
    thermal_mode="auto",
)
def my_kernel_benchmark(config):
    # Launch your kernel here
    my_kernel(A, B, D, ...)
```
"""

NSIGHT_PYTHON_API = """# nsight-python API Reference

`pip install nsight-python` (Apache-2.0, Beta)
Requires: Python >= 3.10, NCU CLI installed on the system.

## @nsight.analyze.kernel — Core Profiling Decorator

Profiles GPU kernels via Nsight Compute, collects metrics, handles thermal management.

```python
from nsight import analyze

@analyze.kernel(
    configs=[(2048,), (4096,), (8192,)],  # parameter sweep
    runs=10,                               # executions per config
    metrics=["gpu__time_duration.sum"],     # NCU metrics to collect
    derive_metric=None,                    # custom metric function
    normalize_against=None,                # baseline annotation name for speedup
    combine_kernel_metrics=None,           # combine multiple kernels: lambda x,y: x+y
    ignore_kernel_list=None,               # kernel names to skip
    clock_control="none",                  # "base" locks clocks, "none" leaves them
    cache_control="all",                   # "all" flushes caches before each pass
    replay_mode="kernel",                  # "kernel" or "range"
    thermal_mode="auto",                   # "auto", "manual", "off"
    output="progress",                     # "quiet", "progress", "verbose"
    output_csv=False,                      # generate CSV files
)
def my_benchmark(size):
    with nsight.annotate("my_kernel"):
        launch_kernel(size)

result = my_benchmark()  # returns ProfileResults
df = result.to_dataframe()
```

## ProfileResults.to_dataframe() columns

| Column | Description |
|---|---|
| Annotation | Name of the annotated region |
| AvgValue | Average metric across runs |
| StdDev | Standard deviation |
| MinValue, MaxValue | Range |
| NumRuns | Run count |
| CI95_Lower, CI95_Upper | 95% confidence interval |
| RelativeStdDevPct | Std dev as % of mean |
| StableMeasurement | True if RelativeStdDevPct < 2% |
| Metric | Metric name |
| Kernel | GPU kernel name(s) |
| GPU | Device name |
| ComputeClock, MemoryClock | Clock frequencies |

## nsight.annotate — Marking Profiling Regions

```python
import nsight

# As context manager:
with nsight.annotate("matmul"):
    C = A @ B

# As decorator:
@nsight.annotate("my_op")
def my_operation():
    ...
```

Constraints: no nesting, no duplicate names within a run.

## @nsight.analyze.plot — Visualization

```python
@analyze.plot(
    filename="perf.png",
    metric="gpu__time_duration.sum",
    title="Kernel Performance",
    plot_type="bar",              # "line" or "bar"
    annotate_points=True,         # show values on points
    show_aggregate="avg",         # "avg" or "geomean"
    row_panels=["dtype"],         # facet by parameter
    col_panels=["transpose"],
    variant_fields=["block_size"],
    variant_annotations=["triton"],
)
@analyze.kernel(configs=configs, runs=10)
def benchmark(size, dtype):
    ...
```

## Common Patterns

### Compare two implementations
```python
@analyze.kernel(runs=10)
def compare(size):
    with nsight.annotate("torch"):
        torch.mm(A, B)
    with nsight.annotate("cutedsl"):
        my_cutedsl_kernel(A, B, C)
```

### Custom TFLOPs metric
```python
def tflops(gpu_time_ns, M, N, K):
    flops = 2 * M * N * K
    return flops / gpu_time_ns  # GFLOPS (ns denominator gives GFLOPS)

@analyze.kernel(
    configs=[(1024, 1024, 1024), (2048, 2048, 2048)],
    metrics=["gpu__time_duration.sum"],
    derive_metric=tflops,
    runs=10,
)
def gemm_benchmark(M, N, K):
    with nsight.annotate("gemm"):
        launch_gemm(M, N, K)
```

### Speedup against baseline
```python
@analyze.plot(metric="speedup", filename="speedup.png")
@analyze.kernel(
    configs=configs,
    normalize_against="torch",  # baseline annotation
    derive_metric=lambda x: x,  # identity (normalized values ARE the speedup)
    runs=10,
)
def speedup_test(size):
    with nsight.annotate("torch"):
        torch.mm(A, B)
    with nsight.annotate("triton"):
        triton_mm(A, B)
```

### Multiple raw metrics
```python
@analyze.kernel(
    metrics=[
        "gpu__time_duration.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ],
    runs=5,
)
def multi_metric(size):
    with nsight.annotate("kernel"):
        launch(size)

result = multi_metric()
df = result.to_dataframe()  # has rows for each metric
```

## Pipeline internals

1. Collection: spawns `ncu` subprocess, re-executes function under profiling
2. Extraction: reads `.ncu-rep` via `ncu_report`, filters by NVTX domain
3. Transformation: aggregates stats (mean, std, CI95, stability check)
4. Visualization: matplotlib plots with NVIDIA color palette
"""

INGERO_REFERENCE = """# ingero — eBPF-based CUDA Causal Tracing

ingero (https://github.com/ingero-io/ingero) is an eBPF-based GPU observability
agent that traces CUDA Runtime and Driver APIs to build causal chains explaining
GPU latency. It has MCP server support.

## What it does

- Scans /proc for processes linked to libcudart.so
- Loads eBPF uprobes onto CUDA runtime/driver API entry/exit points
- Records PID/TID/timestamps into per-layer ring buffers
- Builds causal chains: Python call → aten op → CUDA runtime → driver → GPU kernel
- Explains WHY a kernel was launched and what's blocking it

## When to use ingero vs other tools

| Tool | Use case |
|---|---|
| torch.profiler (Kineto) | Offline trace of a training step — export JSON, analyze later |
| nsight-python (NCU) | Per-kernel metrics — occupancy, memory throughput, warp stalls |
| nsight-systems (nsys) | System-wide timeline — CPU/GPU overlap, multi-stream, NVTX |
| ingero (eBPF) | Live causal tracing — "why is this kernel waiting?" with full stack |
| Intra-kernel (%clock64) | Inside one kernel — which pipeline stage (MMA/copy/sync) is slow |

## Setup

```bash
# Requires Linux, root/CAP_SYS_ADMIN for eBPF
git clone https://github.com/ingero-io/ingero
cd ingero
go build -o ingero ./cmd/ingero
sudo ./ingero  # starts eBPF probes + MCP server
```

## MCP integration

ingero runs as a standalone MCP server. You can add it alongside kernel-mcp
in your Claude Code settings to have both documentation+profiling and live tracing:

```json
{
  "mcpServers": {
    "nvidia-docs": { "command": "uv", "args": ["--directory", "...", "run", "nvidia-docs-mcp"] },
    "ingero": { "command": "sudo", "args": ["./ingero", "--mcp"] }
  }
}
```
"""

ALL_TEMPLATES = {
    "overview": INTRA_KERNEL_OVERVIEW,
    "clock_read": CLOCK_READ_TEMPLATE,
    "gemm_profiling": GEMM_PROFILING_TEMPLATE,
    "pipeline_profiling": PIPELINE_PROFILING_TEMPLATE,
    "mbarrier_profiling": MBARRIER_PROFILING_TEMPLATE,
    "ptx_timing_registers": PTX_TIMING_REGISTERS,
    "nsight_compute_metrics": NSIGHT_COMPUTE_METRICS,
    "nsight_python_api": NSIGHT_PYTHON_API,
    "ingero": INGERO_REFERENCE,
}
