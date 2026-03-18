"""MCP server for NVIDIA PTX ISA and CuTe DSL documentation."""

import sys

from mcp.server.fastmcp import FastMCP

from nvidia_docs_mcp.search import ptx_index, cutedsl_index, cutedsl_source_index

MAX_RESPONSE_CHARS = 50000


def _truncate(text: str, limit: int = MAX_RESPONSE_CHARS) -> str:
    """If text exceeds limit, return a TOC of all sections + the first section's content."""
    if len(text) <= limit:
        return text

    # Split into sections and build a table of contents
    import re
    sections = re.split(r'\n---\n', text)
    headings = []
    for s in sections:
        match = re.search(r'^##?\s+(.+)', s.strip())
        if match:
            headings.append(match.group(1).strip()[:120])
        else:
            headings.append(s.strip()[:80])

    toc = "**Response too large — showing table of contents. Use the `symbol` parameter or a more specific query to get details.**\n\n"
    toc += f"Found {len(sections)} sections:\n\n"
    for i, h in enumerate(headings, 1):
        toc += f"{i}. {h}\n"

    # Include as many full sections as we can fit
    toc += "\n---\n\n"
    remaining = limit - len(toc) - 200
    included = []
    for s in sections:
        if remaining <= 0:
            break
        if len(s) <= remaining:
            included.append(s)
            remaining -= len(s) + 5  # account for separator
        else:
            included.append(s[:remaining] + "\n\n... (section truncated)")
            break

    return toc + "\n\n---\n\n".join(included)

mcp = FastMCP(
    "nvidia-docs",
    instructions=(
        "NVIDIA GPU programming documentation server. "
        "Provides access to PTX ISA instructions, CuTe DSL (Python DSL for CUDA kernels) documentation, "
        "and CuTe DSL source code with examples. "
        "Use these tools to understand PTX instructions, CuTe DSL APIs, read source implementations, "
        "and learn from real-world kernel examples (GEMM, FMHA, MLA, Mamba2, distributed ops)."
    ),
)


@mcp.tool()
def search_ptx(query: str, architecture: str | None = None, top_k: int = 5) -> str:
    """Search the PTX ISA documentation.

    Use this to find PTX instructions, understand memory models, synchronization
    primitives, matrix operations, and more.

    Args:
        query: Search query (e.g. "warpgroup matrix multiply", "async copy shared memory",
               "barrier synchronization hopper")
        architecture: Optional SM architecture filter (e.g. "sm_90", "sm_100")
        top_k: Number of results to return (default 5)
    """
    results = ptx_index.search(query, top_k=top_k, arch_filter=architecture)
    if not results:
        return "No results found. Try broadening your query or removing the architecture filter."

    parts = []
    for r in results:
        header = f"## {r['heading']}"
        if r.get("architectures"):
            header += f" [{', '.join(r['architectures'])}]"
        parts.append(f"{header}\n\n{r['content']}")

    return _truncate("\n\n---\n\n".join(parts))


@mcp.tool()
def get_ptx_instruction(instruction: str, architecture: str | None = None) -> str:
    """Get detailed documentation for a specific PTX instruction.

    Args:
        instruction: PTX instruction name (e.g. "mma", "wgmma.mma_async", "cp.async.bulk",
                     "tcgen05.mma", "ldmatrix", "mbarrier.arrive")
        architecture: Optional SM architecture filter (e.g. "sm_90", "sm_100")
    """
    results = ptx_index.get_instruction(instruction, arch=architecture)
    if not results:
        # Fall back to search
        search_results = ptx_index.search(instruction, top_k=3, arch_filter=architecture)
        if not search_results:
            return f"No documentation found for instruction '{instruction}'. Check the instruction name."
        results = search_results

    parts = []
    for r in results:
        header = f"## {r['heading']}"
        if r.get("architectures"):
            header += f" [{', '.join(r['architectures'])}]"
        parts.append(f"{header}\n\n{r['content']}")

    return _truncate("\n\n---\n\n".join(parts))


@mcp.tool()
def search_cutedsl(query: str, top_k: int = 5) -> str:
    """Search the CuTe DSL documentation.

    CuTe DSL is NVIDIA's Python DSL for writing CUDA kernels. Use this to find
    information about kernel writing patterns, API usage, layouts, tensor operations,
    JIT compilation, framework integration, and more.

    Args:
        query: Search query (e.g. "warpgroup mma", "layout tensor", "copy async",
               "pipeline", "autotuning", "pytorch integration")
        top_k: Number of results to return (default 5)
    """
    results = cutedsl_index.search(query, top_k=top_k)
    if not results:
        return "No results found. Try broadening your query."

    parts = []
    for r in results:
        header = f"## {r['heading']}"
        if r.get("source_page"):
            header += f" (from {r['source_page']})"
        parts.append(f"{header}\n\n{r['content']}")

    return _truncate("\n\n---\n\n".join(parts))


@mcp.tool()
def get_cutedsl_api(module: str, symbol: str | None = None) -> str:
    """Get CuTe DSL API reference for a specific module or symbol.

    Args:
        module: Module name (e.g. "cute", "cute.arch", "cute_nvgpu", "cute_nvgpu.warp",
                "cute_nvgpu.warpgroup", "cute_nvgpu.tcgen05", "cute_nvgpu.cpasync",
                "pipeline", "utils")
        symbol: Optional specific function/class name to look up within the module
    """
    # Map module names to source page patterns
    module_map = {
        "cute": "cute_dsl_api/cute.html",
        "cute.arch": "cute_dsl_api/cute_arch.html",
        "cute.runtime": "cute_dsl_api/cute_runtime.html",
        "cute_nvgpu": "cute_dsl_api/cute_nvgpu.html",
        "cute_nvgpu.common": "cute_dsl_api/cute_nvgpu_common.html",
        "cute_nvgpu.warp": "cute_dsl_api/cute_nvgpu_warp.html",
        "cute_nvgpu.warpgroup": "cute_dsl_api/cute_nvgpu_warpgroup.html",
        "cute_nvgpu.cpasync": "cute_dsl_api/cute_nvgpu_cpasync.html",
        "cute_nvgpu.tcgen05": "cute_dsl_api/cute_nvgpu_tcgen05.html",
        "pipeline": "cute_dsl_api/pipeline.html",
        "utils": "cute_dsl_api/utils.html",
        "utils.sm90": "cute_dsl_api/utils_sm90.html",
        "utils.sm100": "cute_dsl_api/utils_sm100.html",
    }

    target_page = module_map.get(module.lower())
    if not target_page:
        return (
            f"Unknown module '{module}'. Available modules: {', '.join(sorted(module_map.keys()))}"
        )

    cutedsl_index.load()
    results = []
    for entry in cutedsl_index.index:
        if entry.get("source_page") == target_page:
            doc = cutedsl_index.documents.get(entry["file"])
            if doc:
                if symbol:
                    if symbol.lower() in doc.get("content", "").lower() or \
                       symbol.lower() in doc.get("heading", "").lower():
                        results.append(doc)
                else:
                    results.append(doc)

    if not results:
        if symbol:
            return f"No documentation found for '{symbol}' in module '{module}'."
        return f"No documentation found for module '{module}'."

    parts = []
    for doc in results:
        parts.append(f"## {doc['heading']}\n\n{doc['content']}")

    return _truncate("\n\n---\n\n".join(parts))


@mcp.tool()
def list_ptx_instructions(category: str | None = None, architecture: str | None = None) -> str:
    """List available PTX instructions, optionally filtered by category or architecture.

    Args:
        category: Optional category filter (e.g. "matrix", "memory", "sync", "arithmetic",
                  "wgmma", "tcgen05", "mbarrier", "atomic", "float", "control")
        architecture: Optional SM architecture filter (e.g. "sm_90", "sm_100")
    """
    ptx_index.load()

    category_keywords = {
        "matrix": ["wmma", "mma", "ldmatrix", "stmatrix", "movmatrix", "wgmma", "tcgen05.mma"],
        "memory": ["ld", "st", "cp.async", "prefetch", "mov", "shfl", "prmt"],
        "sync": ["bar", "barrier", "fence", "membar", "mbarrier", "atom", "red", "vote", "elect"],
        "arithmetic": ["add", "sub", "mul", "mad", "div", "rem", "abs", "neg", "min", "max"],
        "float": ["fma", "rcp", "sqrt", "sin", "cos", "lg2", "ex2", "tanh", "testp", "copysign"],
        "wgmma": ["wgmma"],
        "tcgen05": ["tcgen05"],
        "mbarrier": ["mbarrier"],
        "atomic": ["atom", "red"],
        "control": ["bra", "brx", "call", "ret", "exit"],
    }

    instructions = set()
    for entry in ptx_index.index:
        doc = ptx_index.documents.get(entry["file"])
        if not doc:
            continue
        instr = doc.get("instruction", "")
        if not instr:
            continue

        if architecture:
            archs = doc.get("architectures", [])
            if archs and architecture.lower() not in [a.lower() for a in archs]:
                continue

        if category:
            keywords = category_keywords.get(category.lower(), [category.lower()])
            if not any(kw in instr.lower() for kw in keywords):
                continue

        instructions.add(instr)

    if not instructions:
        return "No instructions found matching the filter."

    sorted_instrs = sorted(instructions)
    lines = [f"Found {len(sorted_instrs)} instructions:"]
    for instr in sorted_instrs:
        lines.append(f"- {instr}")

    return "\n".join(lines)


@mcp.tool()
def search_cutedsl_source(query: str, top_k: int = 5) -> str:
    """Search the CuTe DSL Python source code and examples.

    This searches the actual implementation code from the CUTLASS repository,
    including the CuTe DSL library source and example kernels (GEMM, FMHA, MLA,
    Mamba2, distributed ops, etc.).

    Args:
        query: Search query (e.g. "warpgroup mma", "flash attention", "pipeline sm90",
               "tcgen05 copy", "persistent gemm", "MLA decode", "blockscaled")
        top_k: Number of results to return (default 5)
    """
    results = cutedsl_source_index.search(query, top_k=top_k)
    if not results:
        return "No results found. Try broadening your query."

    parts = []
    for r in results:
        header = f"## {r['heading']}"
        content = r.get("content", "")
        # Truncate very long source code results
        if len(content) > 4000:
            content = content[:4000] + "\n\n... (truncated, use read_cutedsl_source for full content)"
        parts.append(f"{header}\n\n{content}")

    return _truncate("\n\n---\n\n".join(parts))


@mcp.tool()
def read_cutedsl_source(module_or_file: str) -> str:
    """Read the full source code of a CuTe DSL module or example file.

    Args:
        module_or_file: Module path (e.g. "cutlass.cute.nvgpu.warpgroup.mma",
                        "cutlass.cute.core", "cutlass.pipeline.sm90") or
                        example file path (e.g. "blackwell/dense_gemm.py",
                        "hopper/fmha.py", "blackwell/fmha.py")
    """
    doc = cutedsl_source_index.get_source(module_or_file)
    if not doc:
        return (
            f"Module or file '{module_or_file}' not found. "
            "Use list_cutedsl_modules() or search_cutedsl_source() to find available files."
        )

    source = doc.get("source_code", "")
    if not source:
        # Try to load from raw source file
        raw_file = doc.get("raw_source_file")
        if raw_file:
            from pathlib import Path
            raw_path = Path(__file__).parent.parent.parent / "data" / "cutedsl_source" / raw_file
            if raw_path.exists():
                source = raw_path.read_text()

    if not source:
        return doc.get("content", "No source available.")

    heading = doc.get("heading", module_or_file)
    module = doc.get("module", "")
    file_path = doc.get("file_path", "")

    header = f"# {heading}"
    if module:
        header += f"\n\nModule: `{module}`"
    if file_path:
        header += f"\nFile: `{file_path}`"

    return _truncate(f"{header}\n\n```python\n{source}\n```")


@mcp.tool()
def list_cutedsl_modules(filter: str | None = None) -> str:
    """List available CuTe DSL source modules and example files.

    Args:
        filter: Optional filter string (e.g. "warpgroup", "tcgen05", "blackwell",
                "hopper", "pipeline", "gemm", "fmha", "example")
    """
    results = cutedsl_source_index.list_modules(filter)
    if not results:
        return "No modules found matching the filter."

    sources = [r for r in results if r.get("type") == "source"]
    examples = [r for r in results if r.get("type") == "example"]

    lines = []
    if sources:
        lines.append(f"## Source Modules ({len(sources)})\n")
        for r in sources:
            symbols_str = ""
            if r.get("symbols"):
                top_symbols = r["symbols"][:8]
                if len(r["symbols"]) > 8:
                    top_symbols.append(f"... +{len(r['symbols']) - 8} more")
                symbols_str = f" — {', '.join(top_symbols)}"
            lines.append(f"- `{r['module']}`{symbols_str}")

    if examples:
        lines.append(f"\n## Examples ({len(examples)})\n")
        by_category: dict[str, list] = {}
        for r in examples:
            cat = r.get("category", "misc")
            by_category.setdefault(cat, []).append(r)
        for cat in sorted(by_category):
            lines.append(f"\n### {cat}")
            for r in by_category[cat]:
                lines.append(f"- `{r['file_path']}`")

    return "\n".join(lines)


def main():
    if "--scrape" in sys.argv:
        import asyncio
        from nvidia_docs_mcp.scraper import scrape_all
        asyncio.run(scrape_all())
    elif "--describe-figures" in sys.argv:
        import asyncio
        from nvidia_docs_mcp.describe_figures import describe_all_figures, inject_descriptions_into_index
        model = "anthropic/claude-sonnet-4.6"
        for arg in sys.argv:
            if arg.startswith("--model="):
                model = arg.split("=", 1)[1]
        asyncio.run(describe_all_figures(model=model))
        inject_descriptions_into_index()
    else:
        mcp.run()


if __name__ == "__main__":
    main()
