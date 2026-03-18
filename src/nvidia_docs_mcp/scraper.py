"""Scrapes NVIDIA PTX ISA and CuTe DSL documentation into structured local files."""

import json
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

DATA_DIR = Path(__file__).parent.parent.parent / "data"

PTX_BASE_URL = "https://docs.nvidia.com/cuda/parallel-thread-execution/"
CUTEDSL_BASE_URL = "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/"

CUTEDSL_PAGES = [
    "cute_dsl_general/dsl_introduction.html",
    "cute_dsl_general/dsl_code_generation.html",
    "cute_dsl_general/dsl_control_flow.html",
    "cute_dsl_general/dsl_jit_arg_generation.html",
    "cute_dsl_general/dsl_dynamic_layout.html",
    "cute_dsl_general/dsl_jit_caching.html",
    "cute_dsl_general/dsl_jit_compilation_options.html",
    "cute_dsl_general/framework_integration.html",
    "cute_dsl_general/debugging.html",
    "cute_dsl_general/autotuning_gemm.html",
    "cute_dsl_general/dsl_ahead_of_time_compilation.html",
    "cute_dsl_api/cute.html",
    "cute_dsl_api/cute_arch.html",
    "cute_dsl_api/cute_runtime.html",
    "cute_dsl_api/cute_nvgpu.html",
    "cute_dsl_api/cute_nvgpu_common.html",
    "cute_dsl_api/cute_nvgpu_warp.html",
    "cute_dsl_api/cute_nvgpu_warpgroup.html",
    "cute_dsl_api/cute_nvgpu_cpasync.html",
    "cute_dsl_api/cute_nvgpu_tcgen05.html",
    "cute_dsl_api/pipeline.html",
    "cute_dsl_api/utils.html",
    "cute_dsl_api/utils_sm90.html",
    "cute_dsl_api/utils_sm100.html",
    "limitations.html",
    "faqs.html",
]


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove Sphinx link anchors like [#](... "Link to this definition")
    text = re.sub(r'\[#\]\([^)]*"Link to this (?:definition|heading)"\)', '', text)
    # Remove pilcrow/link characters at end of headings
    text = re.sub(r'\s*[¶\uf0c1]+\s*', ' ', text)
    # Clean up escaped underscores from Sphinx
    text = re.sub(r'\\_', '_', text)
    # Remove "Bases: X" noise from class signatures in headings
    text = re.sub(r':   Bases: `\w+`\n\n', '', text)
    # Clean "class" prefix jammed against module path
    text = re.sub(r'### `class([a-z])', r'### `class \1', text)
    # Replace raw image markdown with descriptive text using figure name
    # e.g. ![_images/wgmma-64N16-A.png](...) -> [Figure: wgmma-64N16-A]
    text = re.sub(
        r'!\[_images/([\w.-]+?)(?:\.png|\.svg|\.jpg|\.gif)\]\([^)]+\)',
        r'[Figure: \1]',
        text
    )
    # Clean up permalink remnants
    text = re.sub(r'\[\s*\]\([^)]*"Permalink to this [^"]*"\)', '', text)
    return text.strip()


def _extract_main_content(soup: BeautifulSoup) -> Tag:
    """Extract the main content area, skipping nav/sidebar in Sphinx pages."""
    # Sphinx/furo theme uses <article> or <main> or div.body
    for selector in ["article", "main", "div.body", "div.document", "div[role='main']"]:
        el = soup.select_one(selector)
        if el:
            return el
    return soup.find("body") or soup


def _extract_sphinx_api_entries(content_root: Tag) -> list[dict]:
    """Extract API entries from Sphinx autodoc output (dl/dt/dd structure)."""
    entries = []
    for dl in content_root.find_all("dl", class_=re.compile(r"py|class|function|method|attribute")):
        for dt in dl.find_all("dt", recursive=False):
            sig = dt.get_text(strip=True)
            # Get the dd (description) that follows
            dd = dt.find_next_sibling("dd")
            desc = ""
            if dd:
                desc = markdownify(str(dd))
            # Extract the symbol name from the id attribute
            sig_id = dt.get("id", "")
            entries.append({
                "signature": sig,
                "id": sig_id,
                "description": _clean_text(desc),
            })
    return entries


def _extract_sections_from_html(soup: BeautifulSoup, use_sphinx: bool = False) -> list[dict]:
    """Split an HTML document into sections based on headings."""
    content_root = _extract_main_content(soup) if use_sphinx else (soup.find("body") or soup)

    sections = []
    # Find all headings within the content area
    all_headings = content_root.find_all(re.compile(r"^h[1-6]$"))

    if not all_headings:
        # No headings found - treat entire content as one section
        md = markdownify(str(content_root))
        title = soup.title.get_text(strip=True) if soup.title else "Documentation"
        return [{"heading": title, "level": 1, "content": _clean_text(md)}]

    for i, heading in enumerate(all_headings):
        level = int(heading.name[1])
        heading_text = heading.get_text(strip=True)
        # Remove trailing "#" link anchors
        heading_text = re.sub(r"\s*[#¶]+\s*$", "", heading_text)

        # Gather content until next heading at same or higher level
        content_parts = []
        sibling = heading.find_next_sibling()
        while sibling:
            if isinstance(sibling, Tag) and sibling.name and re.match(r"^h[1-6]$", sibling.name):
                next_level = int(sibling.name[1])
                if next_level <= level:
                    break
            content_parts.append(str(sibling))
            sibling = sibling.find_next_sibling()

        md_content = markdownify("".join(content_parts))
        sections.append({
            "heading": heading_text,
            "level": level,
            "content": _clean_text(md_content),
        })

    return sections


def _split_ptx_by_instruction(soup: BeautifulSoup) -> list[dict]:
    """Extract individual PTX instruction entries from the large document.

    Looks for heading patterns like "9.7.x.y InstructionName" and captures
    everything until the next heading of equal or higher level.
    """
    sections = []
    all_headings = soup.find_all(re.compile(r"^h[1-6]$"))

    instruction_section_started = False
    for i, heading in enumerate(all_headings):
        text = heading.get_text(strip=True)

        # Detect section 9 (Instruction Set) and subsections
        if re.match(r"^9[\.\s]", text):
            instruction_section_started = True

        if not instruction_section_started:
            continue

        level = int(heading.name[1])

        # Gather content until next heading of same or higher level
        content_parts = []
        sibling = heading.find_next_sibling()
        while sibling:
            if isinstance(sibling, Tag) and sibling.name and re.match(r"^h[1-6]$", sibling.name):
                sibling_level = int(sibling.name[1])
                if sibling_level <= level:
                    break
                # Include sub-headings content inline
            content_parts.append(str(sibling))
            sibling = sibling.find_next_sibling()

        md_content = markdownify("".join(content_parts))

        # Try to extract instruction name from heading
        clean_heading = re.sub(r'[\uf0c1¶]', '', text).strip()
        instr_match = re.search(r"[\d.]+\s*(.+)", clean_heading)
        instr_name = instr_match.group(1).strip() if instr_match else clean_heading

        # Detect architecture targets from content
        archs = set()
        full_text = text + " " + md_content
        for arch in ["sm_50", "sm_52", "sm_53", "sm_60", "sm_61", "sm_62",
                      "sm_70", "sm_72", "sm_75", "sm_80", "sm_86", "sm_89",
                      "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_101a"]:
            if arch in full_text:
                archs.add(arch)

        sections.append({
            "heading": text,
            "instruction": instr_name,
            "level": level,
            "architectures": sorted(archs),
            "content": _clean_text(md_content),
        })

    return sections


def _chunk_large_sections(sections: list[dict], max_chars: int = 8000) -> list[dict]:
    """Split sections that are too large into smaller chunks."""
    result = []
    for section in sections:
        content = section["content"]
        if len(content) <= max_chars:
            result.append(section)
            continue

        # Split by double newlines (paragraph boundaries)
        paragraphs = content.split("\n\n")
        chunk_parts = []
        chunk_len = 0
        chunk_idx = 0

        for para in paragraphs:
            if chunk_len + len(para) > max_chars and chunk_parts:
                result.append({
                    **section,
                    "heading": f"{section['heading']} (part {chunk_idx + 1})",
                    "content": "\n\n".join(chunk_parts),
                })
                chunk_parts = []
                chunk_len = 0
                chunk_idx += 1
            chunk_parts.append(para)
            chunk_len += len(para)

        if chunk_parts:
            suffix = f" (part {chunk_idx + 1})" if chunk_idx > 0 else ""
            result.append({
                **section,
                "heading": f"{section['heading']}{suffix}",
                "content": "\n\n".join(chunk_parts),
            })
    return result


async def scrape_ptx_docs() -> Path:
    """Fetch and parse PTX ISA documentation."""
    out_dir = DATA_DIR / "ptx"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching PTX ISA documentation (this is a large page)...")
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        resp = await client.get(PTX_BASE_URL)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    print("Parsing PTX instructions...")
    sections = _split_ptx_by_instruction(soup)
    sections = _chunk_large_sections(sections)

    # Also extract non-instruction sections (1-8, 10+)
    general_sections = _extract_sections_from_html(soup)

    # Save instruction sections
    index = []
    for i, section in enumerate(sections):
        filename = f"instr_{i:04d}.json"
        (out_dir / filename).write_text(json.dumps(section, indent=2))
        index.append({
            "file": filename,
            "heading": section["heading"],
            "instruction": section.get("instruction", ""),
            "architectures": section.get("architectures", []),
        })

    # Save general sections
    for i, section in enumerate(general_sections):
        filename = f"general_{i:04d}.json"
        (out_dir / filename).write_text(json.dumps(section, indent=2))
        index.append({
            "file": filename,
            "heading": section["heading"],
        })

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))
    print(f"Saved {len(sections)} instruction sections + {len(general_sections)} general sections")
    return out_dir


async def scrape_cutedsl_docs() -> Path:
    """Fetch and parse CuTe DSL documentation pages."""
    out_dir = DATA_DIR / "cutedsl"
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        for page_path in CUTEDSL_PAGES:
            url = CUTEDSL_BASE_URL + page_path
            slug = page_path.replace("/", "_").replace(".html", "")
            print(f"Fetching {page_path}...")

            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                print(f"  Failed: {e}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Derive page title from the slug for context
            page_title = page_path.split("/")[-1].replace(".html", "").replace("_", " ").title()
            if soup.title:
                page_title = re.sub(r"\s*[-—|].*$", "", soup.title.get_text(strip=True))

            sections = _extract_sections_from_html(soup, use_sphinx=True)

            # Also extract API entries from autodoc
            content_root = _extract_main_content(soup)
            api_entries = _extract_sphinx_api_entries(content_root)
            if api_entries:
                # Group API entries into chunks
                api_chunk: list[str] = []
                api_chunk_len = 0
                api_chunk_idx = 0
                for entry in api_entries:
                    text = f"### `{entry['signature']}`\n\n{entry['description']}"
                    if api_chunk_len + len(text) > 6000 and api_chunk:
                        sections.append({
                            "heading": f"{page_title} - API Reference (part {api_chunk_idx + 1})",
                            "level": 2,
                            "content": "\n\n---\n\n".join(api_chunk),
                        })
                        api_chunk = []
                        api_chunk_len = 0
                        api_chunk_idx += 1
                    api_chunk.append(text)
                    api_chunk_len += len(text)
                if api_chunk:
                    suffix = f" (part {api_chunk_idx + 1})" if api_chunk_idx > 0 else ""
                    sections.append({
                        "heading": f"{page_title} - API Reference{suffix}",
                        "level": 2,
                        "content": "\n\n---\n\n".join(api_chunk),
                    })

            sections = _chunk_large_sections(sections)

            for i, section in enumerate(sections):
                filename = f"{slug}_{i:04d}.json"
                section["source_page"] = page_path
                section["source_url"] = url
                section["page_title"] = page_title
                (out_dir / filename).write_text(json.dumps(section, indent=2))
                index.append({
                    "file": filename,
                    "heading": section["heading"],
                    "source_page": page_path,
                    "page_title": page_title,
                })

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))
    print(f"Saved {len(index)} CuTe DSL documentation sections")
    return out_dir


CUTLASS_REPO_URL = "https://github.com/NVIDIA/cutlass.git"
CUTLASS_CLONE_DIR = Path("/tmp/cutlass-source")


def _extract_python_symbols(source: str) -> list[dict]:
    """Extract classes, functions, and their docstrings from Python source."""
    import ast

    symbols = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return symbols

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            docstring = ast.get_docstring(node) or ""

            # Extract signature for functions
            sig = ""
            if kind == "function":
                args = []
                for arg in node.args.args:
                    ann = ""
                    if arg.annotation:
                        ann = f": {ast.unparse(arg.annotation)}"
                    args.append(f"{arg.arg}{ann}")
                ret = ""
                if node.returns:
                    ret = f" -> {ast.unparse(node.returns)}"
                sig = f"({', '.join(args)}){ret}"

            # Extract decorators
            decorators = []
            for dec in node.decorator_list:
                try:
                    decorators.append(f"@{ast.unparse(dec)}")
                except Exception:
                    pass

            # Get source lines for context
            start = node.lineno - 1
            end = node.end_lineno or start + 1
            source_lines = source.split("\n")[start:end]

            symbols.append({
                "name": node.name,
                "kind": kind,
                "signature": sig,
                "decorators": decorators,
                "docstring": docstring,
                "line": node.lineno,
                "source": "\n".join(source_lines),
            })
    return symbols


def index_cutedsl_source() -> Path:
    """Index the CuTe DSL Python source code from the CUTLASS repo."""
    out_dir = DATA_DIR / "cutedsl_source"
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = CUTLASS_CLONE_DIR
    if not repo_dir.exists():
        import subprocess
        print("Cloning CUTLASS repository (shallow)...")
        subprocess.run(
            ["git", "clone", "--depth", "1", CUTLASS_REPO_URL, str(repo_dir)],
            check=True, capture_output=True,
        )

    dsl_root = repo_dir / "python" / "CuTeDSL" / "cutlass"
    examples_root = repo_dir / "examples" / "python" / "CuTeDSL"

    index = []

    # Index source files
    print("Indexing CuTe DSL source code...")
    for py_file in sorted(dsl_root.rglob("*.py")):
        rel_path = str(py_file.relative_to(dsl_root))
        module_path = "cutlass." + rel_path.replace("/", ".").replace(".py", "").replace(".__init__", "")
        source = py_file.read_text(errors="replace")

        if not source.strip():
            continue

        symbols = _extract_python_symbols(source)

        # Create a section for the module overview
        slug = rel_path.replace("/", "_").replace(".py", "")
        filename = f"src_{slug}.json"

        # Build module content with symbols
        parts = [f"# Module: `{module_path}`\n\nFile: `{rel_path}`\n"]
        for sym in symbols:
            dec_str = "\n".join(sym["decorators"]) + "\n" if sym["decorators"] else ""
            if sym["kind"] == "class":
                parts.append(f"## class `{sym['name']}`\n\n{dec_str}{sym['docstring']}")
            else:
                parts.append(f"## `{sym['name']}{sym['signature']}`\n\n{dec_str}{sym['docstring']}")

        # Also include the raw source for full-text search
        content = "\n\n---\n\n".join(parts)

        section = {
            "heading": f"Source: {module_path}",
            "module": module_path,
            "file_path": rel_path,
            "content": content,
            "symbols": [s["name"] for s in symbols],
            "source_code": source,
            "type": "source",
        }

        # Chunk if too large - store source separately
        if len(content) > 8000:
            # Store summary (without raw source)
            section["source_code"] = ""  # Don't include in the main index search
            (out_dir / filename).write_text(json.dumps(section, indent=2))

            # Store raw source as separate file for retrieval
            src_filename = f"raw_{slug}.py"
            (out_dir / src_filename).write_text(source)
            section["raw_source_file"] = src_filename
        else:
            (out_dir / filename).write_text(json.dumps(section, indent=2))

        index.append({
            "file": filename,
            "heading": f"Source: {module_path}",
            "module": module_path,
            "symbols": [s["name"] for s in symbols],
            "type": "source",
        })

    # Index examples
    print("Indexing CuTe DSL examples...")
    for py_file in sorted(examples_root.rglob("*.py")):
        rel_path = str(py_file.relative_to(examples_root))
        source = py_file.read_text(errors="replace")

        if not source.strip():
            continue

        slug = rel_path.replace("/", "_").replace(".py", "")
        filename = f"example_{slug}.json"

        # Extract the module-level docstring for description
        import ast
        try:
            tree = ast.parse(source)
            docstring = ast.get_docstring(tree) or ""
        except SyntaxError:
            docstring = ""

        # Determine category from path
        parts = rel_path.split("/")
        category = parts[0] if parts else "misc"

        section = {
            "heading": f"Example: {rel_path}",
            "file_path": rel_path,
            "category": category,
            "docstring": docstring,
            "content": f"# Example: `{rel_path}`\n\nCategory: {category}\n\n{docstring}\n\n```python\n{source}\n```",
            "source_code": source,
            "type": "example",
        }

        (out_dir / filename).write_text(json.dumps(section, indent=2))
        index.append({
            "file": filename,
            "heading": f"Example: {rel_path}",
            "category": category,
            "type": "example",
        })

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))
    print(f"Saved {len([e for e in index if e['type'] == 'source'])} source modules + "
          f"{len([e for e in index if e['type'] == 'example'])} examples")
    return out_dir


async def scrape_all():
    """Scrape both documentation sources and index source code."""
    await scrape_ptx_docs()
    print()
    await scrape_cutedsl_docs()
    print()
    index_cutedsl_source()
    print("\nDone! All documentation and source code indexed to data/")


if __name__ == "__main__":
    import asyncio
    asyncio.run(scrape_all())
