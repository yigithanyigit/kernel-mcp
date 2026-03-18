"""Use a VLM to generate detailed text descriptions of PTX ISA figures.

Downloads all figure images from the PTX documentation and sends them
through Claude Sonnet via fal.ai's OpenRouter gateway with category-specific
prompts to extract the detailed layout data trapped in the images.
"""

import asyncio
import json
import os
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from openai import AsyncOpenAI

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PTX_BASE_URL = "https://docs.nvidia.com/cuda/parallel-thread-execution/"

SYSTEM_PROMPT = (
    "You are a GPU architecture expert analyzing NVIDIA PTX ISA documentation figures. "
    "Extract every piece of data visible in the figure with extreme precision. "
    "Do not use markdown formatting. Describe layouts as structured text tables where applicable. "
    "This description will be the ONLY way programmers can understand this figure — be thorough."
)

# Category-specific prompts
PROMPTS = {
    "register_fragment_layout": (
        "This is a PTX ISA register fragment layout figure. It shows how matrix elements "
        "are distributed across GPU threads in registers for a matrix instruction.\n\n"
        "Extract ALL of the following in detail:\n"
        "1. The matrix dimensions (M x N x K) and which matrix this is (A, B, C, or D/accumulator)\n"
        "2. The grid structure: rows = threads (T0-T31), columns = register elements\n"
        "3. For EACH thread (or groups of threads), list exactly which matrix elements (e.g., a0, a1, ...) "
        "are in which registers\n"
        "4. The data type (.f16, .bf16, .tf32, .f32, etc.)\n"
        "5. Any grouping patterns (e.g., 'threads 0-3 hold rows 0-7, threads 4-7 hold rows 8-15')\n"
        "6. How elements are packed within registers (e.g., two f16 values per 32-bit register)\n\n"
        "Be extremely precise. This data is used by programmers to understand register allocation."
    ),
    "metadata_layout": (
        "This is a PTX ISA sparse MMA metadata layout figure. It shows how sparsity metadata "
        "is organized across threads for structured sparse matrix operations.\n\n"
        "Extract ALL of the following:\n"
        "1. The MMA shape (e.g., m16n8k32) and data types\n"
        "2. Which threads hold which metadata bits/bytes\n"
        "3. The metadata encoding format (which bits select which elements)\n"
        "4. The relationship between metadata values and the selected non-zero elements\n"
        "5. Any column/row grouping patterns\n\n"
        "Be extremely precise about bit positions and thread assignments."
    ),
    "shared_memory_layout": (
        "This is a PTX ISA shared memory layout figure showing how data is arranged in "
        "shared memory, possibly with swizzle patterns.\n\n"
        "Extract ALL of the following:\n"
        "1. The dimensions of the shared memory tile\n"
        "2. The swizzle pattern (e.g., 32B, 64B, 128B swizzle) if any\n"
        "3. How data elements map to shared memory addresses\n"
        "4. Bank conflict avoidance patterns\n"
        "5. Any color coding or grouping that shows the access pattern\n\n"
        "Describe the layout as precisely as possible including address calculations."
    ),
    "packing_format": (
        "This is a PTX ISA data packing format figure showing how sub-byte or narrow "
        "data types are packed into larger containers.\n\n"
        "Extract ALL of the following:\n"
        "1. The source data type and bit width (e.g., E2M1 = 4-bit, E3M2 = 6-bit)\n"
        "2. How elements are packed into containers (bytes, 16-bit, 32-bit)\n"
        "3. Bit positions of each element within the container\n"
        "4. Any padding bits and their positions\n"
        "5. Whether this is for shared memory, tensor memory, or registers\n\n"
        "Be precise about bit-level layout."
    ),
    "other_layout": (
        "This is a technical figure from the NVIDIA PTX ISA documentation. "
        "It shows a data layout, format, or memory organization pattern.\n\n"
        "Describe this figure in complete detail:\n"
        "1. What type of layout or format is shown\n"
        "2. All dimensions, sizes, and element positions\n"
        "3. How data is organized (row-major, column-major, tiled, etc.)\n"
        "4. Any address calculations or offset patterns\n"
        "5. Relationships between elements (grouping, interleaving, etc.)\n\n"
        "Extract every piece of information visible in the figure."
    ),
    "default": (
        "This is a technical figure from the NVIDIA PTX ISA documentation for GPU programming.\n\n"
        "Describe this figure in complete detail. Extract:\n"
        "1. What concept or data structure is being illustrated\n"
        "2. All labeled components and their relationships\n"
        "3. Any numerical values, dimensions, or sizes shown\n"
        "4. Data flow or organization patterns\n"
        "5. Any text labels, legends, or annotations\n\n"
        "Be thorough — this description will be used by programmers who cannot see the image."
    ),
}


def _categorize_figure(caption: str) -> str:
    lower = caption.lower()
    if "fragment layout" in lower or "register fragment" in lower:
        return "register_fragment_layout"
    elif "shared memory" in lower or "smem" in lower or "swizzle" in lower:
        return "shared_memory_layout"
    elif "metadata" in lower:
        return "metadata_layout"
    elif "packing" in lower:
        return "packing_format"
    elif "layout" in lower:
        return "other_layout"
    return "default"


async def _describe_one(
    client: AsyncOpenAI,
    image_url: str,
    caption: str,
    category: str,
    model: str,
) -> str:
    prompt = PROMPTS.get(category, PROMPTS["default"])
    full_prompt = f"Figure caption: \"{caption}\"\n\n{prompt}"

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": image_url},
                ],
            },
        ],
        temperature=0.2,
        max_tokens=2500,
    )
    return response.choices[0].message.content or ""


async def describe_all_figures(
    model: str = "anthropic/claude-sonnet-4.6",
    concurrency: int = 5,
):
    """Describe all PTX ISA figures using a VLM via fal.ai OpenRouter."""
    fal_key = os.environ.get("FAL_KEY")
    if not fal_key:
        raise ValueError("Set FAL_KEY environment variable")

    client = AsyncOpenAI(
        base_url="https://fal.run/openrouter/router/openai/v1",
        api_key="not-needed",
        default_headers={"Authorization": f"Key {fal_key}"},
    )

    # Parse the HTML to get all figures
    print("Fetching PTX ISA page...")
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as http:
        resp = await http.get(PTX_BASE_URL)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    figures = []
    for fig in soup.find_all("figure"):
        img = fig.find("img")
        cap = fig.find("figcaption")
        if not img:
            continue
        src = img.get("src", "")
        if not src or "_images/" not in src:
            continue

        caption_text = ""
        if cap:
            # Extract clean caption text
            spans = cap.find_all("span", class_="caption-text")
            if spans:
                caption_text = " ".join(s.get_text(strip=True) for s in spans)
            else:
                caption_text = cap.get_text(strip=True)

        figures.append({
            "filename": src.split("/")[-1],
            "url": PTX_BASE_URL + src,
            "caption": caption_text,
            "id": fig.get("id", ""),
            "category": _categorize_figure(caption_text),
        })

    print(f"Found {len(figures)} figures to describe")

    # Load existing descriptions (resumable)
    desc_path = DATA_DIR / "ptx" / "figure_descriptions.json"
    desc_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if desc_path.exists():
        existing = json.loads(desc_path.read_text())
        remaining = len(figures) - len(existing)
        print(f"  Already have {len(existing)} descriptions, {remaining} remaining...")

    sem = asyncio.Semaphore(concurrency)

    async def process_one(fig: dict) -> tuple[str, str]:
        if fig["filename"] in existing:
            return fig["filename"], existing[fig["filename"]]

        async with sem:
            try:
                desc = await _describe_one(
                    client, fig["url"], fig["caption"], fig["category"], model
                )
                print(f"  OK: {fig['filename'][:55]} [{fig['category']}]")
                return fig["filename"], desc
            except Exception as e:
                print(f"  FAIL: {fig['filename'][:55]} - {e}")
                return fig["filename"], ""

    # Process in batches and save after each
    batch_size = 10
    for i in range(0, len(figures), batch_size):
        batch = figures[i:i + batch_size]
        results = await asyncio.gather(*[process_one(f) for f in batch])
        for filename, desc in results:
            if desc:
                existing[filename] = desc
        desc_path.write_text(json.dumps(existing, indent=2))
        print(f"  Progress: {len(existing)}/{len(figures)}")

    print(f"\nDone! {len(existing)} figure descriptions saved")
    return desc_path


def inject_descriptions_into_index():
    """Replace image references in PTX index with VLM-generated descriptions."""
    desc_path = DATA_DIR / "ptx" / "figure_descriptions.json"
    if not desc_path.exists():
        print("No figure descriptions found. Run --describe-figures first.")
        return

    descriptions = json.loads(desc_path.read_text())
    print(f"Loaded {len(descriptions)} figure descriptions")

    updated = 0
    for json_file in sorted((DATA_DIR / "ptx").glob("*.json")):
        if json_file.name in ("index.json", "figure_descriptions.json"):
            continue

        data = json.loads(json_file.read_text())
        content = data.get("content", "")
        if "[Figure:" not in content:
            continue

        def replace_figure(match):
            filename = match.group(1)
            for key in [filename + ".png", filename + ".svg", filename + ".jpg", filename]:
                if key in descriptions:
                    desc = descriptions[key]
                    return f"\n\n[Figure: {filename}]\n{desc}\n"
            return match.group(0)

        new_content = re.sub(r'\[Figure:\s*([^\]]+)\]', replace_figure, content)

        if new_content != content:
            data["content"] = new_content
            json_file.write_text(json.dumps(data, indent=2))
            updated += 1

    print(f"Updated {updated} index files with figure descriptions")


if __name__ == "__main__":
    import sys
    model = "anthropic/claude-sonnet-4.6"
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]
    asyncio.run(describe_all_figures(model=model))
    inject_descriptions_into_index()
