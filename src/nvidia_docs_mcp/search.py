"""Search engine for the scraped NVIDIA documentation."""

import json
import math
import re
from pathlib import Path

def _find_data_dir() -> Path:
    """Locate the data directory, checking multiple possible locations."""
    # 1. Relative to source (editable install / dev mode)
    candidate = Path(__file__).parent.parent.parent / "data"
    if (candidate / "ptx" / "index.json").exists():
        return candidate
    # 2. Relative to CWD (running from repo root)
    candidate = Path.cwd() / "data"
    if (candidate / "ptx" / "index.json").exists():
        return candidate
    # 3. Fall back to source-relative (will error on load with clear message)
    return Path(__file__).parent.parent.parent / "data"


DATA_DIR = _find_data_dir()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_.]+", text.lower())


class DocIndex:
    def __init__(self, doc_type: str):
        self.doc_dir = DATA_DIR / doc_type
        self.index: list[dict] = []
        self.documents: dict[str, dict] = {}
        self._idf_cache: dict[str, float] = {}
        self._doc_tokens: dict[str, list[str]] = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        index_path = self.doc_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"No index found at {index_path}. Run `nvidia-docs-mcp --scrape` first."
            )
        self.index = json.loads(index_path.read_text())

        # Load all documents and pre-tokenize
        for entry in self.index:
            filepath = self.doc_dir / entry["file"]
            if filepath.exists():
                doc = json.loads(filepath.read_text())
                self.documents[entry["file"]] = doc
                # Build searchable text from all fields
                searchable = " ".join([
                    doc.get("heading", ""),
                    doc.get("instruction", ""),
                    doc.get("content", ""),
                    " ".join(doc.get("architectures", [])),
                    doc.get("source_page", ""),
                ])
                self._doc_tokens[entry["file"]] = _tokenize(searchable)

        # Pre-compute IDF
        n = len(self._doc_tokens)
        if n > 0:
            term_doc_count: dict[str, int] = {}
            for tokens in self._doc_tokens.values():
                for term in set(tokens):
                    term_doc_count[term] = term_doc_count.get(term, 0) + 1
            for term, count in term_doc_count.items():
                self._idf_cache[term] = math.log((n + 1) / (count + 1)) + 1

        self._loaded = True

    def search(self, query: str, top_k: int = 10, arch_filter: str | None = None) -> list[dict]:
        self.load()
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: dict[str, float] = {}
        for filename, doc_tokens in self._doc_tokens.items():
            if arch_filter:
                doc = self.documents[filename]
                doc_archs = doc.get("architectures", [])
                if doc_archs and arch_filter.lower() not in [a.lower() for a in doc_archs]:
                    continue

            token_set = set(doc_tokens)
            token_counts = {}
            for t in doc_tokens:
                token_counts[t] = token_counts.get(t, 0) + 1

            score = 0.0
            for qt in query_tokens:
                # Exact match
                if qt in token_set:
                    tf = token_counts[qt] / len(doc_tokens)
                    idf = self._idf_cache.get(qt, 1.0)
                    score += tf * idf
                else:
                    # Partial/prefix match (for things like "wgmma" matching "wgmma.mma_async")
                    for dt in token_set:
                        if qt in dt or dt in qt:
                            tf = token_counts[dt] / len(doc_tokens)
                            idf = self._idf_cache.get(dt, 1.0)
                            score += tf * idf * 0.5
                            break

            # Boost heading/instruction matches
            doc = self.documents[filename]
            heading_lower = doc.get("heading", "").lower()
            instr_lower = doc.get("instruction", "").lower()
            for qt in query_tokens:
                if qt in heading_lower:
                    score *= 2.0
                if qt in instr_lower:
                    score *= 3.0

            # Demote niche/specialized variants unless explicitly queried.
            # Dense is the general case. Sparse, convolution, weight-stationary
            # are specializations that should only rank high when asked for.
            query_lower = " ".join(query_tokens)
            niche_terms = {
                "sparse": ["sparse", ".sp", "sparsity", "ordered_metadata"],
                "convolution": ["convolution", "conv", "collector", "ashift", "im2col"],
                "weight_stationary": [".ws", "weight_stationary"],
            }
            is_niche = False
            for niche, keywords in niche_terms.items():
                if not any(k in query_lower for k in keywords):
                    check_text = heading_lower + " " + instr_lower
                    if any(k in check_text for k in keywords):
                        is_niche = True
                        break

            if is_niche:
                # Cap the score — niche results should never outrank generic ones
                # unless the user explicitly asks for them
                score = min(score, 0.01)

            if score > 0:
                scores[filename] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for filename, score in ranked:
            doc = self.documents[filename]
            results.append({
                "heading": doc.get("heading", ""),
                "instruction": doc.get("instruction", ""),
                "architectures": doc.get("architectures", []),
                "content": doc.get("content", ""),
                "source_page": doc.get("source_page", ""),
                "score": round(score, 4),
            })
        return results

    def get_instruction(self, name: str, arch: str | None = None) -> list[dict]:
        """Look up a PTX instruction by exact or partial name."""
        self.load()
        name_lower = name.lower().strip()
        results = []

        for entry in self.index:
            doc = self.documents.get(entry["file"])
            if not doc:
                continue
            instr = doc.get("instruction", "").lower()
            heading = doc.get("heading", "").lower()

            if name_lower == instr or name_lower in instr or instr.startswith(name_lower):
                if arch:
                    doc_archs = doc.get("architectures", [])
                    if doc_archs and arch.lower() not in [a.lower() for a in doc_archs]:
                        continue
                results.append({
                    "heading": doc.get("heading", ""),
                    "instruction": doc.get("instruction", ""),
                    "architectures": doc.get("architectures", []),
                    "content": doc.get("content", ""),
                })

        return results


    def get_source(self, module_or_file: str) -> dict | None:
        """Get a source file or module by path or module name."""
        self.load()
        query = module_or_file.lower().strip()
        for entry in self.index:
            doc = self.documents.get(entry["file"])
            if not doc:
                continue
            module = doc.get("module", "").lower()
            file_path = doc.get("file_path", "").lower()
            if query == module or query in module or query == file_path or query in file_path:
                return doc
        return None

    def list_modules(self, filter_str: str | None = None) -> list[dict]:
        """List available modules/examples."""
        self.load()
        results = []
        for entry in self.index:
            doc = self.documents.get(entry["file"])
            if not doc:
                continue
            if filter_str:
                searchable = " ".join([
                    doc.get("module", ""),
                    doc.get("file_path", ""),
                    doc.get("category", ""),
                    " ".join(doc.get("symbols", [])),
                    doc.get("heading", ""),
                ]).lower()
                if filter_str.lower() not in searchable:
                    continue
            results.append({
                "heading": doc.get("heading", ""),
                "module": doc.get("module", ""),
                "file_path": doc.get("file_path", ""),
                "category": doc.get("category", ""),
                "type": doc.get("type", ""),
                "symbols": doc.get("symbols", []),
            })
        return results


# Singleton instances
ptx_index = DocIndex("ptx")
cutedsl_index = DocIndex("cutedsl")
cutedsl_source_index = DocIndex("cutedsl_source")
