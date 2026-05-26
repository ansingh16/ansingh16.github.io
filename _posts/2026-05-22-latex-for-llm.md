---
title: 'LatexForLLM: Turning LaTeX Papers into Graphs for Smarter LLM Retrieval'
date: 2026-05-22
permalink: /posts/2026/05/latex-for-llm/
tags:
  - llm
  - latex
  - graphs
  - nlp
  - mcp
---

If you have ever pasted an entire research paper into ChatGPT or Claude and watched your token budget evaporate, you know the problem. A typical 10-page paper burns 8,000-12,000 tokens, yet the model only needs a few hundred to answer most questions about it. I built **LatexForLLM** to fix this. It parses LaTeX documents into a typed graph and retrieves only the sections, equations, and figures that matter. On benchmark tasks against a realistic 200-line paper, graph-based retrieval cuts word count by **~54% on average** (up to ~80% for focused queries) compared to pasting the full document.

The project is open-source: [github.com/ansingh16/LatexForLLM](https://github.com/ansingh16/LatexForLLM).

In this post I'll walk through the architecture and key implementation details: how the parser turns raw `.tex` into a node-edge schema, how the graph engine scores and expands queries, and how the whole thing plugs into Claude via the [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol).


## The Problem

Researchers routinely use LLMs to help with writing, reviewing, and understanding papers. The standard workflow is brute-force: dump the whole `.tex` file (or worse, the PDF) into the context window and hope the model picks out what is relevant. This is wasteful in a few ways. Most API-priced models charge per token, so sending 10K tokens when you need 500 is a 20x overspend. The more irrelevant text you include, the more likely the model is to hallucinate or lose focus on your actual question. And long papers can exceed the context window entirely, forcing you to manually split them.

What if, instead of sending the whole document, you could ask "give me everything related to energy conservation in Section 3" and get back only the relevant paragraphs, the equations they reference, and the figures they cite?

## Why a Graph and Not Just Chunking?

Most RAG systems split documents into fixed-size text chunks and embed them. This works reasonably well for prose, but it falls apart for scientific papers. Equations get split mid-expression, losing mathematical meaning. Cross-references break because a paragraph referring to "Equation 12" ends up in a different chunk than Equation 12 itself. And document structure is lost entirely, so there is no way to ask "give me everything in Section 3" because the chunker does not know where sections begin and end.

The graph approach preserves all of this. Equations are atomic nodes. Cross-references are edges. The section hierarchy is explicit. When you query for a paragraph, a graph walk naturally pulls in the equations, figures, and citations that paragraph depends on.


## Architecture Overview

The system has four layers:

```
┌─────────────────┐     ┌────────────────┐     ┌──────────────┐     ┌────────────┐
│  latexparser.py  │────>│   graph.py     │────>│ latex2graph  │     │ mcp_server │
│  (AST → nodes)   │     │ (nodes → graph │     │   (CLI)      │     │ (Claude    │
│                  │     │  + retrieval)  │     │              │     │  Desktop)  │
└─────────────────┘     └────────────────┘     └──────────────┘     └────────────┘
```

1. **`latexparser.py`** parses raw LaTeX into a canonical node schema (sections, paragraphs, equations, figures, tables, lists) with resolved labels and cross-references.
2. **`graph.py`** builds a `GraphStore` of typed nodes and edges, runs lexical or semantic search, expands structural context via graph walks, and produces token-budgeted payloads.
3. **`latex2graph.py`** is the CLI entry point that wires parsing, graph construction, querying, and export behind `argparse` subcommands.
4. **`mcp_server.py`** exposes four tools over the [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) so Claude Desktop can query the graph directly.


## Step 1: Parsing LaTeX into a Typed Node Schema

The parser is built on top of [pylatexenc](https://github.com/phfaist/pylatexenc), which provides a [`LatexWalker`](https://pylatexenc.readthedocs.io/en/latest/latexwalker/), a lightweight AST-style parser for LaTeX markup. Unlike a full TeX engine, `pylatexenc` does not compile the document. It walks the token stream and produces a tree of typed nodes: `LatexMacroNode`, `LatexEnvironmentNode`, `LatexGroupNode`, `LatexCharsNode`, etc.

The `RobustLatexParser` class wraps `LatexWalker` and applies domain-specific logic to turn the raw AST into a structured document schema.

### Multi-file resolution

Before parsing begins, a `_resolve_includes()` function inlines all `\input{}` and `\include{}` directives recursively. It tracks visited file paths to detect and skip circular includes:

```python
def _resolve_includes(tex_text, base_dir, visited=None):
    if visited is None:
        visited = set()
    _input_re = re.compile(r"\\(?:input|include)\{([^}]+)\}")

    def _inline(match):
        filename = match.group(1).strip()
        if not filename.endswith(".tex"):
            filename += ".tex"
        candidate = (base_dir / filename).resolve()
        if candidate in visited:
            warnings.append(f"Skipping circular \\input: {candidate}")
            return ""
        content = candidate.read_text(encoding="utf-8", errors="replace")
        child_text, child_warnings = _resolve_includes(
            content, base_dir, visited | {candidate}
        )
        return child_text

    return _input_re.sub(_inline, tex_text), warnings
```

This means a paper split across `intro.tex`, `methods.tex`, `results.tex` etc. is seamlessly concatenated before the AST walk begins.

### The AST walk

The core `_walk()` method traverses the `pylatexenc` node list. For each node it encounters, it dispatches based on type:

- **`LatexMacroNode`** with name in `{section, subsection, subsubsection}`: flush the current paragraph buffer, create a new section node, and push it onto a `section_stack` that tracks the nesting hierarchy. Parent-child relationships between sections are determined by comparing levels (section=1, subsection=2, subsubsection=3).

- **`LatexMacroNode`** with name `label`: attach the label to the most recently created structural node (equation, figure, table, or paragraph). This is tracked via `last_structural_node_id`.

- **`LatexMacroNode`** with name in `{ref, eqref, autoref, cref, Cref, cite, citep, citet}`: inject a `[REF:label]` marker into the text buffer. These markers are later extracted during paragraph flushing and converted into typed reference objects.

- **`LatexEnvironmentNode`** with name in `{equation, align, gather, multline}` (and their `*` variants): extract the raw LaTeX source, strip `\label{}` commands, and store as an equation node.

- **`LatexEnvironmentNode`** with name in `{figure, figure*, wrapfigure, SCfigure, marginfigure}`: extract the `\caption{}` text and any `\label{}` commands, store as a figure node.

- **`LatexEnvironmentNode`** with name in `{table, table*}`: same treatment as figures.

- **`LatexEnvironmentNode`** with name in `{itemize, enumerate}`: split by `\item` macros, extract text per item, store as a list node.

- **Inline math (`LatexMathNode`)**: call `latex_verbatim()` and append the raw `$...$` expression to the paragraph buffer. This preserves mathematical content inline. Without it, a sentence like "ranges from $\sim 500$ for run A3" would lose the math and become the garbled "ranges from for run A3".

- **Plain text (`LatexCharsNode`)**: accumulate into a paragraph buffer.

Paragraphs are flushed whenever a section heading or a float environment is encountered, or at the end of the node list. The `_flush_paragraph()` method extracts inline `[LABEL:...]` and `[REF:...]` markers, cleans the text, and only creates a node if the cleaned text exceeds 10 characters. This filters out stray whitespace and formatting artifacts.

### Label resolution

Every `\label{...}` encountered during the walk is recorded in a `label_map` dictionary (`label → node_id`). After the full walk completes, `_resolve_references()` iterates over all paragraph and list nodes, looking up each `[REF:label]` in the label map to fill in the `target_id` and `target_type` fields. References to labels not found in the document (typically bibliography keys from `\cite`) are assigned a `ref:label` ID and typed as `external_ref`.

### Output schema

The parser produces a dictionary with `schema_version`, a `nodes` block (keyed by type), and a `meta` block containing the label map, parse timing metrics, and any warnings:

```json
{
  "schema_version": "1.0",
  "nodes": {
    "sections": [...],
    "paragraphs": [...],
    "equations": [...],
    "figures": [...],
    "tables": [...],
    "lists": [...]
  },
  "meta": {
    "labels": {"eq:energy": "eq_a1b2c3d4", ...},
    "metrics": {"parse_duration_ms": 12.5, "node_counts": {...}},
    "warnings": [...]
  }
}
```

Each node carries an auto-generated ID with a type prefix (`sec_`, `para_`, `eq_`, `fig_`, `tbl_`, `list_`), the parent section ID, any labels, resolved references, and the `source_latex` verbatim text.


## Step 2: Building the Document Graph

The `LatexGraphSystem` class in `graph.py` takes the parser output and builds a `GraphStore` -- an adjacency structure of typed nodes and edges.

### Edge construction

`_build_edges()` creates four categories of edges from the parser output:

**Structural edges** -- For every non-section node (paragraph, equation, figure, table, list), if it has a `section` field, the system creates a bidirectional pair: `section --CONTAINS--> node` and `node --PART_OF--> section`. For nested sections, the same pattern applies between parent and child sections.

**Reference edges** -- For every paragraph or list node with `references`, the system creates `paragraph --MENTIONS--> target` and `target --USED_IN--> paragraph`. If the target label was not found in the document, a new `external_ref` node is created on the fly.

**Proximity edges** -- This is a heuristic that addresses a common pattern in scientific writing: authors describe an equation in prose without using an explicit `\ref{}`. The method `_build_proximity_equation_edges()` connects every paragraph to every equation in the same section via `MENTIONS`/`USED_IN` edges. Since `GraphStore.add_edge()` deduplicates, pairs already connected by an explicit `\ref` are not double-counted:

```python
def _build_proximity_equation_edges(self):
    section_to_equations = {}
    for node_id, node in self.graph.nodes.items():
        if node["type"] == NODE_EQUATION:
            section_id = node.get("section")
            if section_id:
                section_to_equations.setdefault(section_id, []).append(node_id)

    for node_id, node in self.graph.nodes.items():
        if node["type"] != NODE_PARAGRAPH:
            continue
        section_id = node.get("section")
        for eq_id in section_to_equations.get(section_id, []):
            self.graph.add_edge(node_id, eq_id, EDGE_MENTIONS)
            self.graph.add_edge(eq_id, node_id, EDGE_USED_IN)
```

This ensures that graph expansion can pull equations into the retrieved context even when the author wrote "the energy is given by" without a `\ref{eq:energy}`.

**Semantic edges (optional)** -- When `enable_semantic=True`, the system embeds all paragraph texts using [sentence-transformers](https://www.sbert.net/) (default model: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a 22M-parameter model producing 384-dimensional vectors). For each paragraph, it computes [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) against all other paragraphs and creates `SEMANTIC_SIMILAR` edges for the top-5 pairs exceeding a 0.55 threshold. The similarity score is encoded into the edge type string (e.g., `SEMANTIC_SIMILAR:0.78`), which the viewer and edge-ranking code can read.


## Step 3: Query and Retrieval

When you ask "spin alignment evidence", the retrieval pipeline goes through three stages:

### 3a. Candidate selection

The system supports two retrieval modes, chosen automatically:

**Lexical mode** (default, offline) -- `lexical_search_paragraphs()` tokenizes the query (lowercased, stopwords removed) and scores each paragraph using a weighted combination of three signals:

```python
score = 0.45 * text_overlap + 0.45 * section_title_overlap + 0.10 * term_density
```

- `text_overlap` = fraction of query terms found in the paragraph text
- `section_title_overlap` = fraction of query terms found in the parent section's title
- `term_density` = total occurrences of query terms divided by paragraph length

The section-title signal matters here. If you query for "energy equation", a paragraph in a section titled "Energy Conservation" gets a 0.45 boost even if the paragraph itself uses different terminology.

**Semantic mode** (requires `--extra semantic`) -- `search_paragraphs()` encodes the query with the same sentence-transformer model and ranks paragraphs by cosine similarity against their precomputed embeddings.

Both modes return the top-k paragraph IDs with scores.

### 3b. Structural expansion

The top-k paragraph IDs are then passed to `expand_structural_context()`, which performs a bounded [breadth-first search](https://en.wikipedia.org/wiki/Breadth-first_search) (BFS) over the graph edges. For each candidate paragraph, it walks `expansion_hops` (default 1) steps outward, collecting all reachable nodes and the edges connecting them.

Edges are ranked during expansion by a priority scheme:

```python
edge_priority = {
    MENTIONS:         0,   # highest -- cross-references
    USED_IN:          1,
    CONTAINS:         2,
    PART_OF:          3,
    SEMANTIC_SIMILAR: 4,   # lowest
}
```

and within each edge type, destination nodes are ranked by type:

```python
node_priority = {
    PARAGRAPH: 0,   # most useful
    EQUATION:  1,
    FIGURE:    2,
    TABLE:     3,
    LIST:      4,
    SECTION:   5,
    EXTERNAL_REF: 6,
}
```

This means when a query hits a paragraph that references `eq:energy` and lives in `sec:methods`, the expansion preferentially pulls in the equation (via `MENTIONS`) before pulling in the parent section (via `PART_OF`).

### 3c. Token-budgeted payload assembly

The final step is `build_structured_edit_payload()`. It takes all the nodes collected from candidate selection + expansion, orders them by retrieval priority (matched paragraphs first, expanded context second), and packs them into a payload that fits within a `token_budget`.

Token estimation uses a simple word-count heuristic (`len(content.split())`), not a real tokenizer, but close enough for budgeting purposes. Chunks are added greedily in priority order. If a chunk would exceed the remaining budget, it gets truncated word-by-word to fit:

```python
for node_id in ordered_ids:
    estimated = self.estimate_token_count(chunk)
    remaining_budget = token_budget - total_tokens
    if remaining_budget <= 0:
        omitted_node_ids.append(node_id)
        break
    if estimated > remaining_budget:
        chunk = self._truncate_chunk_to_budget(chunk, remaining_budget)
```

Each chunk in the payload carries the node content (text, LaTeX, caption, or list items), the source section, resolved references, neighbor IDs, and the original LaTeX source. This gives the LLM enough context to make safe edits without touching unrelated content.


## Step 4: MCP Server -- Plugging into Claude

The [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) (MCP) is an open standard from Anthropic for connecting LLMs to external tools and data sources. LatexForLLM ships an MCP server built with [FastMCP](https://gofastmcp.com/), a Python framework where you declare a tool with a decorated function and the schema, validation, and documentation are generated automatically.

The server (`mcp_server.py`) exposes four tools:

```python
@mcp.tool()
def load_document(file_path: str) -> dict:
    """Parse a LaTeX file and build its document graph."""
    key = _resolve(file_path)
    system, parser_output = _parse_and_build(key)
    _SYSTEMS[key] = system  # cache by absolute path
    return _document_summary(system, parser_output, key)

@mcp.tool()
def query_context(file_path: str, query: str, top_k: int = 5) -> dict:
    """Retrieve relevant sections for a query."""
    system = _get_or_load(file_path)
    ctx = system.get_query_context(query, top_k=top_k)
    return {
        "top_matches": _serialize_matches(ctx["top_matches"]),
        "nodes": _clean_nodes(ctx["nodes"]),      # strip source_latex
        "edges": _serialize_edges(ctx["edges"]),
    }

@mcp.tool()
def get_edit_payload(file_path, query, instruction, token_budget=1500):
    """Get token-budgeted edit context for targeted changes."""
    system = _get_or_load(file_path)
    payload = system.get_edit_payload(
        query=query, instruction=instruction, token_budget=token_budget
    )
    return {"chunks": _clean_chunks(payload["chunks"]), ...}

@mcp.tool()
def reload_document(file_path: str) -> dict:
    """Re-parse after edits, evicting the cached graph."""
    _SYSTEMS.pop(_resolve(file_path), None)
    # ... rebuild and cache
```

A few design choices I want to call out:

- **In-memory caching** -- Parsed graphs are cached by resolved absolute path in a module-level `_SYSTEMS` dict. The first `load_document` or `query_context` call builds the graph; subsequent calls are instant. `reload_document` evicts the cache after you save edits to disk.

- **Noise stripping** -- The `_clean_nodes()` and `_clean_chunks()` helpers strip `source_latex` and `original_latex` fields before returning results to Claude. Raw LaTeX is noisy and wastes tokens; the LLM only needs the cleaned text, equation LaTeX, and captions.

- **Automatic loading** -- `_get_or_load()` means you don't *have* to call `load_document` first. Any `query_context` or `get_edit_payload` call on an uncached file transparently triggers a parse-and-build.

To use it with Claude Desktop, add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "latexforllm": {
      "command": "uv",
      "args": ["run", "--extra", "mcp", "python", "mcp_server.py"],
      "cwd": "/path/to/LatexForLLM"
    }
  }
}
```

Now when you ask Claude "What does Section 4 say about metallicity gradients?", it calls `query_context` behind the scenes, retrieving only the relevant graph neighbourhood instead of your full paper.


## The Interactive Viewer

Graphs are easier to understand visually. The project includes a self-contained viewer (`viewer/index.html`), a [force-directed graph](https://en.wikipedia.org/wiki/Force-directed_graph_drawing) renderer written in vanilla JS + SVG.

### Force simulation

The layout uses a spring-electrical model inspired by [Fruchterman & Reingold (1991)](https://onlinelibrary.wiley.com/doi/10.1002/spe.4380211102). Every simulation tick computes three forces:

1. **Repulsion** -- All node pairs repel each other with force `F = repulsion * alpha / d²`, with a distance cutoff to avoid O(n²) blowup on large graphs. A collision guard pushes overlapping nodes apart.

2. **Spring attraction** -- Each edge acts as a spring with rest length and strength that vary by edge category: structural edges are short and stiff (120px, 0.065), reference edges are medium (200px, 0.025), and semantic edges are long and weak (260px, 0.012). This naturally groups tightly-connected nodes (same section) while keeping loosely-related nodes (semantic similarity) at a readable distance.

3. **Section gravity** -- A custom force that pulls nodes toward the centroid of their section cluster. This keeps section groupings visually coherent without requiring explicit layout constraints.

For large graphs (>80 nodes), the viewer pre-simulates synchronously (150-350 ticks) before the first render to avoid the chaotic initial animation. Force parameters (repulsion strength, spring scale, center gravity) adapt to graph size via a `scaledForce()` function.

### Interaction features

- **Ego-graph** (press `E`) -- runs a BFS from the selected node and hides everything beyond depth 1-3, using a slider to adjust. Useful for focusing on a single equation's neighborhood.
- **Shortest path** (Shift-click two nodes) -- BFS pathfinding highlights the connecting nodes and edges with hop count.
- **Section collapse** (double-click a section) -- builds a virtual graph where all children of the collapsed section merge into a single super-node. The underlying `buildVirtualGraph()` remaps edges so connections to/from collapsed children appear on the super-node instead.
- **Tree layout** (press `T`) -- switches from force-directed to a hierarchical layout built from the `CONTAINS` edges. A recursive `computeTreeLayout()` places leaves at the bottom and centers parents above their children.
- **Minimap** -- a canvas overlay that draws all node positions as colored dots (by type) and shows the current viewport rectangle. You can click-drag on the minimap to pan.
- **KaTeX rendering** -- equation nodes display their LaTeX content rendered via [KaTeX](https://katex.org/) in the detail panel.

All node types are color-coded (sections: gold, paragraphs: blue, equations: green, figures: pink, tables: tan, lists: lavender), and the node dimensions adapt to graph size. Smaller nodes for graphs with >80 nodes, larger for 15 or fewer.


## Benchmarking -- Honest Numbers

Early versions of the README claimed "90-97% token reduction." That was tautological. The benchmark fixtures were tiny (40-70 words), so the overhead of the prompt template dominated the raw prompt and the structured retrieval had almost nothing to return. Comparing a 79-word raw prompt to a 31-word structured one is not a meaningful measure of retrieval efficiency.

I've since rebuilt the benchmark to be honest. The estimator function is now called `estimate_word_count()` (not `estimate_prompt_tokens` -- it uses whitespace splitting, not a real tokenizer), and all output fields use `*_words` instead of `*_tokens`. The default token budget was raised from 120 to 1500 to match realistic usage.

Three new tasks target a 200-line fixture (`realistic_paper.tex`) with inline math, cross-references, tables, and figures -- the kind of document you would actually use this tool on. Here are the results:

```
Token Efficiency Benchmark (word-count estimate)

task                      | doc words | raw prompt | structured | saved | reduction
---                       | ---:      | ---:       | ---:       | ---:  | ---:
medium-equation           |        42 |         79 |         31 |    48 | 60.76%
messy-spin-alignment      |        46 |         85 |         65 |    20 | 23.53%
realistic-setup-table     |        69 |        108 |         67 |    41 | 37.96%
realistic-dynamo-scaling  |      1008 |       1051 |        395 |   656 | 62.42%
realistic-synchrotron     |      1008 |       1045 |        210 |   835 | 79.90%
realistic-amplification   |      1008 |       1047 |        415 |   632 | 60.36%

Summary: 6 tasks, 54.16% average reduction, 65.36% overall
```

The realistic fixture tasks show 60-80% reduction, and those are the numbers that matter. The small-fixture tasks (23-61%) are still included for regression testing, but they are not representative of real-world savings. On a 1000-word document, a focused query like "synchrotron polarization observations" retrieves only ~210 words, an 80% reduction. A broader query like "magnetic field amplification simulation results" still cuts ~60%.

The takeaway: graph-based retrieval reliably halves the prompt for realistic documents, and focused queries can do significantly better. It is not 90%, and claiming that would be dishonest.

Run it with:

```bash
latex2graph-benchmark --format table
```


## Offline by Default

Structural retrieval (lexical search + graph expansion) requires no network access and no GPU, just `numpy` and `pylatexenc`. The optional semantic mode uses `sentence-transformers`, which downloads the model on first run but can be pointed at a local path (`--model-path /path/to/model`) for fully air-gapped environments. The MCP server runs locally as a subprocess too, so no data leaves your machine.


## What's Next

A few things I want to tackle next:

- **Batch processing** for multi-paper literature review workflows
- **Diff-aware reloading** so only changed sections get reparsed
- **Integration with other editors** beyond Claude Desktop

If you work with LaTeX and LLMs, give it a try: [github.com/ansingh16/LatexForLLM](https://github.com/ansingh16/LatexForLLM). It's MIT-licensed, and contributions are welcome.


## References

- [pylatexenc](https://github.com/phfaist/pylatexenc) -- the LaTeX parser library used for AST walking
- [sentence-transformers (SBERT)](https://www.sbert.net/) -- sentence embedding framework; [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) is the default model
- [Model Context Protocol (MCP)](https://www.anthropic.com/news/model-context-protocol) -- Anthropic's open standard for AI-tool integration
- [FastMCP](https://gofastmcp.com/) -- the Pythonic framework for building MCP servers
- [Fruchterman & Reingold (1991)](https://onlinelibrary.wiley.com/doi/10.1002/spe.4380211102) -- "Graph Drawing by Force-Directed Placement", the algorithm inspiring the viewer layout
- [KaTeX](https://katex.org/) -- fast math typesetting for the web, used in the viewer's equation rendering
- [Force-directed graph drawing (Wikipedia)](https://en.wikipedia.org/wiki/Force-directed_graph_drawing) -- background on spring-electrical layout algorithms
