import json
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from .core import (
    SpatialGenerationError,
    all_unit_moves,
    build_held_out_payload,
    canonical_adjacencies,
    choose_weighted_item,
    direction_text,
    is_identifiable_heuristic,
    shape,
    token_at,
    token_coord_xy,
)

_DEMONSTRATION_MAX_ATTEMPTS = 600
_TRACE_MAX_RETRIES = 320
_TOKEN_COVERAGE_RATIO = 0.90
_SYMMETRIC_DUPLICATE_RATIO = 0.35
_TRACE_PATHS_BY_DIFFICULTY = {1: (4, 6), 2: (5, 7), 3: (6, 8)}
_TRACE_PATH_LENGTHS_BY_DIFFICULTY = {1: (3, 5), 2: (4, 6), 3: (5, 7)}
_TRACE_NEW_TOKEN_BONUS = 3.0
_TRACE_NEW_EDGE_BONUS = 2.0
_TRACE_NEW_NODE_BONUS = 1.5
_TRACE_IMMEDIATE_BACKTRACK_FACTOR = 0.10
_TRACE_SHORT_CYCLE_FACTOR = 0.35
_TRACE_EDGE_REUSE_PENALTY = 0.65
_TRACE_NODE_REVISIT_PENALTY = 0.20

TraceEdge = Tuple[str, int, int, str]


def build_context_diagnostics(
    context_text: str,
    line_count: int,
    layout_identifiable: bool,
    rejection_counts: Dict[str, int],
) -> Dict[str, Any]:
    """Create standard context diagnostics shared by all formatters."""
    return {
        "layout_identifiable": layout_identifiable,
        "prompt_char_len": len(context_text),
        "prompt_token_estimate": max(1, len(context_text) // 4),
        "prompt_line_count": line_count,
        "rejection_counts": rejection_counts,
    }


def evaluate_demonstration_sample(
    struct: Dict[str, Any],
    sample: List[Tuple[str, int, int, str]],
    all_tokens: Set[str],
    all_rows: Set[int],
    all_cols: Set[int],
) -> str | None:
    """Return first failed demonstration gate key, or None on success."""
    if not is_identifiable_heuristic(struct, sample):
        return "insufficient_identifiability"

    pair_dirs: Dict[frozenset[str], Set[Tuple[int, int]]] = defaultdict(set)
    for a, dx, dy, b in sample:
        pair_dirs[frozenset((a, b))].add((dx, dy))
    symmetric_dupes = sum(1 for dirs in pair_dirs.values() if len(dirs) > 1)
    if symmetric_dupes > int(_SYMMETRIC_DUPLICATE_RATIO * max(1, len(sample))):
        return "too_many_symmetric_duplicates"

    token_cov = {a for a, *_ in sample} | {b for *_x, b in sample}
    if len(token_cov) < int(_TOKEN_COVERAGE_RATIO * len(all_tokens)):
        return "insufficient_token_coverage"

    rows: Set[int] = set()
    cols: Set[int] = set()
    for src, _dx, _dy, dst in sample:
        for tok in (src, dst):
            coord = struct["token_map"][tok]
            cols.add(coord[0])
            rows.add(coord[1] if len(coord) > 1 else 0)
    if rows != all_rows:
        return "missing_row_coverage"
    if cols != all_cols:
        return "missing_column_coverage"
    return None


def sample_demonstration_edges(
    struct: Dict[str, Any],
    rng: random.Random,
) -> Tuple[List[Tuple[str, int, int, str]], Dict[str, int]]:
    """Sample demonstration edges that satisfy identifiability and coverage gates."""
    all_moves = all_unit_moves(struct)
    width, height = shape(struct)
    total = len(all_moves)
    rej = Counter()

    frac_by_diff = {1: 0.58, 2: 0.52, 3: 0.48}
    base_target = int(total * frac_by_diff.get(struct["difficulty"], 0.55))
    base_target = max(width + height + 2, base_target)
    base_target = min(base_target, total - 1) if total > 1 else 1

    all_tokens = set(struct["token_map"].keys())
    all_rows = set(range(height))
    all_cols = set(range(width))

    for attempt in range(_DEMONSTRATION_MAX_ATTEMPTS):
        target = min(total - 1, base_target + attempt // 40) if total > 1 else 1
        sample = rng.sample(all_moves, target)
        failure_key = evaluate_demonstration_sample(
            struct, sample, all_tokens, all_rows, all_cols
        )
        if failure_key is not None:
            rej[failure_key] += 1
            continue
        return sample, dict(rej)

    if not all_moves:
        raise SpatialGenerationError(
            "No adjacency moves available for demonstration context."
        )
    raise SpatialGenerationError("Failed to build identifiable demonstration context.")


def trace_start_candidates(
    struct: Dict[str, Any],
    token_list: List[str],
    covered: Set[str],
    rows_seen: Set[int],
    cols_seen: Set[int],
) -> List[str]:
    """Prefer starts that increase token, row, or column coverage."""
    candidates: List[str] = []
    for tok in token_list:
        coord = struct["token_map"][tok]
        x = coord[0]
        y = coord[1] if len(coord) > 1 else 0
        expands_coverage = (
            tok not in covered or y not in rows_seen or x not in cols_seen
        )
        if expands_coverage:
            candidates.append(tok)
    return candidates


def trace_weighted_options(
    options: List[TraceEdge],
    prev: List[str],
    covered: Set[str],
    seen: Set[TraceEdge],
    edge_count: Counter[TraceEdge],
    node_count: Counter[str],
    diagnostics: Counter[str],
) -> List[Tuple[float, TraceEdge]]:
    """Build weighted transition options.

    Weight policy:
    - reward unseen destinations, unseen edges, and new destination nodes;
    - downweight short cycles using recent-node history;
    - downweight repeated edges/destinations to keep path evidence diverse.
    """
    weighted: List[Tuple[float, TraceEdge]] = []
    for edge in options:
        _src, _dx, _dy, dst = edge
        weight = 1.0
        if dst not in covered:
            weight += _TRACE_NEW_TOKEN_BONUS
        if edge not in seen:
            weight += _TRACE_NEW_EDGE_BONUS
        if dst not in node_count:
            weight += _TRACE_NEW_NODE_BONUS
        if len(prev) >= 2 and dst == prev[-2]:
            weight *= _TRACE_IMMEDIATE_BACKTRACK_FACTOR
            diagnostics["immediate_backtrack_avoided"] += 1
        if len(prev) >= 3 and dst == prev[-3]:
            weight *= _TRACE_SHORT_CYCLE_FACTOR
            diagnostics["short_cycle_avoided"] += 1
        weight /= 1.0 + _TRACE_EDGE_REUSE_PENALTY * edge_count[edge]
        weight /= 1.0 + _TRACE_NODE_REVISIT_PENALTY * node_count[dst]
        weighted.append((weight, edge))
    return weighted


def record_trace_edge(
    struct: Dict[str, Any],
    edge: TraceEdge,
    seen_edges: List[TraceEdge],
    seen_unique: Set[TraceEdge],
    covered_tokens: Set[str],
    edge_count: Counter[TraceEdge],
    node_count: Counter[str],
    touched_rows: Set[int],
    touched_cols: Set[int],
) -> Tuple[int, int, str]:
    """Apply one sampled edge to trace state and return (dx, dy, dst)."""
    seen_edges.append(edge)
    seen_unique.add(edge)
    src, dx, dy, dst = edge
    covered_tokens.add(src)
    covered_tokens.add(dst)
    edge_count[edge] += 1
    node_count[src] += 1
    node_count[dst] += 1
    for tok in (src, dst):
        x, y = token_coord_xy(struct, tok)
        touched_cols.add(x)
        touched_rows.add(y)
    return dx, dy, dst


def sample_trace_paths(
    struct: Dict[str, Any],
    rng: random.Random,
) -> Tuple[
    List[List[TraceEdge]],
    List[TraceEdge],
    Set[Tuple[int, int]],
    Dict[str, int],
]:
    """Sample trace paths while balancing novelty, coverage, and anti-cycles."""
    width, height = shape(struct)
    all_moves = all_unit_moves(struct)
    edge_lookup: Dict[str, List[TraceEdge]] = defaultdict(list)
    for edge in all_moves:
        edge_lookup[edge[0]].append(edge)

    traces: List[List[TraceEdge]] = []
    seen_edges: List[TraceEdge] = []
    composed_vectors: Set[Tuple[int, int]] = set()
    n_paths = rng.randint(*_TRACE_PATHS_BY_DIFFICULTY.get(struct["difficulty"], (5, 7)))
    path_len_lo, path_len_hi = _TRACE_PATH_LENGTHS_BY_DIFFICULTY.get(
        struct["difficulty"], (4, 6)
    )
    tokens = list(struct["token_map"].keys())

    covered_tokens: Set[str] = set()
    seen_unique: Set[TraceEdge] = set()
    edge_count: Counter[TraceEdge] = Counter()
    node_count: Counter[str] = Counter()
    touched_rows: Set[int] = set()
    touched_cols: Set[int] = set()
    rej = Counter()
    for _ in range(n_paths):
        start_candidates = trace_start_candidates(
            struct, tokens, covered_tokens, touched_rows, touched_cols
        )
        cur = rng.choice(start_candidates) if start_candidates else rng.choice(tokens)
        path: List[TraceEdge] = []
        dx_sum = 0
        dy_sum = 0
        prev_nodes: List[str] = [cur]
        for _step in range(rng.randint(path_len_lo, path_len_hi)):
            options = edge_lookup[cur]
            if not options:
                break
            weighted = trace_weighted_options(
                options,
                prev_nodes,
                covered_tokens,
                seen_unique,
                edge_count,
                node_count,
                rej,
            )
            e = choose_weighted_item(weighted, rng)
            path.append(e)
            dx, dy, dst = record_trace_edge(
                struct,
                e,
                seen_edges,
                seen_unique,
                covered_tokens,
                edge_count,
                node_count,
                touched_rows,
                touched_cols,
            )
            dx_sum += dx
            dy_sum += dy
            prev_nodes.append(dst)
            if len(prev_nodes) > 3:
                prev_nodes.pop(0)
            cur = dst
        if path:
            traces.append(path)
            if dx_sum and dy_sum:
                composed_vectors.add((dx_sum, dy_sum))
    if touched_rows and len(touched_rows) < height:
        rej["missing_row_coverage"] += 1
    if touched_cols and len(touched_cols) < width:
        rej["missing_column_coverage"] += 1
    return traces, seen_edges, composed_vectors, dict(rej)


def fallback_trace_paths(
    struct: Dict[str, Any],
) -> Tuple[
    List[List[Tuple[str, int, int, str]]],
    List[Tuple[str, int, int, str]],
    Set[Tuple[int, int]],
]:
    """Deterministic backup trace construction to guarantee identifiability."""
    width, height = shape(struct)
    path: List[Tuple[str, int, int, str]] = []
    x = y = 0
    direction = 1
    for row in range(height):
        steps = (width - 1) if width > 1 else 0
        for _ in range(steps):
            nx = x + direction
            src = token_at(struct, x, y)
            dst = token_at(struct, nx, y)
            path.append((src, 1 if direction == 1 else -1, 0, dst))
            x = nx
        if row < height - 1:
            src = token_at(struct, x, y)
            dst = token_at(struct, x, y + 1)
            path.append((src, 0, 1, dst))
            y += 1
            direction *= -1
    composed = {(1, 1), (-1, 1)} if height > 1 and width > 1 else set()
    return [path], list(path), composed


def trace_coverage_ok(
    struct: Dict[str, Any], seen_edges: List[Tuple[str, int, int, str]]
) -> bool:
    """Require context edges to mention most tokens in the layout."""
    covered_tokens = {a for a, *_ in seen_edges} | {b for *_x, b in seen_edges}
    return len(covered_tokens) >= int(_TOKEN_COVERAGE_RATIO * len(struct["token_map"]))


def build_trace_context_edges(
    struct: Dict[str, Any], rng: random.Random
) -> Tuple[
    List[List[Tuple[str, int, int, str]]],
    List[Tuple[str, int, int, str]],
    Set[Tuple[int, int]],
    Dict[str, int],
]:
    """Resample trace evidence until identifiability and coverage are satisfied."""
    traces, seen_edges, composed, rejections = sample_trace_paths(struct, rng)
    for _ in range(_TRACE_MAX_RETRIES):
        identifiable = is_identifiable_heuristic(struct, seen_edges)
        coverage_ok = trace_coverage_ok(struct, seen_edges)
        ok = identifiable and coverage_ok
        if ok:
            break
        if not identifiable:
            rejections["insufficient_identifiability"] = (
                rejections.get("insufficient_identifiability", 0) + 1
            )
        if not coverage_ok:
            rejections["insufficient_token_coverage"] = (
                rejections.get("insufficient_token_coverage", 0) + 1
            )
        traces, seen_edges, composed, trial_rejections = sample_trace_paths(struct, rng)
        for key, value in trial_rejections.items():
            rejections[key] = rejections.get(key, 0) + value
    if not is_identifiable_heuristic(struct, seen_edges):
        traces, seen_edges, composed = fallback_trace_paths(struct)
        rejections["fallback_trace_used"] = rejections.get("fallback_trace_used", 0) + 1
    if not is_identifiable_heuristic(struct, seen_edges):
        raise SpatialGenerationError("Failed to build identifiable trace context.")
    return traces, seen_edges, composed, rejections


def format_demonstration(
    struct: Dict[str, Any], rng: random.Random
) -> Tuple[str, Dict[str, Any]]:
    """Format demonstration-style local movement examples."""
    shown, rejections = sample_demonstration_edges(struct, rng)
    rng.shuffle(shown)
    lines = [
        "Context:",
        "Local movement examples (the orientation is fixed: right increases x, down increases y):",
    ]
    for src, dx, dy, dst in shown:
        lines.append(f"From '{src}', a step {direction_text(dx, dy)} reaches '{dst}'.")
    payload = build_held_out_payload(struct, shown, context_kind="demonstration")
    context_text = "\n".join(lines) + "\n\n"
    payload["context_diagnostics"] = build_context_diagnostics(
        context_text=context_text,
        line_count=len(lines),
        layout_identifiable=is_identifiable_heuristic(struct, shown),
        rejection_counts=rejections,
    )
    return context_text, payload


def format_declarative_nl(
    struct: Dict[str, Any], _rng: random.Random
) -> Tuple[str, Dict[str, Any]]:
    """Format declarative adjacency statements with fixed orientation language."""
    width, height = shape(struct)
    _ = width
    shown = canonical_adjacencies(struct)
    lines = ["Context:"]
    if height == 1:
        leftmost = token_at(struct, 0, 0)
        lines.append(
            "The locations form a 1D line where x increases from left to right."
        )
        lines.append(f"'{leftmost}' is the leftmost token (x=0).")
    else:
        top_left = token_at(struct, 0, 0)
        lines.append(
            "The locations form a 2D grid where x increases left-to-right and y increases top-to-bottom."
        )
        lines.append(f"'{top_left}' is in the top-left corner at coordinate (0,0).")
    for a, dx, dy, b in shown:
        if dx == 1:
            lines.append(f"'{a}' is immediately left of '{b}'.")
        elif dy == 1:
            lines.append(f"'{a}' is directly above '{b}'.")

    payload = build_held_out_payload(
        struct, shown, context_kind="declarative_natural_language"
    )
    context_text = "\n".join(lines) + "\n\n"
    payload["context_diagnostics"] = build_context_diagnostics(
        context_text=context_text,
        line_count=len(lines),
        layout_identifiable=is_identifiable_heuristic(struct, shown),
        rejection_counts={},
    )
    return context_text, payload


def format_trace(
    struct: Dict[str, Any], rng: random.Random
) -> Tuple[str, Dict[str, Any]]:
    """Format a step-by-step trace context from sampled movement paths."""
    lines = ["Context:", "Step-by-step paths through the layout:"]
    traces, seen_edges, composed, rejections = build_trace_context_edges(struct, rng)

    for idx, path in enumerate(traces, start=1):
        parts = [f"Path {idx}: start at '{path[0][0]}'"]
        for _src, dx, dy, dst in path:
            parts.append(f"-> {direction_text(dx, dy)} -> '{dst}'")
        lines.append(" ".join(parts))

    payload = build_held_out_payload(
        struct, seen_edges, composed_vectors=composed, context_kind="trace"
    )
    context_text = "\n".join(lines) + "\n\n"
    payload["context_diagnostics"] = build_context_diagnostics(
        context_text=context_text,
        line_count=len(lines),
        layout_identifiable=is_identifiable_heuristic(struct, seen_edges),
        rejection_counts=rejections,
    )
    return context_text, payload


def format_formal_specification(
    struct: Dict[str, Any], _rng: random.Random
) -> Tuple[str, Dict[str, Any]]:
    """Format a fully explicit specification of coordinates and movement rules."""
    payload = {
        "grid_dimensions": struct["dimensions"],
        "coordinates": struct["token_map"],
        "movement_semantics": {
            "right": [1, 0],
            "left": [-1, 0],
            "down": [0, 1],
            "up": [0, -1],
        },
    }
    lines = [
        "Context:",
        "Coordinates use (x, y). x increases left-to-right. y increases top-to-bottom.",
        json.dumps(payload, indent=2),
    ]
    context = "\n".join(lines) + "\n\n"
    return context, {
        "held_out_moves": [],
        "context_vectors": [],
        "context_step_magnitudes": [],
        "context_composed_vectors": [],
        "context_edges": [],
        "context_tokens": sorted(struct["token_map"].keys()),
        "context_kind": "formal_specification",
        "context_diagnostics": build_context_diagnostics(
            context_text=context,
            line_count=len(lines),
            layout_identifiable=True,
            rejection_counts={},
        ),
    }


def format_spatial_translation(
    struct: Dict[str, Any],
    fmt: str,
    application: str,
    rng: random.Random,
) -> Tuple[str, Dict[str, Any]]:
    """Format a spatial translation instance for the requested context format.

    The application axis is currently ignored by this formatter interface.
    """
    _ = application
    formatters = {
        "demonstration": format_demonstration,
        "declarative_natural_language": format_declarative_nl,
        "trace": format_trace,
        "formal_specification": format_formal_specification,
    }
    return formatters[fmt](struct, rng)


__all__ = ["format_spatial_translation"]
