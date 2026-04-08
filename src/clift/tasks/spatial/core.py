"""Shared spatial translation primitives and generation logic."""

import random
from collections import defaultdict, deque
from typing import Any, Dict, List, Sequence, Set, Tuple

from clift.common import DOMAIN_WORDS, ITEM_WORDS, ROOM_WORDS, VARIABLE_WORDS

SPATIAL_DIFFICULTY = {
    1: {"dimensions": [16]},
    2: {"dimensions": [5, 5]},
    3: {"dimensions": [8, 8]},
}

_ALL_TOKEN_WORDS = ROOM_WORDS + ITEM_WORDS + DOMAIN_WORDS + VARIABLE_WORDS
_MIN_DEGREE2_FRACTION_2D = 0.8
_MIN_EDGE_EVIDENCE_EXTRA_2D = 2
_INVALID_BOUNDS = (0, -1, 0, -1)


class SpatialGenerationError(RuntimeError):
    """Raised when a valid spatial instance cannot be generated under constraints."""


def coord_key(coord: Sequence[int]) -> str:
    """Serialize a coordinate vector into the coordinate_map key format."""
    return ",".join(str(v) for v in coord)


def token_keys(struct: Dict[str, Any]) -> Set[str]:
    """Return the set of tokens present in a generated structure."""
    return set(struct["token_map"].keys())


def token_coord_xy(struct: Dict[str, Any], token: str) -> Tuple[int, int]:
    """Return token coordinates as (x, y), normalizing 1D layouts to y=0."""
    coord = struct["token_map"][token]
    return coord[0], (coord[1] if len(coord) > 1 else 0)


def shape(struct: Dict[str, Any]) -> Tuple[int, int]:
    """Return structure dimensions as (width, height)."""
    dims = struct["dimensions"]
    if len(dims) == 1:
        return dims[0], 1
    return dims[0], dims[1]


def coord_xy(struct: Dict[str, Any], x: int, y: int) -> List[int]:
    """Encode (x, y) for the structure dimensionality."""
    _w, h = shape(struct)
    return [x] if h == 1 else [x, y]


def token_at(struct: Dict[str, Any], x: int, y: int) -> str:
    """Return the token placed at an absolute grid position."""
    return struct["coordinate_map"][coord_key(coord_xy(struct, x, y))]


def step_phrase(n: int, direction: str) -> str:
    """Render a step count and direction with correct pluralization."""
    return f"{n} step{'s' if n != 1 else ''} {direction}"


def vector_text(dx: int, dy: int) -> str:
    """Render a movement vector as natural-language steps."""
    parts: List[str] = []
    if dx > 0:
        parts.append(step_phrase(dx, "right"))
    elif dx < 0:
        parts.append(step_phrase(abs(dx), "left"))
    if dy > 0:
        parts.append(step_phrase(dy, "down"))
    elif dy < 0:
        parts.append(step_phrase(abs(dy), "up"))
    return " and ".join(parts)


def direction_text(dx: int, dy: int) -> str:
    """Render a unit axis step as a direction word."""
    if dx == 1 and dy == 0:
        return "right"
    if dx == -1 and dy == 0:
        return "left"
    if dx == 0 and dy == 1:
        return "down"
    if dx == 0 and dy == -1:
        return "up"
    raise ValueError(f"Non-unit direction for trace edge: {(dx, dy)}")


def valid_start_ranges(
    width: int, height: int, dx: int, dy: int
) -> Tuple[range, range]:
    """Return valid origin ranges for applying a movement vector."""
    if dx >= 0:
        x_range = range(0, width - dx)
    else:
        x_range = range(-dx, width)
    if dy >= 0:
        y_range = range(0, height - dy)
    else:
        y_range = range(-dy, height)
    return x_range, y_range


def apply_vector(x: int, y: int, dx: int, dy: int) -> Tuple[int, int]:
    """Apply a movement vector to a coordinate."""
    return x + dx, y + dy


def choose_weighted_item[T](weighted: List[Tuple[float, T]], rng: random.Random) -> T:
    """Roulette-sample one weighted item from precomputed weights."""
    total_w = sum(weight for weight, _item in weighted)
    r = rng.random() * total_w
    acc = 0.0
    chosen = weighted[-1][1]
    for weight, candidate in weighted:
        acc += weight
        if acc >= r:
            chosen = candidate
            break
    return chosen


def all_unit_moves(struct: Dict[str, Any]) -> List[Tuple[str, int, int, str]]:
    """Enumerate all valid one-step directed moves in the layout."""
    width, height = shape(struct)
    moves: List[Tuple[str, int, int, str]] = []
    for y in range(height):
        for x in range(width):
            src = token_at(struct, x, y)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = apply_vector(x, y, dx, dy)
                if 0 <= nx < width and 0 <= ny < height:
                    dst = token_at(struct, nx, ny)
                    moves.append((src, dx, dy, dst))
    return moves


def canonical_adjacencies(struct: Dict[str, Any]) -> List[Tuple[str, int, int, str]]:
    """Canonical adjacency set: right/down only to avoid symmetric duplicates."""
    width, height = shape(struct)
    edges: List[Tuple[str, int, int, str]] = []
    for y in range(height):
        for x in range(width):
            src = token_at(struct, x, y)
            if x + 1 < width:
                edges.append((src, 1, 0, token_at(struct, x + 1, y)))
            if y + 1 < height:
                edges.append((src, 0, 1, token_at(struct, x, y + 1)))
    return edges


def generate_spatial_translation(seed: int = 42, difficulty: int = 1) -> Dict[str, Any]:
    """Generate token placement over a latent one-dimensional or two-dimensional grid."""
    rng = random.Random(seed)
    dimensions = SPATIAL_DIFFICULTY[difficulty]["dimensions"]
    width, height = (
        (dimensions[0], 1) if len(dimensions) == 1 else (dimensions[0], dimensions[1])
    )

    n_cells = width * height
    if n_cells > len(_ALL_TOKEN_WORDS):
        raise ValueError(f"Insufficient unique token words for {n_cells} cells.")

    tokens = rng.sample(_ALL_TOKEN_WORDS, n_cells)
    token_map: Dict[str, List[int]] = {}
    coordinate_map: Dict[str, str] = {}
    i = 0
    for y in range(height):
        for x in range(width):
            token = tokens[i]
            coord = [x] if len(dimensions) == 1 else [x, y]
            token_map[token] = coord
            coordinate_map[coord_key(coord)] = token
            i += 1

    return {
        "task_type": "spatial_translation",
        "difficulty": difficulty,
        "dimensions": dimensions,
        "token_map": token_map,
        "coordinate_map": coordinate_map,
    }


def is_connected(tokens: Set[str], edges: List[Tuple[str, int, int, str]]) -> bool:
    """Check graph connectivity over undirected token adjacency."""
    if not tokens:
        return False
    g: Dict[str, Set[str]] = defaultdict(set)
    for a, _dx, _dy, b in edges:
        g[a].add(b)
        g[b].add(a)
    start = next(iter(tokens))
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        for nxt in g[cur]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen == tokens


def unique_edges(
    edges: List[Tuple[str, int, int, str]],
) -> List[Tuple[str, int, int, str]]:
    """Drop duplicate directed edges while preserving first occurrence order."""
    return list(dict.fromkeys(edges))


def token_true_degree(struct: Dict[str, Any], token: str) -> int:
    """Return the true neighborhood size in the latent grid."""
    x, y = token_coord_xy(struct, token)
    width, height = shape(struct)
    deg = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = apply_vector(x, y, dx, dy)
        if 0 <= nx < width and 0 <= ny < height:
            deg += 1
    return deg


def summarize_evidence(
    struct: Dict[str, Any],
    edges: List[Tuple[str, int, int, str]],
) -> Tuple[
    Set[str],
    List[Tuple[str, int, int, str]],
    Dict[str, Set[str]],
    Set[Tuple[int, int]],
]:
    """Build reusable graph summaries for identifiability checks."""
    tokens = token_keys(struct)
    uedges = unique_edges(edges)
    undirected: Dict[str, Set[str]] = defaultdict(set)
    for a, _dx, _dy, b in uedges:
        undirected[a].add(b)
        undirected[b].add(a)
    vecs = {(dx, dy) for _a, dx, dy, _b in uedges}
    return tokens, uedges, undirected, vecs


def is_identifiable_1d(
    struct: Dict[str, Any],
    tokens: Set[str],
    undirected: Dict[str, Set[str]],
    vecs: Set[Tuple[int, int]],
) -> bool:
    """Evaluate 1D evidence sufficiency for layout reconstruction."""
    for tok in tokens:
        obs_deg = len(undirected[tok])
        true_deg = token_true_degree(struct, tok)
        min_req = 1 if true_deg == 1 else 2
        if obs_deg < min_req:
            return False
    return any(v in vecs for v in {(1, 0), (-1, 0)})


def is_identifiable_2d(
    width: int,
    height: int,
    tokens: Set[str],
    uedges: List[Tuple[str, int, int, str]],
    undirected: Dict[str, Set[str]],
    vecs: Set[Tuple[int, int]],
) -> bool:
    """Evaluate 2D evidence sufficiency for layout reconstruction."""
    degs = [len(undirected[tok]) for tok in tokens]
    if any(d < 1 for d in degs):
        return False
    if sum(1 for d in degs if d >= 2) < int(_MIN_DEGREE2_FRACTION_2D * len(tokens)):
        return False
    has_x = any(dx != 0 for dx, _dy in vecs)
    has_y = any(dy != 0 for _dx, dy in vecs)
    return (
        has_x and has_y and len(uedges) >= width + height + _MIN_EDGE_EVIDENCE_EXTRA_2D
    )


def is_identifiable_heuristic(
    struct: Dict[str, Any], edges: List[Tuple[str, int, int, str]]
) -> bool:
    """Check whether edge evidence is sufficient to resolve token layout."""
    width, height = shape(struct)
    tokens, uedges, undirected, vecs = summarize_evidence(struct, edges)
    if not uedges:
        return False

    appeared = {a for a, *_ in uedges} | {b for *_x, b in uedges}
    if appeared != tokens:
        return False
    if not is_connected(tokens, uedges):
        return False

    if height == 1:
        return is_identifiable_1d(struct, tokens, undirected, vecs)
    return is_identifiable_2d(width, height, tokens, uedges, undirected, vecs)


def solve_relative_layout_bounds(
    struct: Dict[str, Any], edges: List[Tuple[str, int, int, str]]
) -> Tuple[bool, Dict[str, Tuple[int, int]], Tuple[int, int, int, int]]:
    """Solve token positions and feasible global translation bounds."""
    tokens = token_keys(struct)
    if not tokens:
        return False, {}, _INVALID_BOUNDS
    adj: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
    for a, dx, dy, b in unique_edges(edges):
        adj[a].append((b, dx, dy))
        adj[b].append((a, -dx, -dy))
    if not is_connected(tokens, edges):
        return False, {}, _INVALID_BOUNDS

    root = next(iter(tokens))
    rel: Dict[str, Tuple[int, int]] = {root: (0, 0)}
    q = deque([root])
    while q:
        cur = q.popleft()
        cx, cy = rel[cur]
        for nxt, dx, dy in adj[cur]:
            cand = (cx + dx, cy + dy)
            if nxt in rel:
                if rel[nxt] != cand:
                    return False, {}, _INVALID_BOUNDS
                continue
            rel[nxt] = cand
            q.append(nxt)

    if set(rel.keys()) != tokens:
        return False, {}, _INVALID_BOUNDS

    width, height = shape(struct)
    xs = [v[0] for v in rel.values()]
    ys = [v[1] for v in rel.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    sx_lo, sx_hi = -min_x, width - 1 - max_x
    sy_lo, sy_hi = -min_y, height - 1 - max_y
    if sx_lo > sx_hi or sy_lo > sy_hi:
        return False, {}, _INVALID_BOUNDS
    return True, rel, (sx_lo, sx_hi, sy_lo, sy_hi)


def answer_is_unique_under_evidence(
    struct: Dict[str, Any],
    edges: List[Tuple[str, int, int, str]],
    *,
    application: str,
    token: str,
    dx: int,
    dy: int,
) -> bool:
    """Return whether the implied answer token is uniquely determined by evidence."""
    ok, rel, _bounds = solve_relative_layout_bounds(struct, edges)
    if not ok or token not in rel:
        return False

    tx, ty = rel[token]
    if application == "inverse":
        target_rel = (tx - dx, ty - dy)
    else:
        target_rel = (tx + dx, ty + dy)
    rel_to_tok = {xy: tok for tok, xy in rel.items()}
    return target_rel in rel_to_tok


def build_held_out_payload(
    struct: Dict[str, Any],
    shown_edges: List[Tuple[str, int, int, str]],
    composed_vectors: Set[Tuple[int, int]] | None = None,
    context_kind: str = "",
) -> Dict[str, Any]:
    """Build held-out metadata shared across formatting and probing steps."""
    all_moves = all_unit_moves(struct)
    shown_set = set(shown_edges)
    held_out = [m for m in all_moves if m not in shown_set]
    context_vectors = sorted({(dx, dy) for _a, dx, dy, _b in shown_edges})
    context_magnitudes = sorted(
        {max(abs(dx), abs(dy)) for _a, dx, dy, _b in shown_edges if dx or dy}
    )
    return {
        "held_out_moves": held_out,
        "held_out_unit_moves": held_out,
        "context_vectors": context_vectors,
        "context_step_magnitudes": context_magnitudes,
        "context_composed_vectors": sorted(composed_vectors or set()),
        "context_edges": unique_edges(shown_edges),
        "context_tokens": sorted(
            {a for a, *_ in shown_edges} | {b for *_x, b in shown_edges}
        ),
        "context_kind": context_kind,
    }


__all__ = [
    "SPATIAL_DIFFICULTY",
    "SpatialGenerationError",
    "generate_spatial_translation",
]
