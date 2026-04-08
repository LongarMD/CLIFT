import json
import random
from typing import Any, Dict, List, Tuple

from clift.common import ITEM_WORDS, mod_inverse, permutation_cycle_info

# ============================================================================
# LOOKUP TABLE TASK
# ============================================================================

LOOKUP_DIFFICULTY = {
    1: {"n_symbols": 4, "n_shown": 3},
    2: {"n_symbols": 6, "n_shown": 4},
    3: {"n_symbols": 8, "n_shown": 5},
}


def generate_lookup_table(seed: int = 42, difficulty: int = 1) -> Dict[str, Any]:
    """Generate a random bijective permutation on a symbol set.

    Maps a sampled subset of animal names to the same set. The full image
    ordering is shuffled so it is not the identity list; individual symbols
    may still map to themselves (fixed points).
    """
    rng = random.Random(seed)
    n = LOOKUP_DIFFICULTY[difficulty]["n_symbols"]
    symbols = rng.sample(ITEM_WORDS, n)
    perm = list(symbols)
    rng.shuffle(perm)
    while perm == symbols:
        rng.shuffle(perm)
    mapping = dict(zip(symbols, perm))
    return {
        "task_type": "lookup_table",
        "difficulty": difficulty,
        "symbols": symbols,
        "mapping": mapping,
    }


# --- Formatters -----------------------------------------------------------


def _lookup_format_few_shot(
    struct: Dict[str, Any], rng: random.Random
) -> Tuple[str, List[Any]]:
    mapping = struct["mapping"]
    symbols = struct["symbols"]
    n_shown = LOOKUP_DIFFICULTY[struct["difficulty"]]["n_shown"]
    pairs = [(s, mapping[s]) for s in symbols]
    rng.shuffle(pairs)
    shown = pairs[:n_shown]
    held_out = pairs[n_shown:]
    lines = [f"Input: {s} → Output: {t}" for s, t in shown]
    return "\n".join(lines) + "\n\n", held_out


def _lookup_format_natural(
    struct: Dict[str, Any], rng: random.Random
) -> Tuple[str, List[Any]]:
    mapping = struct["mapping"]
    items = list(mapping.items())
    rng.shuffle(items)
    descs = "; ".join(f"'{s}' becomes '{t}'" for s, t in items)
    ctx = (
        "A secret code book translates each symbol into a different symbol. "
        f"The translations are: {descs}.\n\n"
    )
    return ctx, []


def _lookup_format_step_by_step(
    struct: Dict[str, Any], rng: random.Random
) -> Tuple[str, List[Any]]:
    mapping = struct["mapping"]
    symbols = list(mapping.keys())
    rng.shuffle(symbols)
    n_trace = min(len(symbols), 3 + struct["difficulty"])
    trace_symbols = symbols[:n_trace]
    traces = []
    for s in trace_symbols:
        traces.append(
            f"Encode '{s}':\n"
            f"  Step 1: Find '{s}' in the code book.\n"
            f"  Step 2: The entry reads {s} → {mapping[s]}.\n"
            f"  Result: {mapping[s]}"
        )
    return "\n\n".join(traces) + "\n\n", []


def _lookup_format_structured(
    struct: Dict[str, Any], rng: random.Random
) -> Tuple[str, List[Any]]:
    mapping = struct["mapping"]
    ctx = "Code book:\n```json\n" + json.dumps(mapping, indent=2) + "\n```\n\n"
    return ctx, []


def format_lookup(
    struct: Dict[str, Any], fmt: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    """Format a lookup table instance according to the specified format."""
    formatters = {
        "demonstration": _lookup_format_few_shot,
        "natural_language": _lookup_format_natural,
        "trace": _lookup_format_step_by_step,
        "formal_spec": _lookup_format_structured,
    }
    return formatters[fmt](struct, rng)


# --- Probes ---------------------------------------------------------------


def _lookup_probe_forward(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    mapping = struct["mapping"]
    if held_out:
        src, tgt = held_out[0]
    else:
        src = rng.choice(struct["symbols"])
        tgt = mapping[src]
    return f"Question: What does '{src}' map to? Give only the one word.\nAnswer: ", tgt


def _lookup_probe_inverse(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    mapping = struct["mapping"]
    inv = {v: k for k, v in mapping.items()}
    tgt = rng.choice(list(inv.keys()))
    return (
        f"Question: Which symbol maps to '{tgt}'? Give only the one word.\nAnswer: ",
        inv[tgt],
    )


def _lookup_probe_articulation(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    mapping = struct["mapping"]
    parts = ", ".join(f"{s} → {t}" for s, t in sorted(mapping.items()))
    return (
        "Question: List the complete mapping in the form a → b, c → d, ... One line only.\nAnswer: ",
        parts,
    )


def _lookup_probe_ood(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    """Decode a sequence — given encoded symbols, recover the originals.

    This is OOD because the context demonstrates encoding (forward direction),
    but this probe asks for batch decoding (inverse on multiple symbols).
    """
    mapping = struct["mapping"]
    symbols = list(mapping.keys())
    rng.shuffle(symbols)
    k = min(len(symbols), 2 + struct["difficulty"])
    originals = symbols[:k]
    encoded = [mapping[s] for s in originals]
    enc_str = ", ".join(encoded)
    orig_str = ", ".join(originals)
    return (
        f"Question: The following symbols were encoded using the code book: [{enc_str}]. "
        f"Decode them back to the originals. Answer with the list in form [word1, word2, ...]. One line only.\nAnswer: ",
        f"[{orig_str}]",
    )


def _lookup_probe_planning(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    mapping = struct["mapping"]
    start = rng.choice(struct["symbols"])
    k = 2 + struct["difficulty"]
    current = start
    for _ in range(k):
        current = mapping[current]
    q = (
        f"Question: If you apply the mapping {k} times starting from "
        f"'{start}', what do you get? Give only the one word.\nAnswer: "
    )
    return q, current


def _lookup_probe_structural(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    mapping = struct["mapping"]
    fixed, _ = permutation_cycle_info(mapping)
    return (
        "Question: How many symbols are mapped to themselves (fixed points)? Give only the exact integer.\nAnswer: ",
        str(fixed),
    )


def probe_lookup(
    struct: Dict[str, Any],
    held_out: List[Any],
    application: str,
    rng: random.Random,
) -> Tuple[str, str]:
    """Generate a probe question for the lookup table task."""
    probes = {
        "forward": _lookup_probe_forward,
        "inverse": _lookup_probe_inverse,
        "articulation": _lookup_probe_articulation,
        "ood": _lookup_probe_ood,
        "planning": _lookup_probe_planning,
        "structural": _lookup_probe_structural,
    }
    return probes[application](struct, held_out, rng)


# ============================================================================
# ARITHMETIC RULE TASK
# ============================================================================

ARITHMETIC_DIFFICULTY = {
    1: {"prime": 7, "n_shown": 4},
    2: {"prime": 13, "n_shown": 6},
    3: {"prime": 23, "n_shown": 8},
}


def generate_arithmetic_rule(seed: int = 42, difficulty: int = 1) -> Dict[str, Any]:
    """Generate a modular affine function f(x) = (a*x + b) mod p.

    Guarantees the function is non-trivial (not the identity).
    """
    rng = random.Random(seed)
    p = ARITHMETIC_DIFFICULTY[difficulty]["prime"]
    a = rng.randint(1, p - 1)
    b = rng.randint(0, p - 1)
    if a == 1 and b == 0:
        b = rng.randint(1, p - 1)
    pairs = {x: (a * x + b) % p for x in range(p)}
    return {
        "task_type": "arithmetic_rule",
        "difficulty": difficulty,
        "prime": p,
        "a": a,
        "b": b,
        "pairs": pairs,
    }


# --- Formatters -----------------------------------------------------------


def _arith_format_few_shot(
    struct: Dict[str, Any], rng: random.Random
) -> Tuple[str, List[Any]]:
    pairs = struct["pairs"]
    n_shown = ARITHMETIC_DIFFICULTY[struct["difficulty"]]["n_shown"]
    all_items = list(pairs.items())
    rng.shuffle(all_items)
    shown = all_items[:n_shown]
    held_out = all_items[n_shown:]
    lines = [f"f({x}) = {y}" for x, y in shown]
    return "\n".join(lines) + "\n\n", held_out


def _arith_format_natural(
    struct: Dict[str, Any], rng: random.Random
) -> Tuple[str, List[Any]]:
    pairs = struct["pairs"]
    p = struct["prime"]
    items = list(pairs.items())
    rng.shuffle(items)
    n_examples = 5 + struct["difficulty"]
    examples = ", ".join(f"f({x}) = {y}" for x, y in items[:n_examples])
    ctx = (
        f"A mathematical function f transforms integers in the range "
        f"{{0, 1, ..., {p - 1}}} to integers in the same range. "
        f"Here are some known values: {examples}.\n\n"
    )
    return ctx, list(pairs.items())


def _arith_format_step_by_step(
    struct: Dict[str, Any], rng: random.Random
) -> Tuple[str, List[Any]]:
    """Show the actual modular arithmetic computation steps.

    This reveals the rule parameters (a, b, p), which is the purpose of the
    step-by-step format — it provides full reasoning traces.
    """
    a, b, p = struct["a"], struct["b"], struct["prime"]
    pairs = struct["pairs"]
    items = list(pairs.items())
    rng.shuffle(items)
    n_traces = 3 + struct["difficulty"]
    traces = []
    for x, y in items[:n_traces]:
        ax = a * x
        ax_plus_b = ax + b
        traces.append(
            f"Compute f({x}):\n"
            f"  Step 1: Multiply {x} by {a}: {a} * {x} = {ax}.\n"
            f"  Step 2: Add {b}: {ax} + {b} = {ax_plus_b}.\n"
            f"  Step 3: Take modulo {p}: {ax_plus_b} mod {p} = {y}.\n"
            f"  Result: f({x}) = {y}"
        )
    return "\n\n".join(traces) + "\n\n", list(pairs.items())


def _arith_format_structured(
    struct: Dict[str, Any], rng: random.Random
) -> Tuple[str, List[Any]]:
    pairs = struct["pairs"]
    p = struct["prime"]
    table = {str(x): y for x, y in pairs.items()}
    ctx = (
        f"Function f on the domain {{0, 1, ..., {p - 1}}}:\n"
        f"```json\n{json.dumps(table, indent=2)}\n```\n\n"
    )
    return ctx, []


def format_arithmetic(
    struct: Dict[str, Any], fmt: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    """Format an arithmetic rule instance according to the specified format."""
    formatters = {
        "demonstration": _arith_format_few_shot,
        "natural_language": _arith_format_natural,
        "trace": _arith_format_step_by_step,
        "formal_spec": _arith_format_structured,
    }
    return formatters[fmt](struct, rng)


# --- Probes ---------------------------------------------------------------


def _arith_probe_forward(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    pairs = struct["pairs"]
    if held_out:
        x, y = held_out[0]
    else:
        x = rng.randint(0, struct["prime"] - 1)
        y = pairs[x]
    return f"Question: What is f({x})? Give only the exact integer.\nAnswer: ", str(y)


def _arith_probe_inverse(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    a, b, p = struct["a"], struct["b"], struct["prime"]
    y = rng.randint(0, p - 1)
    x = (y - b) * mod_inverse(a, p) % p
    return (
        f"Question: Find x such that f(x) = {y}. Give only the exact integer.\nAnswer: ",
        str(x),
    )


def _arith_probe_articulation(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    a, b, p = struct["a"], struct["b"], struct["prime"]
    return (
        "Question: Describe the rule f in the form f(x) = ... (mod ...). One line only.\nAnswer: ",
        f"f(x) = {a}x + {b} (mod {p})",
    )


def _arith_probe_ood(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    """Compute f(f(x)) — tests whether the model can compose the function."""
    a, b, p = struct["a"], struct["b"], struct["prime"]
    x = rng.randint(0, p - 1)
    y1 = (a * x + b) % p
    y2 = (a * y1 + b) % p
    return f"Question: What is f(f({x}))? Give only the exact integer.\nAnswer: ", str(
        y2
    )


def _arith_probe_planning(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    a, b, p = struct["a"], struct["b"], struct["prime"]
    k = 2 + struct["difficulty"]
    target = rng.randint(0, p - 1)
    a_inv = mod_inverse(a, p)
    x = target
    for _ in range(k):
        x = (x - b) * a_inv % p
    q = (
        f"Question: Find x such that applying f exactly {k} times "
        f"starting from x gives {target}. Give only the exact integer.\nAnswer: "
    )
    return q, str(x)


def _arith_probe_structural(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    a, b, p = struct["a"], struct["b"], struct["prime"]
    if a == 1:
        n_fixed = p if b == 0 else 0
    else:
        n_fixed = 1
    return (
        "Question: How many values of x satisfy f(x) = x? Give only the exact integer.\nAnswer: ",
        str(n_fixed),
    )


def probe_arithmetic(
    struct: Dict[str, Any],
    held_out: List[Any],
    application: str,
    rng: random.Random,
) -> Tuple[str, str]:
    """Generate a probe question for the arithmetic rule task."""
    probes = {
        "forward": _arith_probe_forward,
        "inverse": _arith_probe_inverse,
        "articulation": _arith_probe_articulation,
        "ood": _arith_probe_ood,
        "planning": _arith_probe_planning,
        "structural": _arith_probe_structural,
    }
    return probes[application](struct, held_out, rng)


# ============================================================================
# CONDITIONAL RULE TASK
# ============================================================================

CONDITIONAL_DIFFICULTY = {
    1: {"prime": 7, "n_shown": 5},
    2: {"prime": 13, "n_shown": 7},
    3: {"prime": 23, "n_shown": 9},
}


def generate_conditional_rule(seed: int = 42, difficulty: int = 1) -> Dict[str, Any]:
    """Generate a piecewise modular affine function with a threshold split.

    f(x) = (a1*x + b1) mod p   if x < threshold
    f(x) = (a2*x + b2) mod p   if x >= threshold

    The two branches use different (a, b) pairs so the model must detect
    both the split point and the two rules.
    """
    rng = random.Random(seed)
    p = CONDITIONAL_DIFFICULTY[difficulty]["prime"]
    threshold = p // 2

    # Generate two distinct affine rules
    a1 = rng.randint(1, p - 1)
    b1 = rng.randint(0, p - 1)
    a2 = rng.randint(1, p - 1)
    b2 = rng.randint(0, p - 1)
    # Ensure the two rules are actually different
    while a1 == a2 and b1 == b2:
        a2 = rng.randint(1, p - 1)
        b2 = rng.randint(0, p - 1)

    pairs = {}
    for x in range(p):
        if x < threshold:
            pairs[x] = (a1 * x + b1) % p
        else:
            pairs[x] = (a2 * x + b2) % p

    return {
        "task_type": "conditional_rule",
        "difficulty": difficulty,
        "prime": p,
        "threshold": threshold,
        "a1": a1,
        "b1": b1,
        "a2": a2,
        "b2": b2,
        "pairs": pairs,
    }


# --- Formatters -----------------------------------------------------------


def _cond_format_few_shot(
    struct: Dict[str, Any], rng: random.Random
) -> Tuple[str, List[Any]]:
    pairs = struct["pairs"]
    n_shown = CONDITIONAL_DIFFICULTY[struct["difficulty"]]["n_shown"]
    all_items = list(pairs.items())
    rng.shuffle(all_items)
    shown = all_items[:n_shown]
    held_out = all_items[n_shown:]
    lines = [f"f({x}) = {y}" for x, y in shown]
    return "\n".join(lines) + "\n\n", held_out


def _cond_format_natural(
    struct: Dict[str, Any], rng: random.Random
) -> Tuple[str, List[Any]]:
    pairs = struct["pairs"]
    p = struct["prime"]
    items = list(pairs.items())
    rng.shuffle(items)
    n_examples = 5 + struct["difficulty"]
    examples = ", ".join(f"f({x}) = {y}" for x, y in items[:n_examples])
    ctx = (
        f"A function f maps integers in {{0, 1, ..., {p - 1}}} to integers "
        f"in the same range. The function may behave differently for "
        f"different parts of its input range. "
        f"Here are some known values: {examples}.\n\n"
    )
    return ctx, list(pairs.items())


def _cond_format_step_by_step(
    struct: Dict[str, Any], rng: random.Random
) -> Tuple[str, List[Any]]:
    """Show computation steps including the threshold check."""
    a1, b1 = struct["a1"], struct["b1"]
    a2, b2 = struct["a2"], struct["b2"]
    p, threshold = struct["prime"], struct["threshold"]
    pairs = struct["pairs"]
    items = list(pairs.items())
    rng.shuffle(items)
    n_traces = 3 + struct["difficulty"]
    traces = []
    for x, y in items[:n_traces]:
        if x < threshold:
            a, b, branch = a1, b1, "low"
        else:
            a, b, branch = a2, b2, "high"
        ax = a * x
        ax_plus_b = ax + b
        traces.append(
            f"Compute f({x}):\n"
            f"  Step 1: Check region: {x} {'<' if x < threshold else '>='} "
            f"{threshold} → {branch} branch.\n"
            f"  Step 2: Multiply {x} by {a}: {a} * {x} = {ax}.\n"
            f"  Step 3: Add {b}: {ax} + {b} = {ax_plus_b}.\n"
            f"  Step 4: Take modulo {p}: {ax_plus_b} mod {p} = {y}.\n"
            f"  Result: f({x}) = {y}"
        )
    return "\n\n".join(traces) + "\n\n", list(pairs.items())


def _cond_format_structured(
    struct: Dict[str, Any], rng: random.Random
) -> Tuple[str, List[Any]]:
    pairs = struct["pairs"]
    p = struct["prime"]
    threshold = struct["threshold"]
    low = {str(x): y for x, y in pairs.items() if x < threshold}
    high = {str(x): y for x, y in pairs.items() if x >= threshold}
    obj = {
        "low_branch (x < " + str(threshold) + ")": low,
        "high_branch (x >= " + str(threshold) + ")": high,
    }
    ctx = (
        f"Piecewise function f on {{0, 1, ..., {p - 1}}} "
        f"with threshold {threshold}:\n"
        f"```json\n{json.dumps(obj, indent=2)}\n```\n\n"
    )
    return ctx, []


def format_conditional(
    struct: Dict[str, Any], fmt: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    """Format a conditional rule instance according to the specified format."""
    formatters = {
        "demonstration": _cond_format_few_shot,
        "natural_language": _cond_format_natural,
        "trace": _cond_format_step_by_step,
        "formal_spec": _cond_format_structured,
    }
    return formatters[fmt](struct, rng)


# --- Probes ---------------------------------------------------------------


def _cond_probe_forward(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    pairs = struct["pairs"]
    if held_out:
        x, y = held_out[0]
    else:
        x = rng.randint(0, struct["prime"] - 1)
        y = pairs[x]
    return f"Question: What is f({x})? Give only the exact integer.\nAnswer: ", str(y)


def _cond_probe_inverse(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    """Find x such that f(x) = y. Pick a y that has a unique preimage."""
    pairs = struct["pairs"]
    # Build reverse map; pick a y with exactly one preimage for clean answers
    rev: Dict[int, List[int]] = {}
    for x, y in pairs.items():
        rev.setdefault(y, []).append(x)
    unique = [(y, xs[0]) for y, xs in rev.items() if len(xs) == 1]
    if unique:
        y, x = rng.choice(unique)
    else:
        # fallback: pick any
        x = rng.choice(list(pairs.keys()))
        y = pairs[x]
    return (
        f"Question: Find x such that f(x) = {y}. Give only the exact integer.\nAnswer: ",
        str(x),
    )


def _cond_probe_articulation(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    a1, b1 = struct["a1"], struct["b1"]
    a2, b2 = struct["a2"], struct["b2"]
    p, threshold = struct["prime"], struct["threshold"]
    answer = (
        f"f(x) = {a1}x + {b1} (mod {p}) if x < {threshold}, "
        f"f(x) = {a2}x + {b2} (mod {p}) if x >= {threshold}"
    )
    return (
        "Question: Describe the rule f. Note that it may use different "
        "formulas for different input ranges. One line only.\nAnswer: ",
        answer,
    )


def _cond_probe_ood(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    """Compute f(f(x)) — composition across branches."""
    pairs = struct["pairs"]
    x = rng.randint(0, struct["prime"] - 1)
    y1 = pairs[x]
    y2 = pairs[y1]
    return f"Question: What is f(f({x}))? Give only the exact integer.\nAnswer: ", str(
        y2
    )


def _cond_probe_planning(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    """Find x such that f^k(x) = target by backward iteration."""
    pairs = struct["pairs"]
    p = struct["prime"]
    k = 2 + struct["difficulty"]
    target = rng.randint(0, p - 1)
    # Build reverse map for backward search
    rev: Dict[int, List[int]] = {}
    for x, y in pairs.items():
        rev.setdefault(y, []).append(x)
    # Walk backward k steps
    current = target
    for _ in range(k):
        preimages = rev.get(current, [])
        if not preimages:
            # No preimage — pick a new target and restart
            target = rng.randint(0, p - 1)
            current = target
            break
        current = rng.choice(preimages)
    # Verify forward
    check = current
    for _ in range(k):
        check = pairs[check]
    if check != target:
        # Fallback: just use a known forward chain
        start = rng.randint(0, p - 1)
        current = start
        for _ in range(k):
            current = pairs[current]
        target = current
        current = start
    q = (
        f"Question: Find x such that applying f exactly {k} times "
        f"starting from x gives {target}. Give only the exact integer.\nAnswer: "
    )
    return q, str(current)


def _cond_probe_structural(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    """Count the threshold value — tests whether the model detected the split."""
    threshold = struct["threshold"]
    return (
        "Question: At what input value does the function's behavior change "
        "(i.e., what is the threshold)? Give only the exact integer.\nAnswer: ",
        str(threshold),
    )


def probe_conditional(
    struct: Dict[str, Any],
    held_out: List[Any],
    application: str,
    rng: random.Random,
) -> Tuple[str, str]:
    """Generate a probe question for the conditional rule task."""
    probes = {
        "forward": _cond_probe_forward,
        "inverse": _cond_probe_inverse,
        "articulation": _cond_probe_articulation,
        "ood": _cond_probe_ood,
        "planning": _cond_probe_planning,
        "structural": _cond_probe_structural,
    }
    return probes[application](struct, held_out, rng)


__all__ = [
    "LOOKUP_DIFFICULTY",
    "ARITHMETIC_DIFFICULTY",
    "CONDITIONAL_DIFFICULTY",
    "generate_lookup_table",
    "format_lookup",
    "probe_lookup",
    "generate_arithmetic_rule",
    "format_arithmetic",
    "probe_arithmetic",
    "generate_conditional_rule",
    "format_conditional",
    "probe_conditional",
]
