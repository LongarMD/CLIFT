import ast
import difflib
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .common import APPLICATIONS, FORMATS

# ====================================================================
# TEXT NORMALISATION & SCORING
# ====================================================================


def normalize_text(text: str) -> str:
    """Normalise text for exact-match comparison."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s,\[\]\-]", "", text)  # Added '-' for negative numbers
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(prediction: str, target: str) -> bool:
    return normalize_text(prediction) == normalize_text(target)


def first_line_match(prediction: str, target: str) -> bool:
    first = prediction.strip().split("\n")[0].strip()
    return normalize_text(first) == normalize_text(target)


def _extract_first_segment(prediction: str) -> str:
    """Extract the model's answer segment, trimming apparent prompt restarts."""
    text = prediction.strip()

    # Split on cues that often indicate the model started a new prompt mid-output
    cue_parts = re.split(
        r"\n\s*(?:Question:|Input:|Q:|Sort by weight:|Target array:|Rule:|Formal Specification:)",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    return cue_parts[0].strip()


def extract_list(text: str) -> Optional[List[int]]:
    """Safely extracts the LAST list of integers from the model's text generation."""
    segment = _extract_first_segment(text)

    # Find ALL well-formed lists in the text
    matches = re.findall(r"\[[-0-9,\s]*\]", segment)
    if matches:
        try:
            # Always take the LAST list, as CoT models put the final answer at the end
            return ast.literal_eval(matches[-1])
        except (ValueError, SyntaxError):
            pass

    # Fallback for truncated lists (e.g. hitting max_tokens before closing ']')
    # We find the LAST '[' and extract all integers that follow it.
    last_bracket_idx = segment.rfind("[")
    if last_bracket_idx != -1:
        content = segment[last_bracket_idx + 1 :]
        nums = re.findall(r"-?\d+", content)
        if nums:
            return [int(n) for n in nums]
        return []

    return None


def contains_match(prediction: str, target: str) -> bool:
    segment = _extract_first_segment(prediction)
    norm_target = normalize_text(target)

    # If target is a single integer, check the LAST few numbers in the CoT
    if re.match(r"^-?\d+$", norm_target):
        pred_nums = re.findall(r"-?\d+", segment)
        if not pred_nums:
            return False
        # Check if the target is among the last 3 numbers generated
        # (This prevents false positives from intermediate math in the CoT)
        return norm_target in pred_nums[-3:]

    norm_pred = normalize_text(segment)
    return norm_target in norm_pred


def extract_single_token_answer(prediction: str) -> str:
    """Extract a single-token answer from model output for strict token tasks."""
    segment = _extract_first_segment(prediction).strip()
    if not segment:
        return ""

    # Drop common answer prefixes.
    segment = re.sub(r"^\s*(?:answer|output)\s*:\s*", "", segment, flags=re.IGNORECASE)

    # Try token-like chunks and skip role/control words.
    role_words = {
        "assistant",
        "user",
        "system",
        "answer",
        "output",
        "context",
        "question",
    }
    candidates = re.findall(r"[A-Za-z0-9_-]+", segment)
    for cand in candidates:
        if cand.lower() not in role_words:
            return cand
    return ""


def single_token_compliance(prediction: str) -> bool:
    """Whether the first segment appears to be exactly one token answer."""
    segment = _extract_first_segment(prediction).strip()
    if not segment:
        return False

    # Allow optional surrounding single/double quotes and trailing punctuation.
    cleaned = segment.strip().strip("`")
    cleaned = re.sub(r"^[\"']+|[\"']+$", "", cleaned)
    cleaned = re.sub(r"[.,;:!?]+$", "", cleaned).strip()
    return re.fullmatch(r"[A-Za-z0-9_-]+", cleaned) is not None


def _spatial_distance_diagnostics(
    inst: Dict[str, Any],
    predicted_token: str,
    target_token: str,
) -> Tuple[Optional[bool], Optional[int]]:
    """Return (predicted_token_in_vocab, manhattan_distance_if_valid)."""
    if inst.get("task") != "spatial_translation":
        return None, None

    latent = inst.get("latent_structure", {}) or {}
    token_map = latent.get("token_map", {}) or {}
    if not token_map:
        return None, None

    if predicted_token not in token_map:
        return False, None

    if target_token not in token_map:
        return True, None

    pred_coord = token_map[predicted_token]
    tgt_coord = token_map[target_token]
    if len(pred_coord) != len(tgt_coord):
        return True, None

    dist = int(sum(abs(int(a) - int(b)) for a, b in zip(pred_coord, tgt_coord)))
    return True, dist


def compute_soft_score(prediction: str, target: str) -> float:
    """
    Computes a continuous score (0.0 to 1.0) based on algorithmic distance.
    """
    target_list = extract_list(target)

    # Fallback: If target is a scalar integer (e.g. Binary Search index or Inverse cause)
    if target_list is None:
        return 1.0 if contains_match(prediction, target) else 0.0

    pred_list = extract_list(prediction)
    if pred_list is None:
        return 0.0

    # Heuristic for Bounds/Pointers (Max Subarray, Binary Search traces)
    if len(target_list) == 2 and len(pred_list) == 2:
        p_start, p_end = pred_list
        t_start, t_end = target_list

        # Guard against backwards bounds generated by the model
        if p_start > p_end:
            p_start, p_end = p_end, p_start

        intersection = max(0, min(p_end, t_end) - max(p_start, t_start) + 1)
        union = max(p_end, t_end) - min(p_start, t_start) + 1

        if union == 0:
            return 0.0
        return float(intersection) / float(union)

    # Sequence Similarity for permutations/arrays (Edit Distance)
    sm = difflib.SequenceMatcher(None, pred_list, target_list)
    return sm.ratio()


# ====================================================================
# DATASET EVALUATION
# ====================================================================


def score_dataset(
    instances: List[Dict[str, Any]],
    predictions: List[str],
) -> pd.DataFrame:
    if len(instances) != len(predictions):
        raise ValueError(
            f"instances and predictions must have the same length; "
            f"got {len(instances)} and {len(predictions)}"
        )
    records = []
    for inst, pred in zip(instances, predictions, strict=True):
        extracted = extract_single_token_answer(pred)
        target_norm = normalize_text(inst["target"])
        extracted_norm = normalize_text(extracted)
        in_vocab, spatial_dist = _spatial_distance_diagnostics(
            inst,
            predicted_token=extracted_norm,
            target_token=target_norm,
        )
        correct_extracted = extracted_norm == target_norm if extracted_norm else False
        records.append(
            {
                "instance_id": inst.get("instance_id", 0),
                "task": inst["task"],
                "format": inst["format"],
                "application": inst["application"],
                "difficulty": inst.get("difficulty", 1),
                "prompt": inst.get("prompt", ""),
                "target": inst["target"],
                "prediction": pred,
                "prediction_extracted": extracted,
                "single_token_compliance": single_token_compliance(pred),
                "latent_structure": inst.get("latent_structure", {}),
                "metadata": inst.get("metadata", {}),
                "ood_subtype": (inst.get("metadata", {}) or {}).get("ood_subtype"),
                "correct": exact_match(pred, inst["target"]),
                "correct_extracted": correct_extracted,
                "correct_fl": first_line_match(pred, inst["target"]),
                "correct_contains": contains_match(pred, inst["target"]),
                "soft_score": compute_soft_score(pred, inst["target"]),
                "predicted_token_in_vocab": in_vocab,
                "spatial_manhattan_distance": spatial_dist,
                "spatial_near_miss": (spatial_dist == 1)
                if spatial_dist is not None and not correct_extracted
                else False,
                "spatial_far_miss": (spatial_dist > 1)
                if spatial_dist is not None and not correct_extracted
                else False,
            }
        )
    df = pd.DataFrame(records)

    # Exclude masked logically invalid prompts
    df = df[df["target"] != "[MASKED]"]

    return df


# ====================================================================
# DIAGNOSTIC MARGINALISATION
# ====================================================================


def _drop_masked_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude masked rows from analytics/scoring aggregations."""
    if "target" not in df.columns:
        return df
    return df[df["target"] != "[MASKED]"]


def _format_application_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Build a full format x application matrix, preserving empty cells as NaN."""
    pivot = df.pivot_table(
        values=metric, index="format", columns="application", aggfunc="mean"
    )
    # Support both legacy CLIFT axes and CLRS-specific axes.
    # If current data uses a different axis set than common.py, fall back to observed values.
    preferred_formats = list(FORMATS) + [
        "demonstration",
        "declarative_natural_language",
        "trace",
        "formal_specification",
    ]
    preferred_apps = list(APPLICATIONS) + ["forward", "inverse", "ood"]

    # De-duplicate while preserving order.
    preferred_formats = list(dict.fromkeys(preferred_formats))
    preferred_apps = list(dict.fromkeys(preferred_apps))

    ordered_formats = [f for f in preferred_formats if f in pivot.index]
    ordered_apps = [a for a in preferred_apps if a in pivot.columns]

    # Ensure we don't accidentally drop unseen-but-valid labels.
    remaining_formats = sorted([f for f in pivot.index if f not in ordered_formats])
    remaining_apps = sorted([a for a in pivot.columns if a not in ordered_apps])
    ordered_formats.extend(remaining_formats)
    ordered_apps.extend(remaining_apps)

    return pivot.reindex(index=ordered_formats, columns=ordered_apps)


def compute_marginals(
    df: pd.DataFrame,
    metric: str = "soft_score",  # Shifted default metric to the new soft score
) -> Dict[str, pd.DataFrame]:
    """Compute diagnostic marginalisations from scored results.

    Args:
        df: DataFrame produced by score_dataset.
        metric: Column name to aggregate.

    Returns:
        Dictionary with marginalisation tables.
    """
    df = _drop_masked_rows(df)
    marginals: Dict[str, pd.DataFrame] = {}

    ta = df.groupby("task")[metric].mean().reset_index()
    ta.columns = ["task", "accuracy"]
    marginals["task_acquisition"] = ta

    fr = df.groupby(["task", "format"])[metric].mean().reset_index()
    fr.columns = ["task", "format", "accuracy"]
    marginals["format_robustness"] = fr

    af = df.groupby(["task", "application"])[metric].mean().reset_index()
    af.columns = ["task", "application", "accuracy"]
    marginals["application_flexibility"] = af

    fm = df.groupby(["task", "format", "application"])[metric].mean().reset_index()
    fm.columns = ["task", "format", "application", "accuracy"]
    marginals["full_matrix"] = fm

    pivot = _format_application_pivot(df, metric)
    marginals["format_x_application"] = pivot

    if "difficulty" in df.columns:
        ds = df.groupby(["task", "difficulty"])[metric].mean().reset_index()
        ds.columns = ["task", "difficulty", "accuracy"]
        marginals["difficulty_scaling"] = ds

        da = df.pivot_table(
            values=metric,
            index="difficulty",
            columns="application",
            aggfunc="mean",
        )
        marginals["difficulty_x_application"] = da

    return marginals


__all__ = [
    "compute_marginals",
    "contains_match",
    "exact_match",
    "first_line_match",
    "normalize_text",
    "score_dataset",
]
