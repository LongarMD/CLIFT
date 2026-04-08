# CLIFT: A Benchmark for Contextual Learning across Inference, Format, and Transfer

[![Hugging Face — CLIFT](https://img.shields.io/badge/🤗%20Hugging%20Face-CLIFT-ffd21e?style=for-the-badge)](https://huggingface.co/datasets/longarmd/CLIFT)

![CLIFT Benchmark Overview](docs/clift_benchmark_overview.png)


## Install

Install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync
```

Core generation works without CLRS. Tasks **`insertion_sort`** and **`binary_search`** use the optional `clrs` sampler; install it with:

```bash
uv sync --extra clrs
```

Development tools (pytest, ruff):

```bash
uv sync --group dev --extra clrs
```

## Dataset

The prebuilt evaluation matrix is in `data/`:

- **`data/clift.jsonl`** — one JSON object per line. Each row is a full instance.
- **`data/manifest.json`** — generation parameters, expected line count, and a SHA-256 of the canonical JSONL payload.

This snapshot uses **10 instances per (task, format, application, difficulty) cell**, master **`seed` 42**, and **all** tasks in `clift.common.TASKS`. It contains **5160** records.

### Regenerating your own dataset

1. **Install with the CLRS extra** so the full task list can be built:

   ```bash
   uv sync --extra clrs
   ```

2. **Generate and export**:

   ```python
   from clift.common import export_jsonl
   from clift.data import generate_clift_dataset

   instances = generate_clift_dataset(n_instances_per_cell=10, seed=42)
   export_jsonl(instances, "data/clift.jsonl")
   ```