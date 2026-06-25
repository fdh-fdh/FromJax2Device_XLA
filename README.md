# From JAX to Device: XLA Learning Notes

This repository organizes a exploration of how JAX programs move
from Python functions to device execution through tracing, Jaxpr, XLA/HLO, and
compiled kernels.

## Project Goals

- Turn the original Colab notebook into a reproducible GitHub project.
- Keep the learning path visible: small examples first, deeper compiler/device
  notes later.
- Separate executable examples, notebooks, and written explanations.
- Build a foundation for future experiments on JAX, XLA, GPU execution, and
  performance inspection.

## Repository Layout

```text
.
|-- docs/                    # Roadmaps, migration notes, and learning notes
|-- examples/                # Small runnable examples
|-- notebooks/               # Original and cleaned Colab notebooks
|-- src/fromjax2device_xla/  # Future reusable Python utilities
`-- tests/                   # Future checks for utility code
```

## Recommended Workflow

1. Export the Colab notebook as `.ipynb`.
2. Save the raw notebook in `notebooks/`.
3. Create a cleaned version with stable outputs removed.
4. Move reusable code from notebook cells into `src/fromjax2device_xla/`.
5. Turn important notebook sections into short notes under `docs/`.
6. Add small examples under `examples/` that can run without opening Colab.

See [docs/colab-to-repo.md](docs/colab-to-repo.md) for the migration checklist.

## Environment

The exact dependency set should be finalized after the notebook is imported.
For now, start with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

JAX installation can vary by CPU/GPU/CUDA platform, so keep platform-specific
installation notes in `docs/setup.md` once the target machine is clear.

## Status

Initial repository scaffold. The Colab notebook still needs to be exported into
`notebooks/` before examples and documentation can be made precise.
