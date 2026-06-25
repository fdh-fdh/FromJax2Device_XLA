# Roadmap

## Phase 1: Preserve the Original Work

- Export the Colab notebook as `.ipynb`.
- Save it in `notebooks/` with a descriptive name, for example
  `notebooks/from_jax_to_device_xla_original.ipynb`.
- Add the Colab URL near the top of the notebook or in `docs/colab-to-repo.md`.
- Remove any private tokens, local paths, or large generated outputs before
  committing.

## Phase 2: Clean the Notebook

- Create a cleaned notebook next to the original.
- Split long cells into focused sections:
  - JAX function
  - tracing
  - Jaxpr
  - lowering
  - HLO / StableHLO
  - compilation
  - device execution
- Add short markdown explanations before each code block.

## Phase 3: Extract Reusable Code

- Move repeated helpers into `src/fromjax2device_xla/`.
- Keep notebook cells as thin calls into those helpers.
- Add small tests only after helpers become stable.

## Phase 4: Build Runnable Examples

- Add one example per concept under `examples/`.
- Prefer short scripts that print one clear artifact:
  - a Jaxpr
  - lowered IR
  - device information
  - timing comparison
- Keep examples independent so readers can run them one by one.

## Phase 5: Polish for GitHub

- Add screenshots or sample outputs to the README.
- Add a table of contents for notes under `docs/`.
- Add tested setup instructions for CPU first, GPU later.
- Make the first commit only after the notebook and scaffold are both readable.
