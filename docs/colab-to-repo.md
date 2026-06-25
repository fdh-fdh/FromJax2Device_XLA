# Colab to GitHub Migration Checklist

Source Colab:

https://colab.research.google.com/drive/1EXxXmV6zLqZIKh3vPMZ2KaWc4AGzvusW

## 1. Export

In Colab:

- Open `File`.
- Choose `Download`.
- Download `.ipynb`.
- Save it under `notebooks/`.

Suggested filename:

```text
notebooks/from_jax_to_device_xla_original.ipynb
```

## 2. Clean Before Commit

Check the notebook for:

- personal Google Drive paths
- API keys or tokens
- very large outputs
- temporary debug cells
- cells that depend on hidden Colab state

If outputs are large, clear notebook outputs before committing.

## 3. Split the Learning Story

Suggested notebook sections:

1. Why JAX needs tracing
2. Python function to Jaxpr
3. Jaxpr to lowered compiler IR
4. XLA / StableHLO inspection
5. Compilation and device placement
6. GPU execution notes
7. Performance observations

## 4. Extract Code Gradually

Move only stable repeated code into `src/fromjax2device_xla/`.

Good first candidates:

- device inspection helpers
- Jaxpr printing helpers
- lowering / compiler IR display helpers
- timing helpers

Avoid extracting exploratory one-off cells too early.

## 5. Create GitHub-Friendly Examples

Each example should be runnable from the repo root:

```bash
python examples/inspect_jaxpr.py
```

Keep each example focused on one idea so the repo stays easy to read.
