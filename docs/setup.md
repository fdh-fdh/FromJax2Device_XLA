# Setup Notes

This project should support a CPU-first setup before adding GPU-specific notes.

## CPU Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then check that JAX can see at least one device:

```bash
python -c "import jax; print(jax.devices())"
```

## GPU Setup

GPU setup depends on CUDA, driver, operating system, and JAX release. Add exact
commands here after the target environment is confirmed.

Keep these notes separate from the README so the front page stays short.
