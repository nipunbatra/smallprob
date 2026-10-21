# Smallprob working notes

- Preserve the simple Python model API: @model, direct distribution declarations,
  optional explicit draw, ordinary lists and loops.
- Keep the graph/frontend, user API and inference engines separate.
- NumPy and array-api-compat are the base runtime dependencies. Optional extras
  supply PyTorch execution (torch), Pyro/PyTorch inference (vi), Matplotlib
  diagrams (viz), and executable notebooks (notebook). Keep these optional;
  discuss further major runtime dependencies before expanding implementation.
- Support NumPy and Torch explicitly; do not imply support for all Array API
  backends. Preserve device/dtype, avoid implicit GPU-to-NumPy conversion,
  and reject unavailable execution choices instead of silently falling back.
- Fixed tensors are detached host snapshots for tracing/validation. Do not
  advertise gradients into external inputs. HMM exact inference is NumPy/CPU.
- Preserve distribution parameter meaning: Normal uses SD in code and variance
  in math notation; StudentT uses location and scale, with mean/variance only
  where its degrees of freedom permit them. Glyphs illustrate families.
- Never silently approximate inference or reinterpret an unsupported model.
- Read DESIGN.md for the internal contract and docs/inference.md for scope.
- Use .venv/bin/python -m pytest -q and .venv/bin/python -m examples.demo after
  relevant changes. Exact inference needs independent mathematical references.
- Run .venv/bin/python -m examples.run_notebook after notebook-facing changes;
  it executes the notebook with this interpreter rather than a global kernel.
  Inspect rendered diagrams and plots, not just cell execution status.
- Label synthetic demonstration results, approximate methods and unsupported
  features accurately. Keep generated files in outputs/ and scratch in work/.
