# Verification, 7 October 2026

Tested with Python 3.12.10 on Windows; direct dependency versions are pinned in
`requirements.txt`. Editable package installation and `pip check` passed.

- `python -m pytest -q`: 3 tests passed.
- `python examples/fictional_room.py`: completed on a 30x40 grid, 10 iterations;
  all fields finite, solid-cell velocities zero, nonzero source response.
- Streamlit `AppTest` initial-page execution after the empty-canvas gating fix:
  no app exceptions; expected prototype title rendered.

The first AppTest timed out because the source launched a 320-iteration all-zero
simulation on an empty DataFrame. The UI condition was corrected to require actual
objects or a wall-image mask; solver equations remain unchanged. Browser-side
freehand interactions and image-backed canvas rendering were not exhaustively
tested. No CFD, mass balance, grid convergence or experimental calibration claim
follows from finite fields or an initial-page check.

`CITATION.cff` passed validation against the official CFF 1.2 schema.
