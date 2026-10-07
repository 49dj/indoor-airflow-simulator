# Indoor thermal / airflow simulator

An interactive prototype for exploring indoor thermal/airflow fields under
configurable room layouts, heat sources, and supply/return-air locations.
This portfolio version organises Yizhe Hu's existing application source;
it does not establish sole authorship of every numerical component.

## Capabilities

The Streamlit interface combines manual room objects and image-based wall
rasterisation, positive/negative source amplitudes, nearest-boundary inlet/outlet
locations, a built-in numerical field solver and heatmap/vector visualisation.
Walls form solid masks; the solver uses masked upwind advection, pressure
projection, Brinkman velocity damping, buoyancy and harmonic-interface conduction.
Wall masks can be exported as JSON run-length encoding.

## Install and run

Python 3.12. Commands run from the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The interface retains the original Chinese controls. Use fictional diagrams:
draw a wall in grey, rectangular heat sources, circular cooling sources, green
supply points and purple return points. Points map to the nearest outer boundary;
they are not arbitrary interior diffuser boundary conditions. No real building
floor plans are included.

For a headless, fictional numerical example:

```bash
python examples/fictional_room.py
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The example reports finite fields and wall constraints in ignored
`results/room.json`. Tests exercise manual/image rasterisation, solid-cell velocity,
finite solver execution and explicit rejection of unavailable backends.

## Structure and physical limits

`app.py` contains UI orchestration. `src/airflow/geometry.py` contains the retained
wall rasteriser; `simulation.py` contains the retained active solver and plotting.
`examples/` contains only fictional objects. No binary surrogate model, image data,
cache or floor plan is distributed.

This is a qualitative numerical prototype, not validated CFD or an engineering
HVAC design tool. Pixel/grid coordinates lack a verified physical length-scale
conversion; displayed source power and relative field amplitudes are not calibrated
engineering predictions. No experimental calibration, grid convergence, mass-balance
study or turbulent-flow validation was performed. The source clips temperature for
stability and masks wall temperatures to zero for display. That display operation
does not imply zero physical wall temperature.

The active source implementation is preserved, including the inlet assignment
indexing and unit-grid derivative approximations; their boundary/scale fidelity
requires further study. Several accepted compatibility parameters are unused.
OpenFOAM placeholder and load-only ROM interfaces are excluded; requesting an
unavailable backend raises an error. Suggested points are heatmap heuristics,
not optimised HVAC placement. [Methods and caveats](docs/methods.md),
[rights](docs/rights.md). Cite this repository and exact commit.

See [observed verification results](docs/verification.md) for executed checks and
their limits.
