# Active solver and scope

Selected sources are app.py, process_layout.py and simulation.py from the existing
application. The active `backend='ffd'` branch actually calls the built-in
Navier–Stokes-inspired/Brinkman solver; it does not import the separate ffd_backend
module. That unused module is excluded. The incomplete OpenFOAM stub, ROM loader,
fallback backend paths and unnecessary model binaries are excluded. The built-in
branch and numerical equations remain unchanged.

The source uses unit-grid finite differences with rolled neighbours, a masked
upwind stencil, iterative pressure projection, explicit pseudo-time updates,
Brinkman damping in solids, simplified buoyancy and cell-dependent thermal
properties. Interface conductivity uses harmonic means. Inlet/return objects
select outer-boundary segments; absent return points can trigger an automatic
opposite-side opening. Thermal source dictionaries use the key `load`; Gaussian
amplitudes are not normalised to a total physically calibrated wattage.

The source applies `u_star[inlet_mask[:,0]]` / right-column masks as row selectors.
This selects whole rows, rather than only corresponding boundary cells. The
portfolio retains this behaviour and flags it for investigation: finite outputs
and no solid-cell velocity do not prove correct inlet enforcement or mass balance.
No numerical correction has been silently substituted.

Accepted but inactive parameters include `diff_alpha`, `diff_steps`, `fan_sigma`
and inlet-temperature compatibility settings. The optional affine `calib` argument
is a user transformation, not evidence of experimental calibration. Unknown
kwargs in the legacy API are still accepted; callers should avoid assuming they
change the physics. Only ffd/builtin backends are allowed.

The app adds an explicit prototype limitation notice and package imports, while
retaining drawing, masks, source objects and exports. Empty canvases no longer
trigger a costly all-zero simulation: the UI checks for actual objects or an image
mask. This changes execution gating only, not the numerical solver. Wall-coloured
rectangles also enter the original source-object parser; callers should treat that
overlap as a UI classification limitation rather than a calibrated wall heat load.
The examples are created
from fictional code-defined objects and do not load source-project images.
