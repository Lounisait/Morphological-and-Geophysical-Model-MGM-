# Morphological and Geophysical Model (MGM)

MGM is a coupled research model for impact-crater landscape evolution and
geophysical signatures. It combines topographic evolution, sediment production
and transport, lithospheric flexure, gravity anomaly modelling, magnetic anomaly
modelling, drainage-event diagnostics, and crater-diameter tracking.

## Repository Layout

```text
MGM/
|-- main_simulation_harmonized.py       # Main command-line entry point
|-- function_flexure.py                 # Flexural load and deflection routines
|-- Gravi.py, sub_routine.py            # Gravity anomaly routines
|-- mag_harmonized.py                   # Magnetic anomaly workflows
|-- mag_crater_dipole_harmonized.py     # Magnetic kernels
|-- Topo.py                             # Topography and input-grid utilities
|-- data/                               # Bundled topography and Bouguer profiles
|-- examples/                           # Reproducible command-line examples
|-- requirements.txt                    # Python dependencies
`-- .gitignore                          # Output and cache exclusions
```

Large outputs are intentionally excluded from Git. By default, runs write under
`outputs/`.

## Installation

From the `MGM/` folder, create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Conda users:

```bash
conda create -n mgm python=3.10
conda activate mgm
python -m pip install -r requirements.txt
```

## Quick Start

Run a short drainage-only simulation with the default crater:

```bash
python main_simulation_harmonized.py \
  --crater-profile medium \
  --total-time 10100 \
  --dt 100 \
  --dt-flex 1000 \
  --flexure-outside-fill-mode zero \
  --drainage-only \
  --no-save-figures \
  --no-progress
```

The default crater preset is `medium`/`2000`.

## Bundled Crater Profiles

Each crater preset selects a matched topography profile, Bouguer profile, and
model-domain half-width.

| Preset | Aliases | Topography profile | Bouguer profile | Domain half-width |
| --- | --- | --- | --- | --- |
| `small` | `700`, `petit` | `data/surf_profile_700.csv` | `data/bouguer_700.csv` | `60000` m |
| `medium` | `2000`, `middle`, `milieu`, `moyen` | `data/surf_profile_2000.csv` | `data/bouguer_2000.csv` | `80000` m |
| `large` | `5000`, `grand` | `data/surf_profile_5000.csv` | `data/bouguer_5000.csv` | `160000` m |

Run one preset:

```bash
python main_simulation_harmonized.py --crater-profile small --drainage-only
python main_simulation_harmonized.py --crater-profile medium --drainage-only
python main_simulation_harmonized.py --crater-profile large --drainage-only
```

## Common Run Options

Run the bundled short example:

```bash
./examples/run_short_example.sh
```

Run the short example for a specific crater:

```bash
./examples/run_short_example.sh small
./examples/run_short_example.sh medium
./examples/run_short_example.sh large
```

Run short drainage-only checks for all three craters:

```bash
./examples/run_all_crater_profiles.sh
```

Run with gravity post-processing:

```bash
python main_simulation_harmonized.py \
  --crater-profile medium \
  --run-gravity \
  --gravity-method all \
  --gravity-kernel numba
```

Run with magnetic post-processing disabled:

```bash
python main_simulation_harmonized.py --crater-profile medium --drainage-only
```

Use the geomorphic-layer flexural load calculation explicitly:

```bash
python main_simulation_harmonized.py \
  --crater-profile medium \
  --flexure-load-mode geomorphic_layers
```

`geomorphic_layers` is the default mode. It computes the flexural load from the
bedrock and soil thickness changes produced by geomorphic processes before the
regional tectonic uplift increment is applied. The previous total-change
formulation is still available for comparison with
`--flexure-load-mode legacy_total`.

The flexure solver uses the Wickert restoring-density convention: the density
that fills the deflection is controlled by `--rho-f` and defaults to `0`
kg/m^3. For example:

```bash
python main_simulation_harmonized.py \
  --crater-profile medium \
  --rho-f 0
```

### Choosing the flexure solver

Two interchangeable flexure solvers are available through `--flexure-solver`:

- `sas` (aliases `turcotte`, `kelvin`): analytical solver based on the Kelvin
  Green function (Turcotte & Schubert / gFlex "Superposition of Analytical
  Solutions"). Very fast for localized loads, but its cost grows with both the
  number of loaded nodes and the margin (roughly O(N^2) for a fully loaded
  domain).
- `fd` (aliases `wickert`, `gflex`): finite-difference solver that solves the
  plate equation `D nabla^4 w + (rho_m - rho_f) g w = q` directly on the grid
  using sparse matrices. It scales much better for dense loads and large grids,
  and handles finite-plate boundaries natively via `--flexure-boundary`
  (`free_slope` or `clamped_edge`).

The default is `sas` (historical behaviour). For production runs (dense
geomorphic load, fine grid, large domain) `fd` is generally recommended. The
two solvers agree to better than ~0.1 % in the interior when the loaded region
is surrounded by a buffer of about 3 alpha; they diverge near the boundaries
and on domains smaller than a few alpha.

```bash
python main_simulation_harmonized.py \
  --crater-profile medium \
  --flexure-solver fd \
  --flexure-boundary free_slope
```

### Flexure margin in units of alpha

The flexural parameter `alpha = (D / ((rho_m - rho_f) g))^(1/4)` sets the length
scale of the flexural response and depends on `Te`. Because the appropriate
margin depends on `Te`, you can specify it directly as a number of alpha with
`--flexure-margin-alpha`. It is converted to km automatically and applied to
both solvers (as a zero-load buffer for `fd`). A value of 2 to 3 alpha is
recommended; when greater than 0 it overrides `--flexure-margin-km`. The run
prints the resolved alpha and margin at start-up.

```bash
python main_simulation_harmonized.py \
  --crater-profile medium \
  --flexure-solver fd \
  --flexure-margin-alpha 3
```

The extended flexure domain (for the `sas` solver) can be filled with `zero`,
`mean`, or `edge` values:

```bash
python main_simulation_harmonized.py \
  --crater-profile medium \
  --flexure-outside-fill-mode zero
```

Use `--debug-flexure` to write `flexure_debug.csv`, which stores load,
deflection, and cumulative-flexure diagnostics at each flexure solve.

Override the preset files or domain limit manually:

```bash
python main_simulation_harmonized.py \
  --surface-profile data/surf_profile_5000.csv \
  --bouguer-profile data/bouguer_5000.csv \
  --limit 160000
```

Change elastic thickness and uplift rate:

```bash
python main_simulation_harmonized.py \
  --crater-profile medium \
  --Te 40000 \
  --uplift 30
```

## Command-Line Options Reference

All options are passed to `main_simulation_harmonized.py`. Boolean options are
disabled by default unless noted.

### Core Physical Parameters

| Option | Unit / values | Meaning |
| --- | --- | --- |
| `--Te` | meters | Elastic thickness of the lithosphere. This controls flexural rigidity through `D = E Te^3 / (12(1 - nu^2))`. The default model value is `30000` m. |
| `--uplift` | m/Ma | Regional tectonic uplift rate. Internally converted to m/yr and applied to the full grid at every geomorphic time step. |
| `--rho-f` | kg/m^3 | Density of the material filling flexural deflection in the Wickert restoring-density convention. Default is `0`, corresponding to air/water-free restoring load. Must be non-negative and smaller than mantle density `rhom`. |

### Initial Crater And Domain

| Option | Unit / values | Meaning |
| --- | --- | --- |
| `--crater-profile` | `small`, `700`, `petit`, `medium`, `2000`, `middle`, `milieu`, `moyen`, `large`, `5000`, `grand` | Selects one of the bundled crater presets. Each preset sets topography profile, Bouguer profile, and domain half-width. |
| `--surface-profile` | CSV path | Radial topographic profile used to generate the initial crater topography. Overrides the preset topography profile. |
| `--bouguer-profile` | CSV path | Radial Bouguer profile used as the initial gravity background. Overrides the preset Bouguer profile. |
| `--limit` | meters | Half-width of the square model domain. For example, `80000` gives a domain from `-80` to `+80` km. |
| `--xy-space` | meters | Horizontal grid spacing. Smaller values increase resolution and runtime. |
| `--seed` | integer | Random seed for the initial topographic noise when no seed NPZ is supplied. |
| `--initial-seed-path` | NPZ path | Reuses a saved initial/final model state containing topography, bedrock, soil depth, and grid metadata. Useful for reproducible restarts or parameter sweeps. |

### Time Control

| Option | Unit / values | Meaning |
| --- | --- | --- |
| `--total-time` | years | Maximum simulated duration. The run can stop earlier if `--stop-on-crater-breach` detects a drainage event. |
| `--dt` | years | Numerical geomorphic time step for Landlab evolution, uplift, load accumulation, and event checks. Must be positive. |
| `--dt-flex` | years | Interval between flexure solves. Load is accumulated between solves, then reset after flexure is applied. |
| `--topography-snapshot-times-ma` | one or more Ma values | Exact requested snapshot times for topography, soil depth, and cumulative flexure. Values are projected onto the numerical time step. |

### Flexure And Load Coupling

| Option | Unit / values | Meaning |
| --- | --- | --- |
| `--flexure-load-mode` | `geomorphic_layers`, `legacy_total` | Controls how the flexural load `qs` is built. `geomorphic_layers` is the default and uses bedrock plus soil changes from geomorphic processes before tectonic uplift. `legacy_total` uses total bedrock and soil change, including direct uplift effects. |
| `--flexure-solver` | `sas`, `turcotte`, `kelvin`, `fd`, `wickert`, `gflex` | Selects the flexure solver. `sas`/`turcotte`/`kelvin` is the analytical Kelvin Green-function solver (default). `fd`/`wickert`/`gflex` is the finite-difference solver. |
| `--flexure-boundary` | `free_slope`, `clamped_edge` | Boundary condition for the `fd` solver only. `free_slope` is a zero-gradient edge; `clamped_edge` keeps the default second-derivative stencil. No effect for `sas`. |
| `--flexure-margin-km` | km | Adds a margin around the grid for the flexure calculation to reduce boundary effects. Applies to both solvers (zero-load buffer for `fd`). Ignored if `--flexure-margin-alpha` > 0. Default is `0`. |
| `--flexure-margin-alpha` | number of alpha | Flexure margin expressed in flexural parameters alpha (depends on `Te`). When > 0 it overrides `--flexure-margin-km` and is converted to km automatically. Recommended: 2 to 3. |
| `--flexure-outside-fill-mode` | `zero`, `mean`, `edge` | (`sas` solver only) Defines how the extended flexure margin is filled: `zero` uses no load outside the original domain, `mean` uses the mean non-crater domain load, and `edge` pads from the nearest domain edge. |
| `--debug-flexure` | flag | Writes `flexure_debug.csv` with load, deflection, cumulative flexure, bedrock, soil, and topography diagnostics at each flexure solve. |
| `--no-direct-uplift-flexure-load` | flag | Legacy compatibility flag. The current default `geomorphic_layers` formulation already excludes direct tectonic uplift from flexural loading. |

### Gravity Options

| Option | Unit / values | Meaning |
| --- | --- | --- |
| `--run-gravity` | flag | Enables gravity post-processing. This can be combined with `--drainage-only` to skip magnetics while still computing gravity. |
| `--gravity-method` | `legacy`, `reference`, `complete`, `both`, `all` | Selects which gravity products are computed. `all` computes the available legacy, reference, complete, Bouguer, terrain, and comparison products. |
| `--gravity-kernel` | `numba`, `python` | Numerical backend for gravity kernels. `numba` is faster when available; `python` is the fallback/reference implementation. |
| `--gravity-spc` | grid nodes | Radius used for sedimentary density-contrast gravity calculations. Larger values include more surrounding cells and increase runtime. |
| `--gravity-terrain-spc` | grid nodes | Radius used for terrain-correction calculations. Larger values are more complete but slower. |
| `--gravity-terrain-density` | kg/m^3 | Density used for Bouguer slab and terrain-correction terms. |
| `--gravity-density-mode` | `decreasing`, `constant`, `compaction`, `compaction_layered` | Sedimentary density-contrast law. `decreasing` uses the default depth trend, `constant` uses one fixed contrast, and the compaction modes compute density from lithology and porosity parameters. |
| `--gravity-constant-delta-rho` | kg/m^3 | Constant sedimentary density contrast used when `--gravity-density-mode constant` is selected. |
| `--gravity-compaction-lithology` | `sand`, `shaly_sand`, `shale` | Lithology preset for compaction-based density laws. |
| `--gravity-compaction-layer-thickness` | meters | Maximum thickness of sublayers used by `compaction_layered`. Smaller values resolve compaction better and increase runtime. |
| `--gravity-host-density` | kg/m^3 | Host-rock density used in compaction-based density contrasts. |
| `--gravity-grain-density` | kg/m^3 | Sediment grain density used in compaction calculations. |
| `--gravity-fluid-density` | kg/m^3 | Pore-fluid density used in compaction calculations. Must be non-negative. |
| `--no-initial-gravity-background` | flag | Disables addition of the initial Bouguer profile from the input CSV to gravity outputs. |

### Drainage And Boundary Conditions

| Option | Unit / values | Meaning |
| --- | --- | --- |
| `--boundary-mode` | `all_open`, `south_open`, `southwest_corner`, `southeast_corner` | Sets model-grid boundary conditions for drainage. Use this to test whether outlet geometry controls crater breaching. |
| `--stop-on-crater-breach` | flag | Stops the simulation as soon as flow crosses the crater rim according to the drainage-event detector. Requires `--drainage-only` to avoid incomplete magnetic post-processing. |
| `--crater-radius-m` | meters | Crater radius used for figures and geometric markers. |
| `--crater-mask-radius-m` | meters | Radius of the crater mask used by drainage-event detection. |
| `--event-check-interval` | years | Frequency for checking whether drainage has crossed the crater rim. |
| `--crater-event-min-area-km2` | km^2 | Minimum drained area required to validate a crater-rim crossing event. |

### Output And Runtime Controls

| Option | Unit / values | Meaning |
| --- | --- | --- |
| `--output-root` | directory path | Root folder for run outputs. The script creates a parameter-labelled subfolder inside it. |
| `--drainage-only` | flag | Skips magnetic post-processing. Gravity can still run if `--run-gravity` is also supplied. |
| `--no-save-figures` | flag | Disables writing figures to disk. Summary JSON and requested diagnostic files can still be written. |
| `--show-figures` | flag | Displays Matplotlib figures interactively. Usually keep disabled for batch runs. |
| `--no-progress` | flag | Disables the tqdm progress bar. Useful for cleaner logs. |
| `--log-progress-interval` | years | Controls how often text progress messages are printed. |
| `--save-topography-maps` | flag | Saves `maps_topography.png`, `maps_flexure.png`, and `topography_flexure_snapshots.npz`. |

## Outputs

Each run writes to a profile-specific output folder such as:

```text
outputs/Te30km_uplift0mMa_xy500m_crater2000_qsgeomorphic_layers_solversas_fillzero_marg0km_t1200100/
```

The folder name now records the flexure solver (`solversas` / `solverfd`) and
the margin (`marg0km`, or `marg3a` when `--flexure-margin-alpha` is used), so
runs that differ only by solver or margin do not overwrite each other.

When enabled, gravity outputs are written under the run folder in `gravity/`.
Topography, flexure, magnetic, drainage, and summary products are written in
the same run folder.
