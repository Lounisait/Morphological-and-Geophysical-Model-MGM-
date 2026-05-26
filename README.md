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

The extended flexure domain can be filled with `zero`, `mean`, or `edge` values:

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

## Outputs

Each run writes to a profile-specific output folder such as:

```text
outputs/Te30km_uplift0mMa_xy500m_crater2000_qsgeomorphic_layers_fillzero_t1200100/
```

When enabled, gravity outputs are written under the run folder in `gravity/`.
Topography, flexure, magnetic, drainage, and summary products are written in
the same run folder.
