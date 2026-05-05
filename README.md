# Morphological and Geophysical Model (MGM)

MGM is a coupled research model for impact-crater landscape evolution and
geophysical signatures. The workflow combines:

- topographic evolution with Landlab components,
- sediment production and transport,
- lithospheric flexure,
- gravity anomaly modelling,
- magnetic anomaly modelling,
- drainage-event and crater-diameter diagnostics.

## Repository Contents

- `main_simulation_harmonized.py`: main simulation entry point.
- `function_flexure.py`: flexural-load and deflection calculations.
- `Gravi.py` and `sub_routine.py`: gravity anomaly calculations.
- `mag_harmonized.py` and `mag_crater_dipole_harmonized.py`: magnetic anomaly calculations.
- `Topo.py`: topography and input-grid generation utilities.
- `data/`: small example input profiles.
- `examples/run_short_example.sh`: short smoke-test run.

Large simulation outputs are intentionally not included in Git.

## Installation

Create a Python environment, then install the dependencies:

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

From the `MGM/` folder:

```bash
python main_simulation_harmonized.py --drainage-only --no-save-figures
```

The default input files are:

- `data/surf_profile_2000.csv`
- `data/bouguer_2000.csv`

Outputs are written under `outputs/` by default.

## Useful Run Options

Short test run:

```bash
python main_simulation_harmonized.py \
  --total-time 10100 \
  --dt 100 \
  --dt-flex 1000 \
  --drainage-only \
  --no-save-figures \
  --no-progress
```

Run with gravity post-processing:

```bash
python main_simulation_harmonized.py \
  --run-gravity \
  --gravity-method all \
  --gravity-kernel numba
```

Run with magnetic post-processing disabled:

```bash
python main_simulation_harmonized.py --drainage-only
```

Change elastic thickness and uplift rate:

```bash
python main_simulation_harmonized.py --Te 40000 --uplift 30
```
