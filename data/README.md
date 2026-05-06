# MGM Input Data

This folder contains bundled radial profiles used by the crater presets in
`main_simulation_harmonized.py`.

## Preset Mapping

| Preset | Topography profile | Bouguer profile | Domain half-width |
| --- | --- | --- | --- |
| `small` | `surf_profile_700.csv` | `bouguer_700.csv` | `60000` m |
| `medium` | `surf_profile_2000.csv` | `bouguer_2000.csv` | `80000` m |
| `large` | `surf_profile_5000.csv` | `bouguer_5000.csv` | `160000` m |

The topography profiles are read by `Topo.generate_topography`. The Bouguer
profiles are read by the gravity workflow when initial gravity background data
are enabled.
