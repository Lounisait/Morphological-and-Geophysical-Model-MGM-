# MGM Examples

These scripts are meant to be run from the `MGM/` folder.

## Short Drainage-Only Run

```bash
./examples/run_short_example.sh
```

The default crater profile is `medium`. You can pass any bundled preset:

```bash
./examples/run_short_example.sh small
./examples/run_short_example.sh medium
./examples/run_short_example.sh large
```

Each run writes under `outputs/example_short_<profile>/`.

## All Bundled Craters

```bash
./examples/run_all_crater_profiles.sh
```

This runs the same short drainage-only check for `small`, `medium`, and `large`.
