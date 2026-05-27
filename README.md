# DICE-M Reproduction

Public working repository for reproducing DICE and DICE with memory caching.

## Provenance

The baseline source snapshot was copied from:

`/home/dsi/eli-bogdanov/DICE(-M)`

The core Python files match the original anonymized GitHub snapshot in:

`elian204/dice_server_version`

at commit:

`7fbad586555268c27796492f50545245a23a6b57`

The later `dice_server_version` `main` branch contains a CLI-oriented refactor,
but that refactor currently does not import cleanly because it references a
missing `petri_net_classes.py` module. This repository starts from the known
working DICE-M source layout and should evolve in small, reviewable commits.

## Reproduction Notes

- Run DICE with `use_memo=False`.
- Run DICE-M with `use_memo=True`.
- Keep train/test splits, random seeds, window lengths, and model inputs fixed
  when comparing cost deviations.
- Breakfast uses the fixed 15 training traces recovered from the original
  experiment setup; all remaining Breakfast traces are used for testing.

## Paired Experiment Runner

List the configured DICE vs DICE-M paired runs:

```bash
python run_dice_memo_comparison.py --list-runs
```

Validate configured dataset/model paths without executing experiments:

```bash
python run_dice_memo_comparison.py --validate-paths
```

Run all configured experiments and write outputs under `dice_memo_comparison/`:

```bash
python run_dice_memo_comparison.py
```

For a small smoke run, select a single run and cap the filtered input traces:

```bash
python run_dice_memo_comparison.py \
  --run-id BPIC_2012__Set_2__w10 \
  --max-input-traces 12 \
  --output-dir /tmp/dice_memo_smoke
```
