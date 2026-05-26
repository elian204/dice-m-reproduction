# DICE-M Reproduction

Private working repository for reproducing DICE and DICE with memory caching.

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
