# Related Files Archive

This archive was assembled for:

- `new copy 12.ipynb`
- `compae generatprs copy.ipynb`

It includes:

- The two notebooks above, with their current saved outputs.
- Root-level helper and generator wrapper modules imported by the notebooks.
- The CoCoGraPE Python source and configuration files, including transitive local imports.
- GRAN Python source and configuration files used by `gran_official_graph_generator.py`.
- GDSS Python source and configuration files used by `gdss_official_graph_generator.py`.
- Root, GRAN, and GDSS dependency/setup files.

It intentionally excludes:

- Git metadata.
- Python caches.
- Trained checkpoints and Lightning logs.
- Generated graphs, datasets, and evaluation artifacts.
- GRAN experiment output and precomputed data.
- GDSS checkpoints, training logs, generated data, and assets.
- Unrelated notebooks and the DiGress repository, which these two notebooks do not import.

The archive preserves repository-relative paths. Extract it so that the notebooks,
`coco_grape`, `GRAN`, and `GDSS` remain beside one another.
