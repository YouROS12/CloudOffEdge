# data — PlantWild (not redistributed)

The experiments use **PlantWild** (Wei et al., 2024: *Plant Disease Recognition in the
Wild*, ACM MM 2024), 18 542 in-the-wild images of 89 classes. Its images are
distributed by its authors under their terms and are not included here; obtain the
dataset from them.

1. Place the images as `data/PlantWild/images/<class name>/<image>` (or set
   `PLANTWILD_ROOT` to wherever its `images/` directory is).
2. The split used throughout the paper is shipped as `artifacts/plantwild_split.csv`
   (relative path, label, split; 13 045 / 1 820 / 3 677) — no split generation needed.
3. For the replay and the fidelity checks, lay the test split out by class:

```bash
python experiments/scripts/plantwild.py --materialize-test data/plantwild_test   # hard links (--copy to copy)
```

Optional: `data/paper1_cached_features/plantwild__{dinov2_vitl14,dinov3_vitl16,clip_vitl14}.pt`
(backbone features from paper 1's pipeline) are only needed to recompute
`artifacts/cloud_soft_labels.pt` with `experiments/scripts/cloud_soft_labels.py`
(or set `PAPER1_CACHED_FEATURES`).
