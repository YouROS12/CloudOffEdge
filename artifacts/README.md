# artifacts — trained models, predictions and the dataset split

| File | Content | SHA-256 (prefix) |
|---|---|---|
| `edge_model/best.pt` | MobileNetV3-Small edge model (61.08 % test top-1) | `c1c091ba51d23d48` |
| `edge_model_effb0/best.pt` | EfficientNet-B0 edge model (69.35 %) | `03064583f323b8ea` |
| `edge_model_v3l/best.pt` | MobileNetV3-Large edge model, replay 1 (69.32 %) | `b77112a02ada3342` |
| `edge_model_v3l_kd/best.pt` | MobileNetV3-Large distilled from the ensemble, replay 2 (71.9 %) | `36e0f03d04159a66` |
| `cloud_head/qualitygate_fusion_3L.pt` | QualityGate fusion head over DINOv2-L/DINOv3-L/CLIP-L features (El Karch et al., 2025) | `bf6c8c6f4a843ae0` |
| `cloud_soft_labels.pt` | ensemble logits for the train and test splits (teacher for distillation) | `9900959773fad427` |
| `*_predictions_{val,test}.csv` (+ `.logits.pt`) | per-image predictions: edge label, confidence, margin, entropy; cloud label and confidence | |
| `plantwild_split.csv` | the exact PlantWild split (MVPDR protocol, seed 1): 13 045 train / 1 820 val / 3 677 test, relative paths, in loader order | |
| `class_names.txt` | the 89 class names in label order (edge and cloud services) | |

Prediction files use `edge_` = MobileNetV3-Small, `edge_effb0_`, `edge_v3l_`,
`edge_v3l_kd_`; image paths are relative to PlantWild's `images/` directory.

Edge checkpoints are dictionaries with `state_dict`, `arch`, `class_names` and
training metadata; the edge service needs only `state_dict` and `arch`. The replay-1
checkpoint on the Pi (`checkpoints/edge_v3l.pt`, SHA-256 `6b9acfe5…`) was a repackaged
copy holding only those two keys; its tensors are identical to
`edge_model_v3l/best.pt`. The replay-2 checkpoint is byte-identical to
`edge_model_v3l_kd/best.pt`.
