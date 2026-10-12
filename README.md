# CloudOffEdge — diagnose locally, escalate rarely

Code, data and manuscript for

> **Diagnose locally, escalate rarely: bandwidth-efficient plant disease recognition with
> foundation models on embedded edge–cloud hardware**
> Youssef Natij, Hajar El Karch, Abdelkader Mezouari — submitted to *Computers and
> Electronics in Agriculture*. Manuscript: [`paper/main.pdf`](paper/main.pdf)
> (Word version: [`paper/main.docx`](paper/main.docx)).

A MobileNetV3-Large classifier on a **Raspberry Pi 5** diagnoses every field photograph
locally and escalates it to an ensemble of three frozen foundation backbones (DINOv2-L,
DINOv3-L, CLIP-L) only when its two most probable classes are close. The ensemble runs
on a **4 GB Jetson Nano** whose last software release predates every framework it
needs, through a chunked ONNX–TensorRT port and chunk-major batched inference.

## Results at a glance

Seven-day, four-farm replay on the devices (3 677 PlantWild test images, identical
images, schedule and payload in both runs):

| | Replay 1 | Replay 2 |
|---|---|---|
| Edge model | MobileNetV3-L | MobileNetV3-L distilled from the ensemble |
| CLIP precision on the Nano | fp32 | fp16 |
| Images escalated | 18.6 % | 12.4 % |
| Uplink (always-cloud: 216.3 MB) | 40.4 MB (5.4× less) | 27.0 MB (8.0× less) |
| Delivered top-1 accuracy | 76.0 % | 75.3 % |
| Answer on the farm (median) | 77 ms | 76 ms |
| Device energy per diagnosis | 9.5 J | 6.4 J |

Offline, on the PlantWild test split: the validation-calibrated margin rule escalates
18.6 % of images and retains 94.6 % of always-cloud accuracy (75.82 % vs. 80.17 %).

## Repository layout

```
paper/          manuscript sources (LaTeX), figures, numbers.tex, main.pdf, main.docx
rasp_deploy/    everything installed on the Raspberry Pi 5 (edge service, virtual farms, systemd)
jetson_deploy/  everything for the Jetson Nano: TensorRT runtime and service, and the
                workstation tools that export and build its engines
experiments/    workstation side: fleet emulator, operations console, training,
                distillation, calibration, measurement and paper-generation scripts
artifacts/      our trained edge models, the cloud head, the ensemble's soft labels,
                per-image predictions and the exact PlantWild split
results/        the measured data behind every number in the paper
docs/           ADAPTATIONS.md: every software adaptation the embedded port required
data/           where PlantWild goes (not redistributed)
```

## Reproduce

**The paper's numbers, figures and PDF from the shipped results** (CPU only, minutes):

```bash
pip install -r experiments/requirements.txt
python experiments/scripts/make_paper_numbers.py     # -> paper/numbers.tex (every measured value)
python experiments/scripts/make_paper_figures.py --fleet-run results/fleet_run
tectonic -X compile paper/main.tex                    # or upload paper/ to Overleaf
python experiments/scripts/make_docx.py              # optional Word version (needs pypandoc_binary)
```

**The experiments** (PlantWild images and a CUDA GPU for training): see
[`data/README.md`](data/README.md) and [`experiments/README.md`](experiments/README.md).

**The embedded system**: [`rasp_deploy/README.md`](rasp_deploy/README.md) (edge) and
[`jetson_deploy/README.md`](jetson_deploy/README.md) (cloud tier).

## Not included, and why

* **PlantWild images** — distributed by their authors (Wei et al., 2024); we ship the
  exact split (`artifacts/plantwild_split.csv`) instead.
* **Foundation-backbone weights, ONNX graphs and TensorRT engines** — third-party weights
  under their own licences, and engines are specific to the device and TensorRT version;
  `jetson_deploy/` rebuilds them.
* **Training of the cloud head** — the QualityGate fusion head is the model of El Karch
  et al. (2025); we ship the trained head and its per-image predictions.

## Contact

Youssef Natij — youssef.natij@uit.ac.ma
