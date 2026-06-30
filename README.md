# BevAD

Official code release for **[What Matters for Scalable and Robust Learning in End-to-End Driving Planners?](https://dmholtz.github.io/bevad/)** — accepted at **CVPR Findings 2026**.

*David Holtz, Niklas Hanselmann, Simon Doll, Marius Cordts and Bernt Schiele*<br>
*Mercedes-Benz AG &amp; Max-Plack-Institute for Informatics, SIC*

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2603.15185-b31b1b)](https://arxiv.org/abs/2603.15185)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://dmholtz.github.io/bevad/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-%F0%9F%92%AC-ff6b6b)](https://huggingface.co/dmhol/BevAD)

---

## 📰 News

- **2026-06** — Initial code and checkpoint release.
- **2026-06** — Publish Fail2Drive results. 
- **2026-03** — Paper release on arXiv.

## 🌐 Overview

```
bevad-workspace/
├── bevad/                  # Main codebase
├── bevad-sim/              # Connector between bevad and CARLA
├── checkpoints/            # Model checkpoints (download separately)
├── data/b2d-xml/           # Bench2Drive route definitions (XML)
├── external/               # Third-party dependencies (CARLA, mmcv)
├── recordings/             # Simulation episode stored here (created by inference.py)
└── inference.py            # Example script for closed-loop inference
```

## ⚙️ Installation

**Requirements:**

- Ubuntu 22.04 (recommended)
- Python 3.10
- CUDA 12 with a GPU of compute capability ≥ 7.0
- [CARLA 0.9.15](https://github.com/carla-simulator/carla/releases/tag/0.9.15) with additional large maps installed
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

**Setup:**

```bash
# Set the number of parallel compilation jobs (adjust to your CPU core count)
export MAX_JOBS=24
# Specify target CUDA architectures
export TORCH_CUDA_ARCH_LIST="7.0+PTX 7.5 8.0+PTX"
# Install all dependencies
uv sync -p 3.10
```

> **Note:** Installation takes 5–15 minutes depending on hardware. The majority of time is spent compiling mmcv from source — setting `MAX_JOBS` to match your available CPU cores speeds this up significantly.

## 🚗 Closed-Loop Simulation

1. Download the BevAD checkpoint from **[Hugging Face](https://huggingface.co/dmhol/BevAD/blob/main/model.ckpt)** and place it at `checkpoints/bevad-m.ckpt`.
2. Run closed-loop simulation on a Bench2Drive route:

```bash
python inference.py \
    --checkpoint checkpoints/bevad-m.ckpt \
    --config bevad/bevad/configs/cvpr/scaling_diffusion.py \
    --route data/b2d-xml/b2d-24224.xml \
    --output-dir recordings
```

All arguments are optional and default to the values shown above. To run a different route, replace the `--route` path with any XML file from `data/b2d-xml/`.



## 🐘 Fail2Drive

Fail2Drive evaluates both in-distribution performance and generalization under distribution shift.
These results were conducted by [Karol Fedurko](https://github.com/kafe-it) during his research internship in our lab.

| Method | In-Distribution DS ↑ | In-Distribution SR ↑ | In-Distribution HM ↑ | Generalization DS ↑ | Generalization SR ↑ | Generalization HM ↑ |
|---|---|---|---|---|---|---|
| TCP | 24.7 | 39.1 | 30.3 | 24.5 | 31.4 | 27.5 |
| UniAD | 47.5 | 36.3 | 41.2 | 44.0 | 27.6 | 33.9 |
| Orion | 53.0 | 52.0 | 52.5 | 51.2 | 46.0 | 48.5 |
| HiP-AD | 74.1 | 70.7 | 72.4 | 67.1 | 56.7 | 61.5 |
| SimLingo | 82.6 | 79.3 | 80.9 | 71.7 | 55.0 | 62.2 |
| TF++ | 83.3 | 78.5 | 80.8 | 75.4 | 61.1 | 67.5 |
| PlanT 2.0 | **87.8** | **85.0** | **86.4** | 73.3 | 58.0 | 64.8 |
| BevAD *(ours)* | 87.4 | 83.3 | 85.3 | **82.3** | **68.7** | **74.9** |

## 📚 Citation

```bibtex
@InProceedings{Holtz_2026_CVPRF,
    author    = {Holtz, David and Hanselmann, Niklas and Doll, Simon and Cordts, Marius and Schiele, Bernt},
    title     = {What Matters for Scalable and Robust Learning in End-to-End Driving Planners?},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {931-941}
}
```

## 📄 License

This project is licensed under the [MIT License](LICENSE). Note that it includes third-party components in the `external/` directory that are subject to their own license terms. See the [LICENSE](LICENSE) file for details.
