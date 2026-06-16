# bevad

This repository will contain the code of the paper "What Matters for Scalable and Robust Learning in End-to-End Driving Planners?".

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
