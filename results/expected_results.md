# Expected results (paper numbers)

VoD val, four seeds (0–3), mean ± std. Checkpoints are evaluated every epoch from 51 to 80; **best** is by entire-area R11 mAP.

| Model | R11 mAP (best epoch) | R40 mAP (best epoch) | Corridor mAP (R11) |
|---|---|---|---|
| Radar M3-Net (repro.) | 54.65 ± 0.88 | 52.35 ± 0.55 | 70.65 ± 0.50 |
| 4D-MambaNet w/o RCS input | 54.59 ± 0.84 | 52.29 ± 0.49 | 71.18 ± 0.17 |
| **4D-MambaNet** | **56.08 ± 0.19** | **53.91 ± 0.20** | **72.88 ± 1.20** |

Per seed, 4D-MambaNet R11: 56.226 / 56.156 / 56.154 / 55.797 (seeds 0–3).

Paired two-sided t-test (n = 4), 4D-MambaNet vs. M3-Net repro:

- R11 best: +1.43, p = 0.059
- R40 best: +1.55, p = 0.017

| TJ4DRadSet (best epoch) | mAP_3D | mAP_BEV |
|---|---|---|
| Radar M3-Net (repro.) | 33.53 ± 0.41 | 38.87 ± 0.83 |
| **4D-MambaNet** | **34.60 ± 0.57** | **39.63 ± 0.30** |

Runtime (RTX 3090): 4D-MambaNet 11.14 ± 0.26 FPS (network), 10.36 ± 0.44 FPS (end-to-end); 14.73 M parameters built, 10.30 M executed at inference.

**Note:** seed-to-seed variance on VoD is about 0.2–0.9 mAP (R11); always compare means over seeds 0–3.
