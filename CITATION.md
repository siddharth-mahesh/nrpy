# How to Cite NRPy

If NRPy contributes to published scientific results, please cite:

1. The **general NRPy paper** below, and
2. Any **additional paper(s)** corresponding to the specific infrastructure or physics workflow you used.

> **Not sure which papers apply?** The safest default is to cite the **general NRPy paper** plus the paper that most closely matches the example, workflow, or infrastructure you used.

## Always cite

- **General NRPy / curvilinear-coordinate numerical relativity foundation**  
  Ian Ruchlin, Zachariah B. Etienne, and Thomas W. Baumgarte,  
  ["SENR/NRPy+: Numerical relativity in singular curvilinear coordinate systems"](https://doi.org/10.1103/PhysRevD.97.064036) ([arXiv](https://arxiv.org/abs/1712.07658)),  
  *Physical Review D* **97**, 064036 (2018).

## Quick reference

| If you used... | Also cite... |
|---|---|
| `nrpy.examples.nrpyelliptic_conformally_flat` | Assumpcao et al. (2022) |
| GPU / CUDA-enabled elliptic workflows | Tootle et al. (2025) |
| `superB` / Charm++ distributed-memory workflows | Jadoo et al. (2025) |
| `nrpy.examples.bhahaha` or `BHaHAHA` | Etienne et al. (2026) |
| `nrpy.examples.seobnrv5_aligned_spin_inspiral -seobnrv5_nrpy` | Pompili et al. (2023) |
| `nrpy.examples.sebobv2` | Mahesh et al. (2025) |
| `nrpy.examples.seobnrv5_aligned_spin_inspiral -seobnrv5_bob` | Mahesh et al. (2025) **and** Pompili et al. (2023) |

## Workflow- and infrastructure-specific citations

| Workflow / infrastructure | Citation |
|---|---|
| **Elliptic initial-data workflows** such as `nrpy.examples.nrpyelliptic_conformally_flat` | Thiago Assumpcao, Leonardo R. Werneck, Terrence Pierre Jacques, and Zachariah B. Etienne, ["Fast hyperbolic relaxation elliptic solver for numerical relativity: Conformally flat, binary puncture initial data"](https://doi.org/10.1103/PhysRevD.105.104037) ([arXiv](https://arxiv.org/abs/2111.02424)), *Physical Review D* **105**, 104037 (2022). |
| **GPU / CUDA-enabled elliptic workflows** | Samuel D. Tootle, Leonardo R. Werneck, Thiago Assumpcao, Terrence Pierre Jacques, and Zachariah B. Etienne, ["Accelerating numerical relativity with code generation: CUDA-enabled hyperbolic relaxation"](https://doi.org/10.1088/1361-6382/add63e) ([arXiv](https://arxiv.org/abs/2501.14030)), *Classical and Quantum Gravity* **42**, 115005 (2025). |
| **`superB` / Charm++ distributed-memory workflows** | Nishita Jadoo, Terrence Pierre Jacques, and Zachariah B. Etienne, ["superB/NRPy: Scalable, Task-Based Numerical Relativity for 3G Gravitational Wave Science"](https://doi.org/10.1088/1361-6382/adee71) ([arXiv](https://arxiv.org/abs/2505.00097)), *Classical and Quantum Gravity* **42**, 155006 (2025). |
| **Apparent-horizon finding** with `nrpy.examples.bhahaha` or `BHaHAHA` | Zachariah B. Etienne, Thiago Assumpcao, Leonardo Rosa Werneck, and Samuel D. Tootle, ["BHaHAHA: A Fast, Robust Apparent Horizon Finder Library for Numerical Relativity"](https://doi.org/10.1088/1361-6382/ae09e9) ([arXiv](https://arxiv.org/abs/2505.15912)), *Classical and Quantum Gravity* (2026). |
| **SEOBNRv5 aligned-spin waveform generation** such as `nrpy.examples.seobnrv5_aligned_spin_inspiral -seobnrv5_nrpy` | Lorenzo Pompili et al., ["Laying the foundation of the effective-one-body waveform models SEOBNRv5: Improved accuracy and efficiency for spinning nonprecessing binary black holes"](https://doi.org/10.1103/PhysRevD.108.124035) ([arXiv](https://arxiv.org/abs/2303.18039)), *Physical Review D* **108**, 124035 (2023). |
| **SEBOB workflows** such as `nrpy.examples.sebobv2` or `nrpy.examples.seobnrv5_aligned_spin_inspiral -seobnrv5_bob` | Siddharth Mahesh, Sean T. McWilliams, and Zachariah Etienne, ["Combining effective one-body inspirals and backwards one-body merger-ringdowns for aligned spin black hole binaries"](https://doi.org/10.1088/1361-6382/ae2413) ([arXiv](https://arxiv.org/abs/2508.20418)), *Classical and Quantum Gravity* (2025). For `-seobnrv5_bob`, also cite the SEOBNRv5 paper listed above. |

## Minimal default recommendation

- Cite the **general NRPy paper**.
- Add the **one paper most closely associated with the example, workflow, or infrastructure you used**.
- If you used `-seobnrv5_bob`, cite **both** the SEBOB paper and the SEOBNRv5 paper.
