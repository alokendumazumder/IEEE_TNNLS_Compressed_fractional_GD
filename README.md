# Fractional Gradient Descent with Matrix Stepsizes (CFGD)

### Overview
This repository contains the official implementation of the algorithms proposed in the paper:
> **Fractional Gradient Descent with Matrix Stepsizes for Non-Convex Optimisation**  
> *Authors: Alokendu Mazumder, Keshav Vyas, and Punit Rathore*  
> *Robert Bosch Center for Cyber Physical Systems, Indian Institute of Science, Bengaluru*  
> *Preprint posted on TechRxiv, 2025*

---

## 🚀 Introduction
This work introduces two algorithms:
- **CFGD-1**
- **CFGD-2** (also referred to as **Distributed CFGD (DCFGD)** in the distributed setting)

Both are novel extensions of fractional gradient descent (FGD), designed for non-convex and matrix-smooth optimisation problems. CFGD incorporates **matrix-valued stepsizes** and **compression mechanisms**, allowing efficient large-scale distributed training.

The algorithms extend standard and fractional gradient descent to the distributed and federated learning domains, showing improved convergence and communication efficiency.

---

## 📂 Repository Structure
```
├── cfgd_vs_cgd.py       # Implementation of CFGD and DC(FGD) algorithms
├── plot.py              # Visualization utilities for convergence and comparison
├── experiments.py       # Experimental setup for single-node and distributed cases
├── get_data.py          # Dataset loading and preprocessing
├── get_scheduler.py     # Learning rate scheduler utilities
├── models.py            # Model definitions for experiments
├── utils.py             # Helper functions
├── figures/             # Folder containing all result figures (9 plots assumed)
│   ├── fig1.png
│   ├── fig2.png
│   ├── fig3.png
│   ├── fig4.png
│   ├── fig5.png
│   ├── fig6.png
│   ├── fig7.png
│   ├── fig8.png
│   └── fig9.png
└── README.md            # Project documentation (this file)
```

---

## ⚙️ Algorithms
The repository implements the following key algorithms:

- **CFGD-1:** Compressed Fractional Gradient Descent with matrix stepsize D applied before sketching.
- **CFGD-2:** Variant where sketching precedes the matrix stepsize operation.
- **DCFGD-1 and DCFGD-2:** Distributed versions of CFGD-1 and CFGD-2 for federated environments.

These are designed to handle both **single-node** and **multi-client distributed setups** efficiently.

---

## 🧠 Key Ideas
- Introduces **matrix-valued stepsizes** to leverage structure in non-convex matrix-smooth objectives.
- Employs **fractional-order gradients (Caputo derivative)** to accelerate convergence.
- Incorporates **communication-efficient sketching/compression** to reduce distributed overhead.
- Demonstrates theoretical **O(1/√T)** convergence for matrix-smooth non-convex functions.
- Provides practical improvements in both iteration and communication complexity compared to DCGD and det-CGD.

---

## 🧩 Implementation Highlights
- **cfgd_vs_cgd.py:** Core implementation of CFGD-1, CFGD-2, DCFGD-1, and DCFGD-2.
- **plot.py:** Includes plotting utilities to reproduce convergence plots.
- **experiments.py:** Recreates results for logistic regression tasks in both single-node and distributed settings.
- **utils.py:** Provides general helper functions and reproducibility tools.

---

## 🧪 Experiments
The experiments are divided into two categories:

### 1. Single Node Experiments
- Tests convergence of CFGD-1 and CFGD-2 on logistic regression tasks.
- Compares against vanilla GD, FGD, and DCGD.
- Demonstrates faster convergence when using matrix-valued stepsizes.

### 2. Distributed Experiments
- Evaluates DCFGD-1 and DCFGD-2 in federated setups.
- Compares performance with DCGD, det-CGD, and det-MARINA.
- Shows superior communication and iteration efficiency.

---

## 📈 Results
Below is a placeholder 3×3 results grid showing sample figures from the `figures/` folder.

| ![](figures/fig1.png) | ![](figures/fig2.png) | ![](figures/fig3.png) |
|:----------------------:|:----------------------:|:----------------------:|
| ![](figures/fig4.png) | ![](figures/fig5.png) | ![](figures/fig6.png) |
| ![](figures/fig7.png) | ![](figures/fig8.png) | ![](figures/fig9.png) |

> **Figure 1–9:** Comparative performance of CFGD and DCFGD under different sketches and step-size configurations. *(You can update these captions later.)*

---

## 📚 Citation
If you use this repository or build upon this work, please cite:

```bibtex
@article{mazumder2025cfgd,
  title={Fractional Gradient Descent with Matrix Stepsizes for Non-Convex Optimisation},
  author={Alokendu Mazumder and Keshav Vyas and Punit Rathore},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2025}
}
```

---

## 🧾 License
This repository is released under the **MIT License**. Please see the `LICENSE` file for more details.

---

## 🙌 Acknowledgements
This research was conducted at the **Robert Bosch Center for Cyber-Physical Systems**, Indian Institute of Science (IISc), Bengaluru.  
We thank the open-source community for providing supporting packages such as PyTorch, NumPy, and CVXPY.

---

## 📬 Contact
For questions or collaborations, please contact:
- **Alokendu Mazumder** — alokendum@iisc.ac.in
- **Punit Rathore** — prathore@iisc.ac.in

---

> *This code accompanies the paper "Fractional Gradient Descent with Matrix Stepsizes for Non-Convex Optimisation" (TechRxiv Preprint, 2025).*

