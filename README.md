# PINNs · Burgers Equation Reproduction

Reproduction of [Raissi, Perdikaris & Karniadakis, JCP 2019](https://arxiv.org/abs/1711.10561).

PINNs solve PDEs by embedding the governing equation into the loss function. A neural network u(x,t) approximates the solution; automatic differentiation computes the PDE residual. Loss = data loss (boundary/initial conditions) + physics loss (PDE residual at interior points). Training minimizes both, producing a continuous solution without mesh generation.

We reproduced the Burgers equation: u_t + u u_x = (0.01/pi) u_xx.

Three iterations:

| Iteration | Approach | Result |
|---|---|---|
| v0 | plain Adam, no normalization | loss plateaued |
| v1 | Xavier init + input normalization | better convergence |
| **v2** | **Adam warm-up -> L-BFGS fine-tune** | **close to reported accuracy** |

Relative L2 error: **1.77e-03** (paper: 6.7e-04).

## Setup

| Item | Value |
|---|---|
| Architecture | 8 layers x 20 neurons, tanh |
| Initialization | Xavier |
| Input normalization | [-1, 1] |
| Boundary/IC | 100 points (from exact solution) |
| Collocation | Latin Hypercube x 10000 |
| Optimizer | Adam 2000 -> L-BFGS <= 30000 |
| Reference | burgers_shock.mat (spectral method) |

## Training log

```
Adam 2000 steps...
  Adam    0, Loss=3.46e+00
  Adam  500, Loss=8.61e-02
  Adam 1000, Loss=6.60e-02
  Adam 1500, Loss=5.36e-02
L-BFGS fine-tune...
  L-BFGS  1000, Loss=1.54e-04
  L-BFGS  2000, Loss=4.09e-05
  L-BFGS  3000, Loss=1.73e-05
  L-BFGS  4000, Loss=8.12e-06
Done! 4940 steps. L2 error: 1.77e-03
```

## Files

```
pinns_burgers_v2.py    main script
pinns_result_v2.png    result figure
```

> **Note**: You also need `burgers_shock.mat` (the reference solution) in the same directory. Download it from the [official PINNs repository](https://github.com/maziarraissi/PINNs) at `appendix/Data/burgers_shock.mat`.

## Run

```bash
pip install torch numpy matplotlib scipy
# First download burgers_shock.mat (see note above)
python pinns_burgers_v2.py
```

## Citation

```bibtex
@article{raissi2019physics,
  title={Physics-informed neural networks},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal={Journal of Computational Physics},
  volume={378},
  pages={686--707},
  year={2019}
}
```
