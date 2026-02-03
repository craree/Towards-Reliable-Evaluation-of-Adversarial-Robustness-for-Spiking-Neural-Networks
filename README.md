# Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks

Official implementation of:

**Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks**  
Jihang Wang, Dongcheng Zhao, Ruolin Chen, Qian Zhang, Yi Zeng

paperlink： https://arxiv.org/abs/2512.22522

This repository provides the **first reliable evaluation framework** for adversarial robustness in Spiking Neural Networks (SNNs).

> We show that the adversarial robustness of many existing SNNs has been **significantly overestimated** due to unreliable gradient approximation and suboptimal attack optimization. 

---

## 🚨 Why This Repository Matters

Most prior SNN robustness evaluations rely on:

- Fixed surrogate gradients (STBP / Atan / Sigmoid)
- Short PGD attacks (5–10 steps)
- Standard PGD / APGD optimizers

We demonstrate that these practices produce **false robustness**.

> SNNs appear robust **because attacks fail**, not because the models are genuinely robust.

This repo provides a **reliable, theoretically grounded, and empirically validated** evaluation pipeline.

---

## 🧠 Core Ideas

This work improves SNN robustness evaluation from two fundamental aspects:

### 1️⃣ Adaptive Sharpness Surrogate Gradient (ASSG)

We theoretically analyze why surrogate gradients suffer from **gradient vanishing** during adversarial attacks and propose an **input-dependent, spatial-temporal adaptive** surrogate gradient.

Instead of a fixed sharpness parameter α, ASSG dynamically computes:

```
α_{i,t}^{l} = ω / E[ |u_{i,t}^{l}| ]
```

during attack iterations, ensuring:

- Accurate gradient approximation
- Controlled gradient vanishing
- Strong adaptivity across neurons, layers, and time steps

---

### 2️⃣ Stable Adaptive Projected Gradient Descent (SA-PGD)

Even with better gradients, PGD/APGD still fail in SNN input space due to its non-smooth landscape.

SA-PGD introduces:

- L1-normalized momentum
- Adaptive step size
- Per-step L∞ clipping for stable optimization

Attacks continue to improve **even after hundreds or thousands of iterations**.

---

## 🔥 Key Results

Using ASSG + SA-PGD, we show:

- ~10% higher Attack Success Rate (ASR) than prior SOTA methods
- Previous gradient methods (BPTR, RGA, PDSG, HART) overestimate robustness
- Some SNNs exhibit **false robustness** without ASSG
- Adversarial training with 5-step PGD is far from optimal

> This work calls for **rethinking adversarial training for SNNs**.

---

## 🧪 Supported Experimental Settings

- **Datasets**: CIFAR-10, CIFAR-100, CIFAR10-DVS
- **Neuron models**: LIF, LIF-2, IF, PSN
- **Encoders**: Direct, Poisson (with EOT)
- **Architectures**: SEWResNet19, VGG9
- **Training methods**: AT, RAT, AT+SR, TRADES
- **Surrogate functions**: Atan, Sigmoid, Gaussian, Rectangular, Triangular

ASSG generalizes to **all surrogate gradients**.

---
## 🚀 Quick Start

### Training 

* To train a SEWResNet19 with AT on CIFAR-10:
```
python train.py --adv_training --pgd_iters 5 --eps 8 --alpha 4 --batch_size 256 --worker 4 --node_type LIF --save_dir AT --device_ids 0 --network SEWResNet19C --Robust_Loss AT --reg 0.004 --beta 10 --time_step 4 --dataset cifar10
```

* To train a SEWResNet19 with TRADES on CIFAR-10:

```
python train.py --pgd_iters 4 --eps 8 --alpha 3 --batch_size 256 --worker 4 --node_type LIF --save_dir TR5 --device_ids 0 --network SEWResNet19C --Robust_Loss TRADES --reg 0.004 --beta 5 --time_step 4 --dataset cifar10
```
---

## Evaluation

* To evaluate with ASSG and SA-PGD:

```
python evaluate.py --attack_type assg-sapgd --batch_size 256 --worker 4 --node_type LIF --network SEWResNet19C --pretrain AT/weight_c.pth --device_ids 6 --time_step 4 --dataset cifar10 --relax 1.5 --beta_1 0.9 --beta_2 0.9
```

* To compare with prior sota method (STBP, RGA, PDSG, BPTR, HART):

```
python evaluate.py --attack_type abla_all --batch_size 256 --worker 4 --node_type LIF --network SEWResNet19C --pretrain AT/weight_c.pth --device_ids 2 --time_step 4 --dataset cifar10
```

To obtain the **best attack success rate (ASR)**, we recommend selecting the optimal `alpha` via **ternary search** over a predefined range, e.g. [0.6, 0.98] for .

You will observe a large ASR gap consistent with the paper. 


## 🔧 Where ASSG Is Implemented

The core implementation of **ASSG** is located in:

- my_node.py  →  class ASSGrad (ASSG for Atan), ASSGrad_R (ASSG for Rectangular), ASSGrad_T (ASSG for Triangular), ASSGrad_S (ASSG for Sigmoid),  ASSGrad_G (ASSG for Gaussian)


We provide a unified interface to switch the backpropagation (surrogate gradient) behavior: 

> model.module.set_backpropagation(...)

---

## 🧩 Key Insight

> “Without ASSG, SNNs may even exhibit false robustness.”

Many previous SNN robustness claims are likely invalid due to flawed evaluation settings.

Please use **ASSG + SA-PGD** for reliable evaluation.
