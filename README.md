# Vishnu v1: Prototype-Oriented Distillation for Federated Continual Self-Supervised Vision
Implementation of Vishnu v1: Prototype-Oriented Distillation for Federated Continual Self-Supervised Vision

Here is a professional, research-grade **README.md** tailored exactly to your current progress. It uses the codenames we established and provides a clear checklist to track your development from Phase 1 to Phase 6.

Copy the content below and save it as `README.md` in your project root.

***

# FC-SSV-1: RAGNAROK 🐺⚡

**Federated Continual Self-Supervised Vision**

> *"Like the myth of Ragnarok (Cyclical Destruction and Rebirth), this system allows edge models to undergo constant updates while strictly preserving the essence of past knowledge."*

[![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-success)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![Framework](https://img.shields.io/badge/PyTorch-Federated-orange)]()

## 📖 Project Overview
**RAGNAROK** is a novel Federated Learning framework designed to address **Catastrophic Forgetting** in edge vision systems. It combines **Self-Supervised Learning (SSL)** with **Information-Bottlenecked Adapters (IBA)** to allow heterogeneous clients to learn from streaming, unlabeled data without sharing raw images.

### Key Innovations
1.  **IBA-Local (Information-Bottlenecked Adapters):** Injecting lightweight plasticity into frozen backbones.
2.  **PFO-Reg (Projective Feature Orthogonalization):** Constraining updates to preserve global feature subspaces.
3.  **DPM-Bank (Distributed Prototype Memory):** Privacy-preserving prototype sharing instead of raw data.
4.  **POD-Consolidator (Prototype-Oriented Distillation):** Server-side mechanism to merge disjoint client knowledge.

---

## 🏗️ Architecture & Modules

| Module Name | Component | Status | Description |
| :--- | :--- | :--- | :--- |
| **IBA-Local** | Client Model | ✅ Done | Adapter layers injected into frozen ResNet-18. |
| **SimCLR-Head** | Loss Function | ✅ Done | Contrastive loss for self-supervised pre-training. |
| **DPM-Bank** | Memory | 🚧 To Do | Local clustering and prototype storage. |
| **PFO-Reg** | Regularizer | 🚧 To Do | Orthogonal loss to prevent forgetting. |
| **Sparse-Comm** | Uplink | 🚧 To Do | Compressing adapters/prototypes for upload. |
| **POD-Engine** | Server | 🚧 To Do | Global aggregation and distillation logic. |

---

## 🗺️ Development Roadmap & Progress

### 🟢 Phase 1: The Local Client Engine (Current Status)
We have successfully built the "Plastic Brain" that lives on the edge device.
- [x] **Project Scaffold:** Directory structure and requirements setup.
- [x] **Backbone:** Implemented frozen ResNet-18 loading.
- [x] **IBA Implementation:** Created `IBALocalAdapter` (bottleneck design).
- [x] **Injection:** Successfully injected adapters into ResNet layers 1-4.
- [x] **SSL Pipeline:** Implemented SimCLR (NT-Xent) loss.
- [x] **Local Loop:** Verified training on CIFAR-10 (Adapters update, Backbone stays frozen).

### 🟡 Phase 2: Memory & Regularization (Next Steps)
Giving the client a memory and a way to respect global knowledge.
- [ ] **Implement DPM-Bank:** Feature extraction + K-Means clustering to generate prototypes.
- [ ] **Implement PFO-Reg:** Loss function to penalize feature drift from the Global Basis $U$.
- [ ] **Combined Loss:** `L_total = L_simclr + λ * L_pfo`.

### 🟠 Phase 3: Federated Simulation
Connecting multiple clients to a central server.
- [ ] **Setup Flower (flwr):** Create Client and Server classes.
- [ ] **Sparse-Comm:** Implement protocol to send *only* Adapters + Prototypes (no raw weights).
- [ ] **Aggregation:** Implement `FedAvg` for adapters.

### 🔴 Phase 4: Server Consolidation
The core novelty—distilling knowledge on the server.
- [ ] **Prototype Registry:** Server stores global prototypes.
- [ ] **POD Distillation:** Distill aggregated adapter into the Global Model using prototypes.
- [ ] **Basis Update:** PCA on global prototypes to update matrix $U$.

### 🟣 Phase 5: Downstream Evaluation
Proving it works.
- [ ] **Linear Probe:** Train a simple classifier on top of the frozen Global Encoder.
- [ ] **Forgetting Benchmark:** Evaluate on Task A (Old) after learning Task B (New).
- [ ] **Metrics:** Plot Accuracy vs. Rounds and Forgetting Curves.

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/yourusername/FC-SSV-RAGNAROK.git
cd FC-SSV-RAGNAROK
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run Local Client (Phase 1 Demo)
This script simulates a single edge device training self-supervised on local data.
```bash
python src/client/local_trainer.py
```
*Expected Output:* Training progress bar showing SimCLR loss decreasing over epochs.

---

## 📂 Repository Structure
```text
FC-SSV-RAGNAROK/
├── checkpoints/          # Saved model weights
├── configs/              # Hyperparameter configurations
├── src/
│   ├── client/
│   │   └── local_trainer.py  # Phase 1: Local SSL Loop
│   ├── models/
│   │   ├── adapter.py        # IBA-Local Module
│   │   └── resnet_ssl.py     # Frozen Backbone + Adapters
│   ├── server/               # (Coming in Phase 3)
│   └── utils/
│       └── losses.py         # SimCLR Loss
├── requirements.txt
└── README.md
```

---

## 📝 License
MIT License.

---
*Maintained by [Your Name]. Part of the FC-SSV Project Series.*