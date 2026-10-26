# Generalist extensions: provenance

This package is a 3D adaptation, not a reproduction of an entire published agent system.

* **Generalist expert, explicit availability channels, five-way region routing:** project extension requested for this experiment. The joint nnU-Net remains independent.
* **Expert UNet / directly supervised Dice+BCE:** existing vendored SimMLM UNet; verified original ICCV 2025 [Dice+BCE implementation](https://github.com/LezJ/SimMLM/blob/c9be74ba913db08013b8e573ce6cbda25da9ea1a/loss/dice_bce_loss.py). This package does not implement SimMLM's MoFe ranking loss.
* **Headwise attention output gate:** follows the mechanism in [NeurIPS 2025 Gated Attention](https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html), official [modeling_qwen3.py, lines 351–366](https://github.com/qiuzh20/gated_attention/blob/f4c2a5f6ffd6ec709e0c60072c95ed4f5ce5b5d2/modeling_qwen3.py#L351). This is a mathematical adaptation to pooled expert states, not copied Qwen model code. The original repository is Apache-2.0; no Qwen dependency is introduced.
* **Receiver-query/sender-key-value targeting and self-inclusive candidate set:** mathematical precedent [TarMAC, ICML 2019](https://proceedings.mlr.press/v97/das19a.html). No claim of official TarMAC code reuse.
* **Spatial 3D values, disagreement input, fixed 0.1 residual, zero correction head, auxiliary loss coefficients:** existing project design / explicitly documented adaptation. No claim these are experimentally optimal or jointly prescribed by the papers.
* **Legacy controls:** preserve the old communication equations with five experts; legacy local means zero messages, while the self-inclusive local control means self-attention only.

Experiment contracts and full mathematical explanation: `/data02/chujingyi/brats_dataset_audit/generalist_review_20261014/02_GENERALIST_IMPLEMENTATION_AND_TRAINING.md` and `03_COMMUNICATION_MATH_AND_REFERENCES.md`.
