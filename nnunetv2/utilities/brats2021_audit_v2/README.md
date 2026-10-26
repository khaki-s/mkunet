# BraTS audit model layout

This checkout groups implementations by architecture:

| Directory | Implementation |
|---|---|
| `baseline/` | Joint nnU-Net backbone, audited baseline variants and shared ExclusiveLoss |
| `semantic_moe/` | Original complete-input modality experts and semantic fusion |
| `agent_v1/` | Shared missingness trainer; ModDrop, SemanticMissing and one-round V1 communication |
| `agent_v2/` | Controlled identity initialization, refined auxiliary loss, optional null peer, diagnostics and checks |
| `evaluation/` | Strict read-only test100 comparison |
| `vendor/` | Existing third-party source and provenance |

Former flat model modules remain compatibility imports. State-dict keys are preserved for V1; V2 has separate trainer names and an explicit versioned resume contract. Shared input, resize, sampling and prediction utilities remain at this package root.

[Chinese implementation, results and training guide](/data02/chujingyi/brats_dataset_audit/architectures/agent_v2/implementation_20260930/README.md).

Use `/data02/chujingyi/nnUNet` for the ongoing SemanticMissing job and historical evaluations; use this checkout for V2. The launcher selects the correct code path and never starts a formal run unless `train`, `resume` or `test` is explicit.
