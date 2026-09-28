# BraTS2021 语义 MoE：300 轮实现与执行

已注册：

- `nnUNetTrainer_BraTSSemanticMoELoss300`：学习 WT/TC/ET 各自的四模态权重。
- `nnUNetTrainer_BraTSSemanticMoEUniformLoss300`：相同专家、初始化、损失与阶段，只把四专家权重固定为 1/4。

直接结构参照为已经有效的 `nnUNetTrainer_BraTSExclusiveLoss300`。本次没有新增 200 轮类。原注册文件名 `nnUNetTrainer_BraTSAudit200.py` 沿用用户目录布局，文件内公开类均为 300。

## 代码位置与复用

| 文件（相对 `/root/nnUNet`） | 职责 |
|---|---|
| `nnunetv2/utilities/brats2021_audit/semantic_blocks.py` | 四个专家、区域路由、logit 融合与 beta |
| `nnunetv2/utilities/brats2021_audit/semantic_network.py` | 原 nnU-Net 联合主干、辅助输出和推理协议 |
| `nnunetv2/utilities/brats2021_audit/semantic_trainer.py` | 300 轮、50/250 阶段、ExclusiveLoss/辅助 loss、AMP、恢复 |
| `nnunetv2/utilities/brats2021_audit/semantic_diagnostics.py` | 梯度、权重统计、固定验证 patch 的专家能力与贡献 |
| `nnunetv2/utilities/brats2021_audit/reproducibility.py` | 网络种子、确定性设置、按逻辑 batch 编号的采样/增强种子 |
| `nnunetv2/training/nnUNetTrainer/variants/brats2021_audit/nnUNetTrainer_BraTSAudit200.py` | nnU-Net CLI 类注册 |

四专家直接调用随代码固定的 `vendor/simmlm_unet.py:UNet`，通道为 `8/16/32/64/80/80`。UNet 类与用户提供的 SimMLM 源码逐字一致；唯一文件级改动是将演示用 `torchsummary` 导入移入 `__main__`，训练无需新增该依赖。来源与 SHA256 见 `vendor/simmlm_provenance.json`。

专家 stride 使用 Dataset137 plans，最后一级 `[2,2,1]`，适配 `128×160×112`。路由沿用五层 Conv/IN/ReLU 的设计，线性层输入由实际 patch 推导为 320，输出为 12；这不是原始 `DMoMEOutputLevel` 的完整复现。构造时最后线性层清零，使初始权重为 1/4。两版均构造相同 router，均匀版冻结并绕过它。

保留原始联合 nnU-Net；增加四个轻量单模态 UNet。专家训练启用 PyTorch 的非重入 activation checkpoint，以降低激活显存；没有修改专家类的计算公式。上游在关闭深监督后仍构造的低分辨率专家头被显式冻结，只有其最终头参与辅助监督。

## 损失和阶段

三个输出通道严格是 WT、TC、ET。内部互斥标签 ED=1、NCR=2、ET=3，NCR 仍需独立评价。

记联合主干输出为 z_joint，四专家输出 z_m；按区域 c 融合：

```text
w = softmax(router(image), modality)     # B × 3 × 4
z_moe,c = sum_m w_cm * z_mc
z_final,c = (1-beta_c)*z_joint,c + beta_c*z_moe,c
beta = sigmoid(beta_logit)              # 三个区域各自一个参数，初值 0.2
```

这四个权重只描述专家分支内部的融合；联合主干还处理全部四个输入。因此 w 不是整个模型的总模态贡献度。

| 轮数（从 1 开始） | 最终输出 | 训练目标 |
|---|---|---|
| 1–50 | 联合主干 | 原 ExclusiveLoss/深监督 + 四专家 region Dice+BCE 的平均值 |
| 51–300 | 联合主干与专家融合 | 原 ExclusiveLoss/深监督 + 0.25×专家融合 region loss + 0.25×四专家 region loss 平均值 |

原 ExclusiveLoss 的 `0.1` 互斥项和 `0.05` 层级项、低分辨率深监督及其权重均保留；仅最高分辨率输出在联合阶段替换为融合 logits。辅助输出直接接受 region Dice+BCE，梯度不经过 beta 才能到达专家。

训练总 loss 含辅助项，验证 loss 只计算原最终输出/深监督损失，两条 loss 的绝对值不应直接比较。保持原 SGD、初始 LR=0.01、Poly(300)、250 train iteration/epoch、50 val iteration/epoch、batch=2 和原 patch。第一版没有 MoFe、固定 MRI 权重先验、熵惩罚或局部质量正则。

## 同种子消融

默认 `BRATS_AUDIT_SEED=0`，所有 audit 300 trainer 共用如下策略：

1. 按原 nnU-Net 初始化流程先构造主干；在保存/恢复 RNG 的上下文内构造新增模块，避免消耗后续随机流。
2. 记录 `BACKBONE_INITIALIZATION` 的 encoder/decoder SHA256。Baseline300、ExclusiveLoss300、两版 SemanticMoE300 在同一 plans、seed 下必须一致；两版专家 SHA256 也必须一致。
3. 采样、裁剪和增强按 `seed + train/val stream + logical batch index` 派生种子，分别设置 Python、NumPy、CPU torch RNG。使用顺序固定的多进程增强器，避免按工作进程完成速度改变 batch 顺序。
4. 开启确定性算法，关闭 cuDNN benchmark；在训练开始及正式验证前重新设置，覆盖本地 CLI 在恢复后重置 cuDNN 标志的行为。
5. 从 epoch 边界恢复时，以 `current_epoch × iterations_per_epoch` 恢复数据流偏移。原始数据、plans、split 和代码应一致。运行环境/硬件变化不承诺位级完全一致。

这保证后续使用当前代码启动的消融具有相同初始化与数据随机策略，并不保证模块一定提高 Dice。之前按旧的非确定性增强器完成的结果不能追溯为相同 patch 序列。

## 日志

普通 `training_log_*.txt` 和 `module_diagnostics.jsonl` 同时写入：

- `SEMANTIC_PHASE`：当前预热/联合阶段、router 是否训练、loss 系数。
- `SEMANTIC_LOSS`：最终 ExclusiveLoss/深监督、四专家辅助 loss、融合辅助 loss。
- `MODULE_GRAD_FEATURE`：每 50 step 采样，各专家/router/beta 的梯度、参数更新、logit 改变量。`grad_none_trainable_tensors` 单独统计应训练但无梯度的参数；冻结参数另计。
- `SEMANTIC_ROUTING`：每轮按 WT/TC/ET、T1/T1ce/T2/FLAIR 输出均值、标准差、最小/最大值、熵和 beta。
- `SEMANTIC_CONTRIBUTION`：第 1、25、50、51、75…轮，在固定验证 patch 上记录各专家的 WT/TC/ET/ED/NCR Dice，以及仅联合主干、固定均匀权重、移除单个专家融合边的对照。

固定 patch 缓存在 `semantic_validation_probe.pt`，恢复时复用，并记录 SHA256。所有 counterfactual Dice 差值均为完整模型减对照，单位百分点，正值表示当前模型在该 patch 上受益。移除专家只移除它的融合边并重归一化，联合主干/router 仍读取四模态；这不是删除输入模态的实验，也不是完整病例 Dice。

预热时 router/beta 没有梯度是设计行为；均匀对照的 router 全程冻结。checkpoint 保存融合阶段 buffer 和全部专家参数，无需外部专家权重文件。第 50 轮的 checkpoint 在验证时保留联合主干输出，继续训练第 51 轮才启用融合。

## 执行

启动脚本已全部改为 300 类名，新增 `semantic_moe_loss` 和 `semantic_moe_uniform_loss`。脚本按照原 `11.txt` 流程保留训练结束后的关机操作。每次只运行一个实验。

学习路由：

```bash
BRATS_AUDIT_SEED=0 nohup bash /root/nnUNet/scripts/brats2021_audit/run_experiment.sh semantic_moe_loss \
  > /root/nnUNet/scripts/brats2021_audit/train_semantic_moe_loss300.log 2>&1 < /dev/null &
```

均匀对照：

```bash
BRATS_AUDIT_SEED=0 nohup bash /root/nnUNet/scripts/brats2021_audit/run_experiment.sh semantic_moe_uniform_loss \
  > /root/nnUNet/scripts/brats2021_audit/train_semantic_moe_uniform_loss300.log 2>&1 < /dev/null &
```

继续同一实验，在脚本模式名后增加 `--c`。两版是不同实验目录，不能交叉 resume。对照使用相同 seed 的 `run_experiment.sh loss`（ExclusiveLoss300）和 `baseline`（Baseline300）。

若直接使用 CLI，在已有 nnU-Net 环境和数据路径下：

```bash
BRATS_AUDIT_SEED=0 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 nnUNet_compile=False \
  nnUNetv2_train 137 3d_fullres 0 --npz -tr nnUNetTrainer_BraTSSemanticMoELoss300
```

均匀版只替换 `-tr` 的类名。预测保留原数据流程，显式指定 `-f 0`、`-c 3d_fullres` 和对应 `-tr`，checkpoint 使用同一实验的 `checkpoint_best.pth` 或 `checkpoint_final.pth`。不要用当前辅助权重重新解释旧 trainer 的 checkpoint。

## 检查范围

检查入口：

```bash
/root/miniconda3/envs/lightmunet/bin/python -m nnunetv2.utilities.brats2021_audit.semantic_check --loader
```

包含原 UNet 类一致性、实际 plans 初始化哈希、三个随机源的 batch 顺序/恢复偏移、预热与联合梯度、深监督/普通推理输出、阶段边界与 checkpoint 输出恢复、完整 patch 的 meta 形状。`--loader` 还读取真实训练 batch 核对相同增强结果。合成训练检查在临时目录执行，不写正式实验权重。

当前无 GPU，没有执行完整 patch 的 CUDA/AMP/显存实测。有卡后可先执行一次：

```bash
/root/miniconda3/envs/lightmunet/bin/python -m nnunetv2.utilities.brats2021_audit.semantic_check --gpu \
  --output /root/nnUNet/documentation/brats2021_semantic_gpu_checks.json
```

GPU 检查使用实际 `batch=2, patch=128×160×112` 做合成 AMP 前后向并记录显存峰值，仍不启动正式训练。结果记录与本次 CPU 检查分开。正式配置没有因无卡环境而改变日志模式。

原始用户代码备份：`/root/brats_dataset_audit/semantic_moe_300_code_backup/`。
