# LegacyV2：与主干同架构规模的第五专家

入口类：`nnUNetTrainer_BraTSAgentGeneralistFullSizeLegacyV2Loss300`。
旧 `nnUNetTrainer_BraTSAgentGeneralistLegacyV2Loss300` 及其所有文件、结果保留。

## 改动边界

- 第五专家改为用同一 plans 创建的官方 `PlainConvUNet`，与主干一致的
  encoder/decoder 宽度、层数、kernel、stride、norm 和 activation。
- 第五专家仍输入四个已屏蔽的 MRI 通道和四个可用性通道，故首层为 8 输入；
  主干为 4 输入。二者独立初始化，不复制或共享权重。
- 四个单模态专家、路由器、beta、peer-only 通信及修正方程保持不变。
  第五专家用于通信的 decoder level 3 特征为 256 通道，因此仅第五投影
  从 `Conv3d(64,16,1)` 改为 `Conv3d(256,16,1)`。
- 第五专家依旧只用最高分辨率输出计算专家辅助损失；官方构造器创建的
  低分辨率头保留并冻结。主干原来的 deep supervision 保留。
- 全部旧模块先按旧顺序构造，随后在 `fork_rng` 内替换第五专家及其投影。
  测试核对其余权重及构造后的 CPU RNG 状态与相同 seed 的旧模型完全一致。
- 300 轮、训练划分、采样、modality dropout、损失函数与系数均继承 LegacyV2。
  epoch 0–49 预热；50 起融合与模态丢弃；100 起专家间通信。

这不是纯粹加宽旧专家：旧专家使用 SimMLM vendor UNet，其中 decoder
的 stride=1 卷积也是 `ConvTranspose3d`；新专家与主干统一为官方
`PlainConvUNet`，并采用其官方 He 初始化。应称为“第五专家架构/容量消融”。

## 实际 Dataset137 plans 参数

plans：`/data02/chujingyi/data/nnUNet_preprocessed/Dataset137_BraTS2021/nnUNetPlans.json`。
宽度 `[32,64,128,256,320,320]`，六级，两层卷积/级。

| 部分 | 参数量（包括保留的冻结头） |
|---|---:|
| 主干 | 30,789,391 |
| 旧轻量 Generalist | 1,928,623 |
| 新同规模 Generalist | 30,792,847 |
| 旧整网 | 40,586,247 |
| 新整网 | 69,453,543 |

新 Generalist 比主干多的 3,456 个参数来自额外四个输入通道，
不是严格逐参数相等。主干与第五专家的 deep supervision 使用也不同。
实际训练 patch 为 `[128,160,112]`、batch size 为 2；GPU 显存与速度尚未实测。

## 融合顺序

此类沿用原先顺序：五专家 peer 通信修正各自 logits → 按 WT/TC/ET
分别加权融合专家 logits → 与主干 logits 做 beta 凸组合。

`z_final,c = (1-beta_c) z_joint,c + beta_c * sum_m(w_cm z_refined,mc)`。

仅把相同 beta 的线性加权改写为
`sum_m w_cm * ((1-beta_c)z_joint,c + beta_c z_refined,mc)`，
因为 `sum_m w_cm=1`，数学上相同，不构成新的融合方法。
本类不向主干中间特征注入专家信息，也不能单独回答两种非线性交互顺序谁更好。

## 执行

在原可训练环境中，选择空闲 GPU：

```bash
cd /data02/chujingyi/nnUNet
export PYTHONPATH=/data02/chujingyi/nnUNet
export nnUNet_raw=/data02/chujingyi/data/nnUNet_raw
export nnUNet_preprocessed=/data02/chujingyi/data/nnUNet_preprocessed
export nnUNet_results=/data02/chujingyi/data/nnUNet_results
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 137 3d_fullres 0 \
  -tr nnUNetTrainer_BraTSAgentGeneralistFullSizeLegacyV2Loss300
```

这是从头开始的独立 300 轮消融，不加载旧 LegacyV2 checkpoint，
不会覆盖旧 trainer 对应的结果目录。若旧会话设置过
`BRATS_AGENT_INITIAL_WEIGHTS`，需先清除；继承的 V2 训练协议拒绝该初始化方式。

CPU 合成数据回归检查（无训练更新，无患者推理）：

```bash
cd /data02/chujingyi/nnUNet
python scripts/brats2021_audit_v2/test_generalist_fullsize.py
```

现有 `evaluate_generalist_15_modalities.py` 和
`predict_generalist_missing_modalities.py` 可通过新类名/模型目录加载此模型；
仍需按原协议显式传入可用模态掩码，不通过全零 patch 猜测模态是否缺失。
