# 交互修正核心：代码来源与实现边界

## 复用与适配的区分

1. **直接复用的模型类**：`dynamic_network_architectures.architectures.unet.PlainConvUNet`。
   - [官方 dynamic-network-architectures 仓库](https://github.com/MIC-DKFZ/dynamic-network-architectures)。
   - 本地 `model.py` 直接 import 该类；encoder、decoder、卷积、归一化均由已有库实现。
   - 新增的是 21 通道输入契约、输出残差限幅、零初始化输出层和 checkpoint 格式。它们是本项目适配代码，**不是 nnInteractive 官方预训练模型的等价复现**。

2. **交互会话依据**：[nnInteractive 官方仓库](https://github.com/MIC-DKFZ/nnInteractive)。
   - 本地已核查快照 commit：`bb29d26b90211162d9b3c8fe5ef2cdf2d4941581`。
   - [该 commit 的 inference_session.py](https://github.com/MIC-DKFZ/nnInteractive/blob/bb29d26b90211162d9b3c8fe5ef2cdf2d4941581/nnInteractive/inference/inference_session.py) 提供 `set_image`、`set_target_buffer`、`add_point_interaction`、`add_bbox_interaction` 和 undo。
   - 本项目遵循“当前分割＋空间提示＋多轮会话”的范式，同时显式适配四模态缺失、WT/TC/ET 嵌套类别和局部修改边界。
   - 对于希望直接使用官方权重的用户，使用同目录官方后端适配器；这与本地新增残差修正器是两个后端，不共享权重。

3. **损失**：直接调用 `torch.nn.functional.binary_cross_entropy_with_logits`，只在 `known=True` 的区域头/体素位置求均值。没有臆造“LLM 置信度就是 GT”的训练损失。

## 张量与语义

`TorchPromptCorrector` 输入 `[B,21,D,H,W]`：

| 通道 | 含义 |
| --- | --- |
| 0–3 | 当前网格上的已归一化 MRI，顺序 T1、T1ce、T2、FLAIR |
| 4–7 | 四个模态的可用性标记 |
| 8–10 | 上一轮 WT、TC、ET 概率 |
| 11–13 | 本轮目标区域 one-hot |
| 14–16 | WT、TC、ET 正点的物理空间 Gaussian |
| 17–19 | WT、TC、ET 负点的物理空间 Gaussian |
| 20 | 显式框内部：正框 +1、负框 -1；框外和仅限制修改 ROI 时为 0 |

MRI、logits、点击、ROI 必须在**同一个实际数组网格**。传入 affine 必须描述这个数组网格到世界毫米坐标的变换；不能把原图 affine 直接套到已被 nnUNet transpose/resample 的数组上。若使用原始 NIfTI 网格，须先将概率逆变换回该网格，再提供同网格归一化 MRI。高斯图使用 affine 的完整线性部分计算距离，包含轴旋转/斜切。

核心推理关系：

```text
既有自动分割 logits + 可用 MRI + 真实用户注册的点/框
  -> 构建 ROI 内 21 通道输入
  -> PlainConvUNet 输出 logits 增量
  -> 对 include/exclude 只保留目标头增量
  -> ROI 内渐变权重
  -> 嵌套关系约束 + 可选点击硬约束
  -> 提交当前轮；可精确撤回
```

框表示交互提示，不意味着框内每个体素都是真值。负 ET 点只产生 `ET=0` 的已知条目，WT/TC 保持未知。明确替换为 NCR 才产生 `(WT,TC,ET)=(1,1,0)` 三项已知监督。点击硬约束带来的满足率与网络自己预测的满足率分别报告，不能将硬约束成功当成学会了分割。

2026-10-26 补充：`BoxPrompt` 单独保存真实标注框；`SpatialROI` 保存允许修改的三维范围。文字命令显式传入 `box_prompt=true` 和已注册 `box_id`；LLM不能发明或替换框。官方 v1 权重默认 `bbox3d=False`，因此用某一轴厚度为1的切片框驱动三维 ROI 内分割。官方后端拒绝把整个三维编辑 ROI 默认为框；未来权重若明确声明支持 `bbox3d` 才允许三维框。依据见 [官方 inference_session.py](https://github.com/MIC-DKFZ/nnInteractive/blob/master/nnInteractive/inference/inference_session.py)。

新模型输出层零初始化；未训练时不会自动产生有质量保证的局部扩张。即便硬约束能把用户点击的单点改对，也不能据此宣称模型已改善周边分割。

## 离线训练输入与运行

训练入口：

```bash
python scripts/brats2021_audit_v2/interactive/train_corrector.py \
  --manifest /absolute/path/corrector_train_manifest.json \
  --splits-file /absolute/path/splits_final.json --fold 0 \
  --output /absolute/path/new_corrector_checkpoint.pt \
  --device cuda --steps 1000 --rounds-per-case 3 --roi-size 64 64 64
```

以上 `1000` 只是运行示例，不是已验证最优训练轮数。这个入口只训练修正器，不改变任何原有 nnUNet trainer 或 checkpoint。

Manifest 示例：

```json
{
  "simulation": [
    {"path": "/absolute/path/train_case.npz", "split": "train", "case_id": "BraTS2021_x"}
  ],
  "feedback": ["/absolute/path/human_confirmed_train_feedback"]
}
```

模拟样本 NPZ：

| 字段 | 形状/说明 |
| --- | --- |
| `mri` | `[4,D,H,W]` 同网格归一化图像 |
| `modality_mask` | `[4]` 二进制标记；缺失图像通道必须为零 |
| `previous_logits` | `[3,D,H,W]` 既有自动模型在训练病例上的预测 |
| `affine` | `[4,4]` 此数组网格到毫米空间 |
| `region_gt` | `[3,D,H,W]` 原训练集真值；或提供下一行字段 |
| `atomic_labels` | `[D,H,W]` 当前内部标签 `0=BG,1=ED,2=NCR,3=ET` |

模拟训练从最大 FP/FN 错误连通区域内部选点，并逐轮将当前修正结果作为下一轮输入。这是本地透明实现的错误驱动采样，未声称复制 nnInteractive 的完整采样策略。MRI/预测可预先覆盖不同模态组合；训练入口不会自动替你生成 15 种输入，也不从缺失模态重建完整信息。

训练入口可用 `--targets ET` 限制模拟目标（默认仍WT/TC/ET），用 `--max-delta 16` 为新纠错器显式配置残差范围（默认仍8）。旧默认对绝对值大于8的错误logit无法单次靠网络残差翻转；硬点击能翻转该点不代表周围区域已学会修正。新参数不是论文推荐最优值，必须在val选择；恢复训练时沿用checkpoint配置。当前CLI仍只模拟点，支持框输入不代表已经训练过框；官方预训练后端是当前框实验入口。

真实反馈的 `feedback.npz` 只用 `labels[known]` 监督；`corrected_logits_for_review_only` 永远不当真值。加载器拒绝未确认反馈以及 `validation/test/unassigned` 病例。训练入口还必须提供 `--splits-file --fold`：在第一次优化之前，将**全部**模拟病例和反馈病例的 `case_id` 对照原始 fold 的 train 名单校验，并检查所有模拟 NPZ 内嵌身份字段；自报 `split=train` 不足以通过。Checkpoint 保存 splits 文件 SHA256、fold 与已核查病例名单。若决定把原测试病例加入训练，应重新定义数据划分并另设未见测试病例，不能继续计入原 test100 的泛化评价。

本实现没有 GUI，也没有完成图像与 LLM 的语义特征对齐。可选语言控制器只输出受校验的动作，坐标与区域均来自真实注册的会话对象。

## 核查范围

`scripts/brats2021_audit_v2/interactive/test_core.py` 检查：

- 真正的 PlainConvUNet 前向、反向、参数更新与 checkpoint 重载；
- 无空间提示时增量恒为零；
- ET 负点不会变成背景监督；明确 NCR 替换监督三个区域头；
- ROI 外逐元素不变、撤回精确复原；
- 无效动作/非有限后端输出不提交病例状态；
- 相互矛盾的区域约束拒绝执行；
- affine 坐标变换与毫米尺度 Gaussian；
- 未确认区域不产生训练梯度；
- 测试病例反馈和仅框提示不能作为稠密训练真值。
