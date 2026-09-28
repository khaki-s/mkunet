# BraTS2021 500 例影像特征与 3D nnU-Net 优化代码计划

## 结论与适用范围

优先级建议是：**① NCR 的困难病例与互斥类别监督；② 按类别利用四种 MRI 模态；③ 在高分辨率层保留局部纹理与边界信息。** 依据是原始标签 1 的小体积和缺失病例、现有模型的 NCR 低分尾部、病例内模态对比，以及 ET 与 ED 的局部高频差异。下面的论文只选 **CVPR 2026、ICCV 2025、NeurIPS 2025 正式主会论文和 2025–2026 年 IEEE TMI 论文**；跨数据集论文仅用作可检验的方法线索，不能把其报告的收益当作本数据集的预期收益。

本报告直接扫描 `/root/autodl-tmp/data/BraTS2021_Training_Data-500` 的 **500 例、2500 个 NIfTI 文件**。500 例分割和四模态在最终扫描中全部读通；形状都是 `240×240×155`，间距均为 `1×1×1 mm`。原始 `BraTS2021_00413` 四模态曾在首次读取时尚未写完，随后文件增长并通过 `gzip -t`，全量统计读取的是完整版本。训练前仍须重新执行完整性检查，避免对正在写入的文件预处理。逐病例结果见 [`per_case.csv`](per_case.csv)，汇总见 [`dataset_summary.json`](dataset_summary.json)、[`label_summary_500.json`](label_summary_500.json)、[`paired_contrasts.json`](paired_contrasts.json)。

现有 baseline 是 `Dataset137_BraTS2021/nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres` 的 **fold 0**：预处理数据 400 例，固定划分 320 例训练、80 例验证，另有 100 例原始数据未进入这个预处理数据集。该 100 例可作为最终内部留出集，但不是官方外部测试。计划首先固定原 320/80 划分做公平消融，选定方案前不使用留出 100 例调参。baseline 使用四通道 T1/T1ce/T2/FLAIR、`128×160×112` patch、batch 2、前景过采样 0.33、脑区掩码 Z-score、区域 Dice+BCE，训练 200 epoch；已经包含 GaussianBlur 和模拟低分辨率增强。因此不应直接把全局加锐或更强模糊增强当作改进。

**标签必须先对齐。** 原始互斥标注：`0=其他（含脑外零值）`，`1=NCR/NET（坏死及非增强肿瘤核心，不能仅凭 MRI 断定组织学坏死）`，`2=ED`，`4=ET`。当前 nnU-Net 转换后：`0=背景，1=ED，2=NCR，3=ET`；区域目标为 `WT={ED,NCR,ET}`、`TC={NCR,ET}`、`ET={ET}`。这已经通过同一病例的原始与预处理掩码逐体素核实。后文的独立 ED/NCR/ET Dice 是从保存的预测 NIfTI 重算的，不能由 WT/TC/ET Dice 直接推断。

## 数据本身：频数、灰度、局部频率和边界

500 例的**完整视野**共 44.64 亿体素。标签 0 含大量颅外零值，不适合直接作为“正常脑组织”灰度参考；灰度分析只取四模态非零并集内的标签 0。病例级脑区中位大小为 1,503,313 体素，脑区内标签 0 的占比中位数为 **94.05%**。

| 原始标签 | 全数据体素数 | 全视野比例 | 病例级体积中位数 [Q1, Q3]，mm³ | 空标签病例 |
|---|---:|---:|---:|---:|
| 0，其他 | 4,416,226,639 | 98.930% | 8,839,046 [8,791,011, 8,877,788] | 0 |
| 1，NCR/NET | 5,913,143 | 0.132% | 6,830 [1,635, 16,762] | **13/500** |
| 2，ED | 31,138,976 | 0.698% | 54,882 [27,299, 87,338] | 0 |
| 4，ET | 10,721,242 | 0.240% | 18,828 [8,965, 31,023] | 0 |

WT 体积的病例级中位数是 88,955 mm³；其中 ED、ET、NCR 的占比中位数分别为 **66.9%、22.0%、8.1%**。nnU-Net 的 patch 有 2,293,760 体素；即使 patch 包住整个中位 NCR，NCR 也只占约 0.30% patch。这个比值只是尺度比较，并非实际采样 patch 的占比。由于所有体素都是 1 mm³，这里的体素数也等于 mm³ 数值。

MRI 灰度没有跨患者统一的绝对物理标尺。每例在非零脑区计算 `z=(I−脑区均值)/脑区标准差`，再求各类体素的中位 z，最后对病例取中位数。T1ce 的脑区原始灰度标准差在病例间 Q1–Q3 为 **126–893**，支持先做病例内标准化。下表是描述性统计，不是组织诊断阈值：

| 原始标签 | T1 z | T1ce z | T2 z | FLAIR z | 可见特征 |
|---|---:|---:|---:|---:|---|
| 0，脑区内其他 | +0.127 | −0.016 | −0.345 | −0.061 | 混合了多种正常和非肿瘤组织，不能视作单一灰度类 |
| 1，NCR/NET | −0.577 | −0.518 | +1.423 | +2.155 | 常见 T1/T1ce 偏低、T2/FLAIR 偏高，但表型变化大 |
| 2，ED | +0.091 | −0.251 | +0.593 | +1.851 | 大范围 FLAIR 高信号，局部纹理相对平缓 |
| 4，ET | −0.122 | **+1.843** | +0.682 | +2.002 | T1ce 最有辨识力，内部强度与轮廓并不均一 |

病例内配对比跨病例的原始灰度更可靠：T1ce 的 `ET−NCR` 中位差 **+2.347 z**（487 个有 NCR 病例中 99.0% 为正）；FLAIR 的 `ED−脑区标签0` 为 **+1.911 z**（500/500 为正）；T2 的 `NCR−ED` 为 **+0.799 z**（87.9% 为正）。相比之下 FLAIR 的 `NCR−ED` 仅 **+0.225 z**，只有 **59.3%** 病例为正。因此 FLAIR 很适合定位广泛水肿，却不足以稳定地区分 NCR 与 ED；NCR 需要联合 T1/T2/T1ce 的信息，ET 则要重视 T1ce。

为量化图像空间的局部高频，审计脚本每例每类固定随机种子采样至多 12,000 个体素，在**同类 6 邻域内部**计算 `|z_i−六邻域均值|` 的中位数。边界对比是跨标签 6 邻域的 `|z_i−z_j|` 中位数；它混合了不同邻接类别。二者都是纹理/边界可见度的**代理量**，不是 MRI 原始 k-space 频谱、点扩散函数、真实模糊半径，也不能单独代表标注置信度。

| 类别 | T1ce 局部高频 | FLAIR 局部高频 | T1ce 跨类边界对比 | FLAIR 跨类边界对比 |
|---|---:|---:|---:|---:|
| NCR | 0.097 | 0.085 | 0.892 | 0.258 |
| ED | **0.057** | **0.066** | 0.203 | 0.328 |
| ET | **0.228** | 0.093 | 0.735 | 0.254 |

T1ce 的 ET 局部高频比 ED 高，病例内差值中位数为 **+0.167 z**，499 个可计算病例中 **99.8%** 为正。ED 的 T1ce 局部高频与边界对比都低，FLAIR 边界对比相对更高。由此可检验“在 ET/NCR 的高分辨率特征中保留细节、在 ED 保留低频连通范围”的设计；不能据此断言 ED 的采集本身更模糊。原始 NIfTI 没有 raw k-space 或重复扫描，无法从这批文件直接测得采集噪声谱、PSF 或真实标注不确定性。

## baseline 揭示的实际错误

[`baseline_summary.json`](baseline_summary.json) 的官方区域验证摘要覆盖 fold 0 的 80 例；[`exclusive_summary.json`](exclusive_summary.json) 从预测与 GT 重算互斥类别。下列值都是 **80 个病例 Dice 的算术平均**：

| 指标 | 平均 Dice | 病例级中位 Dice | 解释 |
|---|---:|---:|---|
| WT 区域 | 0.947 | 0.960 | NCR+ED+ET 的并集 |
| TC 区域 | 0.936 | 0.973 | NCR+ET 的并集 |
| ET 区域/互斥 ET | 0.909 | 0.938 | 单独 ET |
| 互斥 ED | 0.887 | 0.920 | WT 准确并不保证 ED 准确 |
| **互斥 NCR** | **0.819** | 0.935 | 少量严重失败拉低均值；非空 78 例平均为 0.840 |

NCR 有 **10/80** 例 Dice <0.5，其中 2 例 GT 没有 NCR 却预测出 NCR；ET 仅 2/80、ED 3/80 低于 0.5。8 个非空低分 NCR 中多数体积很小，但 `BraTS2021_00134` 的 NCR 有 **97,830 mm³，Dice 只有 0.009**，所以不能把所有失败归咎于小病灶。按总混淆体素计，真 NCR 被预测为 ED 约 11.6%，预测为 ET 约 7.9%，值得分别追踪。

[`confidence_summary.json`](confidence_summary.json) 从 80 例存储的 WT/TC/ET sigmoid 概率读取，每轴每隔一体素抽样一次，在肿瘤包围盒外扩 16 mm 范围内计算。ET 假阳性体素的**病例级概率中位数再取中位数为 0.958**，假阴性为 0.034；模型会很自信地犯错。ET 真值边缘 `|p−0.5|<0.15` 的体素占比病例级中位数仅约 1.54%，内部接近 0。因此这些概率是**模型置信度**，不能称为标注置信度，也不能仅用低概率/高熵来筛困难 patch；需结合真实训练误差、类别体积和标注边界。

## 方向一：NCR 困难病例、稀有类别与可信监督（优先级 1）

目标是修复互斥 NCR 的低分尾部，同时限制空 NCR 病例的假阳性。先做轻量采样与损失消融，再考虑专门的 ROI 结构。五篇正式论文如下：

| 论文 | 正式出处 | 对本任务的可用线索与限制 |
|---|---|---|
| [PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_PGR-Net_Prior-Guided_ROI_Reasoning_Network_for_Brain_Tumor_MRI_Segmentation_CVPR_2026_paper.html) | CVPR 2026 | 直接针对脑肿瘤稀疏 ROI；先借鉴训练 patch 的 ROI 选择，不立即替换整套 backbone。 |
| [Delving Aleatoric Uncertainty in Medical Image Segmentation via Vision Foundation Models](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Delving_Aleatoric_Uncertainty_in_Medical_Image_Segmentation_via_Vision_Foundation_CVPR_2026_paper.html) | CVPR 2026 | 类别相关的困难度/损失权重；本数据只有单套标签，不能把模型错误直接定义成 aleatoric 不确定性。 |
| [Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance](https://openaccess.thecvf.com/content/CVPR2026/html/Guarino_Better_than_Average_Spatially-Aware_Aggregation_of_Segmentation_Uncertainty_Improves_Downstream_CVPR_2026_paper.html) | CVPR 2026 | 按 patch、类别、边界汇总不确定性用于失败预警；论文重点是检测失败，不能当成已证明的训练损失。 |
| [SPA: Efficient User-Preference Alignment against Uncertainty in Medical Image Segmentation](https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_SPA_Efficient_User-Preference_Alignment_against_Uncertainty_in_Medical_Image_Segmentation_ICCV_2025_paper.html) | ICCV 2025 | 提醒边界和标注偏好可能含歧义；本数据无多阅片者标签，先做病例审查，不凭单个掩码宣称真实标注置信度。 |
| [Exploring the Connection between Uncertainty and Tissue Boundaries in Medical Image Segmentation](https://doi.org/10.1109/TMI.2026.3720501) | TMI 2026，已正式在线发表，Early Access | 不确定性引导边界损失；所用数据并非 BraTS，应只做可撤销的小权重边界损失消融。 |

**首个代码实验 H1**：沿用 WT/TC/ET 区域 head、200 epoch 和同一 320/80 划分。采样器保留 50% 原 nnU-Net 采样，其余从训练集做 `NCR 25% / ED 12.5% / ET 12.5%` 的中心采样；NCR 空标签病例单列为假阳性控制样本。先只改采样，之后再加损失。损失保持原区域 Dice+BCE，另从区域概率计算 `p_ED≈p_WT−p_TC`、`p_NCR≈p_TC−p_ET` 的非负部分，增加小权重互斥类 Dice 和 `p_ET≤p_TC≤p_WT` 违例惩罚；写单元测试核对原始 `1/2/4` 与内部 `2/1/3` 的映射及深监督尺度。困难挖掘从第 20 epoch 后依据**训练 fold 内**的在线高损失/高置信误判更新，不能用验证 80 例或留出 100 例的预测来回灌训练。

**判定**：优先看 NCR 病例平均、10% 分位 Dice、非空小体积组和无 NCR 病例的 FP mm³；同时约束 ED/ET/WT 不下降。单纯提高区域平均 Dice 但 NCR 低分尾部不改善，不算这个方向成功。

## 方向二：按类别和局部质量使用四模态（优先级 2）

物理依据是 T1ce 中 ET 与 NCR 差异几乎处处为正，FLAIR 明显分出 ED 与脑区其他组织，而 NCR 与 ED 在 FLAIR 上重叠较多。四模态最终均齐全，因此论文中的“缺失模态”是鲁棒融合方法参考，**不是当前数据的主要缺失问题**。

| 论文 | 正式出处 | 对本任务的可用线索与限制 |
|---|---|---|
| [REDNet: Reliable Evidential Discounting Network for Multi-Modality Medical Image Segmentation](https://ieeexplore.ieee.org/document/11087642/) | TMI 2026，45(1) | 包含 BraTS2021；低质量模态降权和跨模态差异聚合可作门控设计依据。 |
| [Does Adding a Modality Really Make a Positive Impacts in Incomplete Multi-modal Brain Tumor Segmentation?](https://ieeexplore.ieee.org/document/10830281/) | TMI 2025，44(5) | PNDC 分析模态可能带来负贡献；必须先做 leave-one-modality-out 诊断，不能假设四通道总是更好。 |
| [SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality](https://openaccess.thecvf.com/content/ICCV2025/html/Li_SimMLM_A_Simple_Framework_for_Multi-modal_Learning_with_Missing_Modality_ICCV_2025_paper.html) | ICCV 2025 | BraTS2018 上的动态模态专家门控；先借鉴轻量门控而非完整专家体系。 |
| [Uni-Encoder Meets Multi-Encoders: Representation Before Fusion for Brain Tumor Segmentation with Missing Modalities](https://openaccess.thecvf.com/content/CVPR2026/html/Song_Uni-Encoder_Meets_Multi-Encoders_Representation_Before_Fusion_for_Brain_Tumor_Segmentation_CVPR_2026_paper.html) | CVPR 2026 | 单编码器全局表征配合模态专属高分辨率特征；本计划保留 nnU-Net 主干，只试浅层模态分支。 |
| [Virtual Nodes Guided Dynamic Graph Neural Network for Brain Tumor Segmentation with Missing Modalities](https://openaccess.thecvf.com/content/CVPR2026/html/Tao_Virtual_Nodes_Guided_Dynamic_Graph_Neural_Network_for_Brain_Tumor_CVPR_2026_paper.html) | CVPR 2026 | 动态模态连接抑制干扰；图网络较重，仅在简单门控有效后考虑。 |

**首个代码实验 M1**：先在 baseline 的验证 80 例上做四次“去掉一个输入通道”的**只读诊断**，按 NCR/ED/ET 和病灶大小记录 Dice 变化；该诊断不作为新模型训练数据。随后在四通道 Z-score 输入后加一个零额外通道的有界空间门控 `x'_m=x_m·g_m(x)`，`g_m∈[0.5,1.5]` 且初始化为 1，使原 checkpoint 首层卷积可直接加载。只在门控有益时加类别特定的浅层残差分支。分别消融：门控、轻度模态 dropout、二者组合。T1ce 与 T1 的强度差只能作为**标准化图像特征**测试，不能解释为定量钆浓度；验证时要看 NCR↔ED、NCR↔ET 的混淆是否下降。

## 方向三：保留局部高频与连续边界（优先级 3）

ET 的 T1ce 内部高频明显高于 ED，而 ED 范围大、较平滑，外缘主要靠 FLAIR。当前数据各向同性，复杂的各向异性重采样并不是首要问题；高频增强也可能放大噪声和标签边界偏差。

| 论文 | 正式出处 | 对本任务的可用线索与限制 |
|---|---|---|
| [Rethinking Brain Tumor Segmentation From the Frequency Domain Perspective](https://ieeexplore.ieee.org/document/11032150/) | TMI 2025，44(11) | 直接针对脑肿瘤与 ET，低/高频分解和自适应 Laplacian 可指导轻量特征分支。 |
| [CROWn: A Unified Framework for Anti-Aliased Downsampling and Phase-Calibrated Fusion in 3D Medical Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_CROWn_A_Unified_Framework_for_Anti-Aliased_Downsampling_and_Phase-Calibrated_Fusion_CVPR_2026_paper.html) | CVPR 2026 | 3D 下采样抗混叠与 skip 相位校准；论文强调各向异性，本数据先只试简化的抗混叠。 |
| [PMRNet: Physics-informed Multi-scale Refinement Network for Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Kang_PMRNet_Physics-informed_Multi-scale_Refinement_Network_for_Medical_Image_Segmentation_CVPR_2026_paper.html) | CVPR 2026 | 多尺度与边界解码的思路；不把论文术语中的“physics-informed”误当成已测得本数据 MRI 物理参数。 |
| [M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_M-Net_MRI_Brain_Tumor_Sequential_Segmentation_Network_via_Mesh-Cast_ICCV_2025_paper.html) | ICCV 2025 | 跨切片连续性；现有 3D nnU-Net 已有 3D 上下文，只在切片断裂错误确实突出时追加该机制。 |
| [Neighborhood Self-Dissimilarity Attention for Medical Image Segmentation](https://proceedings.neurips.cc/paper_files/paper/2025/hash/36b80eae70ff629d667f210e13497edf-Abstract-Conference.html) | NeurIPS 2025 主会 | 低参数量的局部差异注意力；可用于 ET/NCR 高差异区域的跳连门控，需验证其是否误放大噪声。 |

**首个代码实验 F1**：在最高分辨率的两个 decoder skip 上，对原特征做固定、可微的 3D 低通 `L` 与残差 `H=F−L`，以初始化近零的可学习 gate 输出 `F'=F+α·gate(F)·H`。先只试这一个小模块，再单独试下采样前固定抗混叠，最后才试二者组合。避免在全 patch 上做大规模 FFT 或全局锐化。边界损失单独消融，在训练掩码的 2 mm shell 内加小权重；分别查看 ED 外缘 HD95/Surface Dice、ET 小病灶边界和假阳性。现有 GaussianBlur/低分辨率增强保持不变作为对照。

## 可执行代码路线与实验约束

新实现放在独立命名空间，避免覆盖当前 `/root/nnUNet` 中已有未提交修改。建议新增文件如下；路径是**待实现目标**，此报告没有替用户启动训练或修改现有 trainer：

```text
/root/nnUNet/nnunetv2/training/nnUNetTrainer/variants/brats2021_audit/
  nnUNetTrainer_BraTSHard200.py       # H1：类条件采样、可选互斥类别损失
  nnUNetTrainer_BraTSModGate200.py    # M1：轻量模态门控
  nnUNetTrainer_BraTSFreqSkip200.py   # F1：高/低频 skip 分支
  nnUNetTrainer_BraTSCombined200.py   # 仅在单项均有效后组合
/root/nnUNet/nnunetv2/utilities/brats2021_audit/
  label_mapping.py                    # raw 0/1/2/4 ↔ internal 0/2/1/3；WT/TC/ET
  class_sampler.py                     # 只访问训练 fold 的索引和标签
  modality_gate.py                     # 保持 4 通道输入和 checkpoint 兼容
  frequency_skip.py                    # 3D 低通、残差、门控
  exclusive_loss.py                    # 区域概率的层级/互斥约束
/root/nnUNet/scripts/brats2021_audit/
  qc_nifti.py                          # 500×5 文件完整性、形状、标签、仿射检查
  evaluate_exclusive.py                # WT/TC/ET + 独立 ED/NCR/ET + 校准
```

下面是**类实现后**的训练调用示例；每个新 trainer 名称对应单独结果目录，当前没有执行这些训练：

```bash
export nnUNet_raw=/root/autodl-tmp/data/nnUNet_raw
export nnUNet_preprocessed=/root/autodl-tmp/data/nnUNet_preprocessed
export nnUNet_results=/root/autodl-tmp/data/nnUNet_results
nnUNetv2_train 137 3d_fullres 0 -tr nnUNetTrainer_BraTSHard200
```

1. **锁定数据与基线**：把 320/80 fold-0 病例 ID 和额外 100 例 ID 写成不可变 manifest；读取 NIfTI 的每个模态、检查 gzip、shape/affine、`{0,1,2,4}`，记录文件大小/哈希。`00413` 已在审计中从临时截断恢复，QC 必须以训练当时的文件为准。若重新从 raw 预处理，先核实当前文件与原 400 例的 `nnUNet_raw`/预处理记录一致。保存 baseline 的独立类混淆矩阵和 80 例逐病例结果。
2. **最小单项消融**：E0 原 trainer；H1 只采样；H2 采样+互斥类/层级损失；M1 只模态门控；F1 只高频 skip；F2 只抗混叠。所有实验使用同一 320/80 划分、200 epoch、同一数据增强与推理后处理；记录随机种子、训练时间、峰值显存和保存的逐例预测。新网络若从 checkpoint 微调，应把“同 checkpoint、同微调预算”作为独立比较，不能和从头 200 epoch 的结果混算。
3. **必要测试**：合成 3D 掩码验证原始/内部标签映射、WT/TC/ET 关系和空 NCR；随机 patch 验证采样比例与不泄漏验证 ID；一次前向、反向与完整体积滑窗推理检查维度、梯度、显存；核对从区域概率还原互斥类时的非负性和层级一致性。
4. **评价与决策**：主指标是互斥 NCR/ED/ET 病例均值、10% 分位与病灶体积分层；同时报告 WT/TC/ET、HD95/Surface Dice、无 NCR 病例 FP mm³、病灶级检出、T1ce/FLAIR 关键混淆以及 ROI 内 Brier/ECE。每个候选与 E0 做 80 例**配对**差值及病例 bootstrap 区间，单看全脑 accuracy 或平均区域 Dice 不足以定论。对 NCR 的目标先定为均值提升约 2 个百分点且低分尾部改善，同时 ED/ET/WT 不出现实质退化；这是实验门槛，不是承诺收益。
5. **最终确认**：仅把有效单项组合成 H+M、H+F、H+M+F，先在 fold 0 确认无相互抵消，再复核其他 folds。模型与超参数确定后，才对额外 100 例做一次内部留出评估；若之后用 100 例追加训练，必须另设未使用的测试集，不能再把它当独立评估。

复现实验的统计脚本已保存在本目录：[`audit_labels.py`](audit_labels.py)、[`audit_dataset.py`](audit_dataset.py)、[`audit_pairs.py`](audit_pairs.py)、[`audit_baseline.py`](audit_baseline.py)、[`audit_exclusive.py`](audit_exclusive.py)、[`audit_confidence.py`](audit_confidence.py)。`audit_dataset.py` 的高频量是描述性代理量，网络中的频率模块需通过上述消融单独证明有效。

## 论文来源核验

下面恰好 15 篇，每方向五篇。CVPR/ICCV 链接指向 CVF 正式论文页，NeurIPS 指向正式主会论文页，TMI 指向 IEEE 文章或 DOI；EUGB 为 2026 年正式在线发表的 TMI Early Access。上表已经给出每篇与本任务的适用范围，这些论文不提供本数据集上的直接增益保证。

1. Lu 等，[PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_PGR-Net_Prior-Guided_ROI_Reasoning_Network_for_Brain_Tumor_MRI_Segmentation_CVPR_2026_paper.html)，CVPR 2026，pp. 22816–22825。
2. Li 等，[Delving Aleatoric Uncertainty in Medical Image Segmentation via Vision Foundation Models](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Delving_Aleatoric_Uncertainty_in_Medical_Image_Segmentation_via_Vision_Foundation_CVPR_2026_paper.html)，CVPR 2026，pp. 30011–30020。
3. Guarino 等，[Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance](https://openaccess.thecvf.com/content/CVPR2026/html/Guarino_Better_than_Average_Spatially-Aware_Aggregation_of_Segmentation_Uncertainty_Improves_Downstream_CVPR_2026_paper.html)，CVPR 2026，pp. 13145–13156。
4. Zhu 等，[SPA: Efficient User-Preference Alignment against Uncertainty in Medical Image Segmentation](https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_SPA_Efficient_User-Preference_Alignment_against_Uncertainty_in_Medical_Image_Segmentation_ICCV_2025_paper.html)，ICCV 2025，pp. 23731–23740。
5. Zeng 等，[Exploring the Connection between Uncertainty and Tissue Boundaries in Medical Image Segmentation](https://doi.org/10.1109/TMI.2026.3720501)，IEEE TMI，2026-08-06 在线发表，DOI: 10.1109/TMI.2026.3720501。
6. [REDNet: Reliable Evidential Discounting Network for Multi-Modality Medical Image Segmentation](https://ieeexplore.ieee.org/document/11087642/)，IEEE TMI 45(1)，2026，pp. 16–27，DOI: 10.1109/TMI.2025.3591124。
7. Qiu 等，[Does Adding a Modality Really Make a Positive Impacts in Incomplete Multi-modal Brain Tumor Segmentation?](https://ieeexplore.ieee.org/document/10830281/)，IEEE TMI 44(5)，2025，pp. 2194–2205，DOI: 10.1109/TMI.2025.3526818。
8. Li 等，[SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality](https://openaccess.thecvf.com/content/ICCV2025/html/Li_SimMLM_A_Simple_Framework_for_Multi-modal_Learning_with_Missing_Modality_ICCV_2025_paper.html)，ICCV 2025，pp. 24068–24077。
9. Song 等，[Uni-Encoder Meets Multi-Encoders: Representation Before Fusion for Brain Tumor Segmentation with Missing Modalities](https://openaccess.thecvf.com/content/CVPR2026/html/Song_Uni-Encoder_Meets_Multi-Encoders_Representation_Before_Fusion_for_Brain_Tumor_Segmentation_CVPR_2026_paper.html)，CVPR 2026，pp. 15627–15638。
10. Tao 等，[Virtual Nodes Guided Dynamic Graph Neural Network for Brain Tumor Segmentation with Missing Modalities](https://openaccess.thecvf.com/content/CVPR2026/html/Tao_Virtual_Nodes_Guided_Dynamic_Graph_Neural_Network_for_Brain_Tumor_CVPR_2026_paper.html)，CVPR 2026，pp. 37528–37537。
11. Shao 等，[Rethinking Brain Tumor Segmentation From the Frequency Domain Perspective](https://ieeexplore.ieee.org/document/11032150/)，IEEE TMI 44(11)，2025，pp. 4536–4553，DOI: 10.1109/TMI.2025.3579213。
12. Huang 等，[CROWn: A Unified Framework for Anti-Aliased Downsampling and Phase-Calibrated Fusion in 3D Medical Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_CROWn_A_Unified_Framework_for_Anti-Aliased_Downsampling_and_Phase-Calibrated_Fusion_CVPR_2026_paper.html)，CVPR 2026，pp. 8514–8524。
13. Kang，[PMRNet: Physics-informed Multi-scale Refinement Network for Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Kang_PMRNet_Physics-informed_Multi-scale_Refinement_Network_for_Medical_Image_Segmentation_CVPR_2026_paper.html)，CVPR 2026，pp. 15659–15668。
14. Lu 等，[M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_M-Net_MRI_Brain_Tumor_Sequential_Segmentation_Network_via_Mesh-Cast_ICCV_2025_paper.html)，ICCV 2025，pp. 20116–20125。
15. Chen 等，[Neighborhood Self-Dissimilarity Attention for Medical Image Segmentation](https://proceedings.neurips.cc/paper_files/paper/2025/hash/36b80eae70ff629d667f210e13497edf-Abstract-Conference.html)，NeurIPS 2025 主会，DOI: 10.52202/085713-1278。
