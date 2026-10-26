# 五专家特征注入：实现、来源与边界

2026-10-20。此目录为新增架构，不替换已训练的四专家或五专家类。

## 实际复用哪些类

* **`torch.nn.MultiheadAttention`**：直接实例化官方 PyTorch 类，`embed_dim=48, num_heads=3, dropout=0, batch_first=True`。Q/K/V投影、缩放点积、mask、softmax、多头合并均由它实现，没有手写一套注意力算子。[官方接口](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)。
* **`dynamic_network_architectures.building_blocks.unet_decoder.UNetDecoder`**：环境版本0.4.4。新 `ExpertInjectedUNetDecoder` 继承它并复用已经初始化的 `encoder/stages/transpconvs/seg_layers`，只在原forward循环中增加一次注入。保留原参数名、deep supervision顺序、单遍解码。原库Apache-2.0许可见同目录 `LICENSE.dynamic_network_architectures`。[官方源文件](https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/building_blocks/unet_decoder.py)。
* **现有GeneralistLegacy实现**：完整继承五专家损失、模态采样、300轮、phase0/50/100、最后logit融合。当前已经训练的类不变。

## 顶会方法如何映射到这次实现

| 来源 | 已核对的作者实现 | 采用的部分 | 没有照搬的部分 |
|---|---|---|---|
| [Controllable-LPMoE，ICCV2025](https://github.com/CSYSI/Controllable-LPMoE/tree/02c73635fc640ff50d6699ee6d11e7ae0a0e3ebe) | `ICCV_25_LPMoE/mmseg_custom/models/backbones/LPMoE_modules.py::Module2_1`，query/feat归一化、`query + gamma * attn`；`Module2`双向交互 | 特征交互的接收者是主干；残差注入；从恒等映射开始 | 作者2D MSDeformAttn及双向交互；本版使用标准MHA、同位置3D专家轴、单向专家→joint decoder |
| [MoME，CVPR2025](https://github.com/konyul/MoME/tree/ddbec03ba05b154e72a2ea8e3c83ddafc1813466) | `projects/mmdet3d_plugin/models/utils/multi_expert.py` | 接收者query决定采用哪些专家；缺失输入显式屏蔽 | 检测object query、相机/激光空间投影、作者模态类别CE目标；本版是已配准MRI的低分辨率同位置query |

以上论文已有各自任务的实验依据；**本项目的3D组合与注入位置仍需自己的消融验证，不能称为作者论文复现，也不能保证Dice提高**。没有复制两篇论文的loss或未经适配的2D CUDA算子。保持现有loss，是为了只测“是否把专家证据送入主干特征”这一个变化。

## 数学与数据流

设joint decoder第一个约1/8分辨率状态为 $h_J(v)$，专家相同层的状态为 $h_m(v)$。投影与LayerNorm后形成一个query和最多五个key/value：

$$
q_r(v)=W^Q_r\,LN(P_J h_J(v)),\quad
k_{r,m}(v)=W^K_r\,LN(P_m h_m(v)),\quad
u_{r,m}(v)=W^V_r\,LN(P_m h_m(v)).
$$

$$
A_{r,m}(v)=\operatorname{softmax}_{m\in\mathcal A}
\frac{q_r(v)^\top k_{r,m}(v)}{\sqrt{16}},\quad
h'_J(v)=h_J(v)+W_O\,\operatorname{MHA}(q,\{k_m,u_m\}).
$$

这里 $W_O$ 表示MHA输出后新增的1×1×1卷积；MHA还具有官方类内部的输出投影。三个head是**潜在特征head，并未绑定WT/TC/ET**。可用集合 $\mathcal A$ 是存在的单模态专家加Generalist。Generalist只读取已掩膜四通道和四个可用性通道。

* attention只在同一空间位置的5个专家上归一化，不做全体素两两attention，主复杂度约 $O(VMd)$。
* 独立专家投影 $P_m$ 解决不同网络通道基底不一致的问题；通道归一化减少纯尺度差异影响。
* 当前模态缺失时，仍执行Generalist；缺失单模态专家不会执行，注意力严格屏蔽。
* 专家状态先产生；joint decoder从bottleneck向高分辨率只运行一遍。专家特征不直接cat到joint通道。原nnU-Net自己的encoder skip concat保持原有结构。
* 注入从 `epoch=50`（从1计数第51轮）开始；前50轮专家独立监督和joint预热不变。Comm从 `epoch=100`（第101轮）开始，Local对照消息仍为零。
* 新输出卷积权重零初始化、残差系数固定为1：初始函数等于原模型，首个更新能训练输出投影，随后梯度能流入attention和专家状态。没有同时初始化成零的可训练乘子。
* 最终仍用原公式 `z_final=(1-beta)*z_joint+beta*sum_m(w_m*z_expert_m)`。新空间attention $A(v)$ 与旧全局输出权重 $w$ 是两条不同路径，不能混称为同一门控。
* 注入后joint高分辨率及相关DS预测都会接收专家特征；DS权重、loss定义不变。原 `generalist_interventions` 中 `joint_only` 在新类里是**含专家注入的joint分支**，不能解释为纯UNet。要切断新路径，用 `disable_feature_injection=True`。

## 类与checkpoint

```bash
nnUNetv2_train 137 3d_fullres 0 -tr nnUNetTrainer_BraTSAgentGeneralistFeatureLocal300
nnUNetv2_train 137 3d_fullres 0 -tr nnUNetTrainer_BraTSAgentGeneralistFeatureComm300
```

两个类除了原Local/Comm开关，其余参数、初始化、训练安排相同。**Local指专家之间的消息为零，不表示主干只读一个专家**；两个新类的joint decoder都会读取当前可用的专家特征。先保持seed0、300轮、原plans、同一split；用户自行启动训练。

新network的旧参数路径不变，新增 `feature_injection.*` 与 `feature_injection_active`，`architecture_version=4`。同seed的旧参数逐张量相等且新残差初始为零。**旧五专家checkpoint不能作为新类的 `--c` checkpoint**；新类从头训练可保持预算一致。如果以后做预训练迁移，必须单独记录权重迁移和额外训练预算，不能把 `strict=False` 当作公平从头训练。

旧五专家所有类别与文件均保留。已有实现对整个V2目录的源码做合同指纹，新增代码会影响旧类继续训练时的指纹；这不影响旧checkpoint常规推理。旧类若要续训，应使用它们原训练的源码快照或显式审计迁移，不能静默跳过合同验证。

## 诊断与导出

训练的 `AGENT_STEP.feature_injection` 记录空间attention的均值、空间标准差、注入更新/主干特征RMS比例和模块梯度。原路由日志保持B×3×5。

`return_aux=True`返回 `feature_attention`，维度为B×3个潜在head×5专家×D×H×W（注入特征网格），以及原 `weights`。可用：

```python
from nnunetv2.utilities.brats2021_audit_v2.agent_v2.feature_injection.diagnostics import export_feature_attention
parts = network(image, return_aux=True, modality_mask=physical_mask)
export_feature_attention(parts, "feature_attention.npz",
    patch_metadata={"space": "preprocessed_DHW", "window_origin": [0, 0, 0],
                    "mirror_axes_applied": [], "source_case": "case_id"})
```

这里必须传入当前窗口真实坐标。此导出不会自动逆转TTA、拼接滑窗或映射到native MRI。空间平均attention仅说明选择比例，不能当作因果贡献；禁止把它塞回旧输出路由CSV当WT/TC/ET权重。

## 验证范围

`python -m nnunetv2.utilities.brats2021_audit_v2.agent_v2.feature_injection.checks --output /tmp/feature_checks.json`

覆盖CPU小plans：Local/Comm全部15mask×phase0/50/100初始等价；mask权重及不执行缺失专家；一次joint解码；隐藏输入不变性；真实train/validation step及epoch-end collator；checkpoint恢复及下一步位级一致；只以joint loss反传时的新专家→joint梯度；学习后切断新路径可改变joint输出。阶段边界检查仅各运行一步，logger缺失历史用fixture补齐，不能视为完成100轮训练。

`python -m nnunetv2.utilities.brats2021_audit_v2.agent_v2.feature_injection.runtime_checks --output /tmp/feature_runtime.json` 另外覆盖：实际checkpoint发现/严格加载、deep_supervision=False、仅T1ce单文件、滑窗/TTA、NIfTI导出与affine一致、五专家路由汇总以及空间attention的NPZ导出。只使用合成文件，不读取病例MRI。

本检查不能代替真实plans的GPU显存、AMP、吞吐及完整patient滑窗评估。尚未启动任何正式训练，尚无新架构Dice改善结论。

当前真实plans为128×160×112、batch=2，注入层joint通道256、专家通道64，新增模块共49,824个参数。独立统计得到的参数数目不等于GPU峰值显存测量。
