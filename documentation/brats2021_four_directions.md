# BraTS2021：四方向 3D 实现、数学依据与执行路线

版本：2026-09-13。根据用户确认，将 MoE 纳入模态方向，Mamba 单列第四方向。核心文件为 `nnunetv2/utilities/brats2021_audit/{trainer,blocks,network}.py`，补充索引、评估及检查脚本。所有实验均由用户启动。

## 评估两份 plan 后的选择

建议先做 **3D**。3D plan 基于 500 例影像审计及 fold 0 的 80 例失败分析，直接对应当前 NCR 低分尾部和模态混淆；另一份 plan 的 2.5D 优先级主要针对 2D 缺失体积上下文的情况，不能直接套到当前 3D 主线。继续沿用 PlainConvUNet，实验顺序建议 H（采样/监督）→ M（门控/MoE）→ F（局部频率）→ S（Mamba）。若主要关心结构创新，可以在 H1 跑完后优先 M2，再跑 S1。

两份 plan 的评估队列不同：80 例验证集的区域 WT≈0.947 和另一份文档 100 例结果的 WT≈0.9203 不能直接作增益比较。第二份文档已经使用过 100 例的模型结果来讨论路线，因此这 100 例**不能再称为从未观察过的最终测试集**。后续冻结使用规则，并披露此前的探索；严格无偏的最终确认需要另一个未使用队列。当前 320/80 划分保留；新增索引只读取 320 例训练标签，训练 fold 内有 9 例空 NCR。

现有 FADSR、legacy_fadsr、两份原 plan 和 `11.txt` 均保留。此次新增实现不继承旧 FADSR 的额外损失、特殊优化器或历史训练配置。

## 2026-09-17 更新：Loss 基础上的 MoE 与模块日志

新增 `nnUNetTrainer_BraTSMoENoPriorLoss200` 与 `nnUNetTrainer_BraTSMoEUniformLoss200`，均继承已实验的 ExclusiveLoss；请勿把原不带 Loss 的 MoE trainer 当作这两个对照。下一项优先运行无先验版，有收益再运行均匀融合版。

[GPU 命令、日志字段、采样频率与检查范围](brats2021_moe_loss_and_logs.md)。统一注册文件保留用户已有 GateLoss 类；现有脚本加入 `moe_no_prior_loss` / `moe_uniform_loss` 入口。

## 文件与 trainer 对照

| 实验 | CLI trainer | 变化 |
|---|---|---|
| E0 | `nnUNetTrainer_BraTSAuditBaseline200` | 原 nnU-Net 初始化、encoder、decoder、区域 Dice+BCE |
| H1 | `nnUNetTrainer_BraTSHard200` | 只改类别条件采样 |
| H2 | `nnUNetTrainer_BraTSHardLoss200` | H1 + 互斥类与层级辅助损失 |
| H0L | `nnUNetTrainer_BraTSExclusiveLoss200` | 只加辅助损失，补足采样×损失消融 |
| M1 | `nnUNetTrainer_BraTSModGate200` | 输入有界空间模态门控 |
| M2 | `nnUNetTrainer_BraTSMoE200` | 半分辨率 skip 的五专家软路由，带软 MRI 初始偏置 |
| M2-U | `nnUNetTrainer_BraTSMoEUniform200` | 同样五专家和投影，固定均匀融合，控制专家容量 |
| M2-N | `nnUNetTrainer_BraTSMoENoPrior200` | 同 M2，但 router 无 MRI 初始偏置 |
| F1 | `nnUNetTrainer_BraTSFreqSkip200` | 两个最高分辨率 skip 的局部高频残差 |
| S1 | `nnUNetTrainer_BraTSMamba200` | bottleneck 直接调用官方 SegMamba-V2 MambaLayer |
| MS | `nnUNetTrainer_BraTSMoEMamba200` | MoE+Mamba，单项成立后使用 |

核心结构沿用 legacy_fadsr 的职责分工：

- [trainer.py](../nnunetv2/utilities/brats2021_audit/trainer.py)：200 epoch、合法配置检查、采样入口、损失、实验记录。
- [blocks.py](../nnunetv2/utilities/brats2021_audit/blocks.py)：模态门控、MoE、频率、Mamba 适配、标签与损失数学。
- [network.py](../nnunetv2/utilities/brats2021_audit/network.py)：复用原 encoder/decoder，选定接入点，保持深监督及预测重建接口。
- [trainer 注册文件](../nnunetv2/training/nnUNetTrainer/variants/brats2021_audit/nnUNetTrainer_BraTSAudit200.py)：供 nnUNetv2_train/nnUNetv2_predict 按类名发现。
- [sampling.py](../nnunetv2/utilities/brats2021_audit/sampling.py)：训练标签坐标与划分校验。
- [evaluate.py](../nnunetv2/utilities/brats2021_audit/evaluate.py)：区域/互斥病例 Dice、NCR 体积分层、空类 FP、配对 bootstrap。

## H：稀有 NCR 与互斥监督

标签约定为原始 `0/1/2/4 → 内部 0/2/1/3`，即内部 `1=ED,2=NCR,3=ET`；region 输出顺序 `WT,TC,ET`。预处理分割的 `-1` 是 nnU-Net 脑外非零掩码标记，沿用原增强流程变为背景，不作为前景中心。

H1 的 patch 提议分布为

\[
P=.5P_{stock}+.25P_{NCR}+.125P_{ED}+.125P_{ET}.
\]

每个类别分支先从**训练 fold 中该类非空的病例**均匀抽病例，再从该病例的类别坐标均匀抽中心。坐标取自真正互斥标签，每类每病例最多保留 10,000 个固定随机子样本。不能拿稀疏的 TC、ET 采样坐标做集合差来得到 NCR。原始 50% 分支保持 nnU-Net 的抽病例和过采样规则，空 NCR 病例仍在这一分支出现；未额外提高其抽样概率。

特别注意：baseline 配置 `oversample_foreground_percent=.33`，但 batch 2 的离散规则实际是一张随机 patch、一张前景 patch；“保留 .33 配置”不等于实际仅 33% 前景提议。H1 上述比例指增强前的中心提议，旋转缩放后小 NCR 仍可能消失。

H2 保留 region Dice+BCE。对 `p=sigmoid(z)` 先作上闭包：

\[
\tilde p_E=p_E,\quad \tilde p_T=\max(p_T,p_E),\quad
\tilde p_W=\max(p_W,\tilde p_T).
\]

再得到合法的互斥概率

\[
q=(1-\tilde p_W,\tilde p_W-\tilde p_T,\tilde p_T-\tilde p_E,\tilde p_E).
\]

它非负且和为 1；这是明确选择的层级闭包，并非最小距离投影。相比只 clamp 两个差值，这样能同时得到一致的背景/ED/NCR/ET 概率。训练目标由 region target 计算 `t_ED=y_W-y_T,t_NCR=y_T-y_E,t_ET=y_E`。

\[
L=L_{region}+.1L_{exclusive}+.05\operatorname{mean}
\{[p_T-p_W]_+ + [p_E-p_T]_+\}.
\]

`L_exclusive` 为病例×ED/NCR/ET 平均：非空类用 soft Dice loss，空类用该类预测概率均值，避免空 Dice 的不稳定梯度。三个互斥类权重相等；稀有 NCR 的额外关注首先来自 H1。所有有效深监督尺度使用同一公式，尺度权重沿用原 trainer，最低分辨率权重仍为 0。

推理仍由 nnU-Net 对 WT/TC/ET sigmoid 后按原覆盖顺序导出 1/2/3；辅助闭包只用于损失，没有额外改变推理后处理。H1 是静态类别条件采样，不声称已实现第 20 epoch 后在线困难挖掘，也不把训练误差称为真实标注不确定性。先由 H1/H2 判断是否值得追加更复杂的错误缓存。

## M：MRI 模态选择与 MoE

M1：

\[
x'_m=x_m[1+.5\tanh a_m(x)],\qquad g_m\in[.5,1.5].
\]

router 在半分辨率计算再插值，末层初始化为零，故初始严格等于原四通道输入。只改输入门控，不增加输入通道。

M2 在 E1 skip（`64×80×56`，64 个主干通道）使用 5 个轻量专家：T1、T1ce、T2、FLAIR 各一个，另一个联合四模态专家。每专家宽度为 8；联合专家允许直接比较 T1/T1ce/T2/FLAIR 特征，避免 NCR 只能在独立单模态答案中选一个。

\[
E_m=e_m(x_m),\quad E_J=e_J(x_1,x_2,x_3,x_4),
\quad w_{c,m}(v)=\operatorname{softmax}_{m}(a_{c,m}(x,F)(v)+b_{c,m}),
\]
\[
U_c(v)=\sum_m w_{c,m}(v)E_m(v),\quad
F'=F+\tanh(\alpha)W[U_{ED},U_{NCR},U_{ET}].
\]

三个 query 槽的 MRI 偏置为 `b_ED,FLAIR=log 2`、`b_NCR,joint=log 2`、`b_ET,T1ce=log 2`，其余为 0；对应初始化重点专家 1/3，其余各 1/6。偏置是可训练参数，可被证据推翻。`alpha=1e-3` 初始化，避免大幅扰动已有主干。

这是 **dense soft MoE**，所有专家均计算；它选择性融合信息，但不提供 sparse top-k 的算力节省。初始先验是待检验假设，不是从 MRI 统计推导出的最优概率。三个 query 槽通过最终区域分割间接监督，没有独立类辅助 head，所以“ED/NCR/ET 槽”不保证训练后仍各自对应这些组织。解释时需用验证标签分层查看空间权重，并做屏蔽专家消融，不能只凭平均 gate 把路由当作医学因果解释。

每 epoch 记录 3×5 平均路由矩阵，用于发现权重集中；整体平均可能被背景主导，不能代替肿瘤内诊断。本版本没有强制负载均衡，因为要求 ET 均匀使用五专家可能与模态差异相冲突。先用 M2-U 控制多专家容量，用 M2-N 检验软先验是否有效，再决定是否需要弱正则。

来源：[SimMLM 官方 dmome.py](https://github.com/LezJ/SimMLM/blob/main/models/dmome.py) 的输出路由包含完整单模态网络及 CUDA/形状假设。这里依据 3D plan 的轻量设计另写适配模块，明确不称为 SimMLM 复现。该模块的代码、位置、参数与原论文体系不同。

## F：局部频率与边界细节

在 E0、E1 两条 skip 上计算固定 3×3×3 box 低通 `L(F)`，replicate padding 保证常数特征在边缘仍不产生伪高频。

\[
H=F-L(F),\qquad F'=F+\tanh(\alpha)\sigma(g(F))H.
\]

gate 使用逐通道 1×1×1 卷积，初始化输出 .5；`alpha=1e-3`，允许学习增强或抑制，限制过强高频扰动。模块只改变送入 decoder 的 skip，encoder 的下采样路径保持原样。目标是测试 ET/NCR 局部细节是否有用，ED 大范围低频语义仍由主干承担。

[HFF 官方代码](https://github.com/VinyehShaw/HFF) 与[论文](https://ieeexplore.ieee.org/document/11032150/)支持高低频分解作为研究线索；本实现不是其双分支 HFF-Net、ALC 或 MATLAB 频域处理的复现。这里测的是特征局部差异，不是 MRI k-space 或 PSF。F1 单独验证上述模块；抗混叠下采样和 2 mm 边界损失属于后续各自独立的消融，未混入此次 F1。

## S：直接复用官方 Mamba

实际导入 `vendor.segmambav2.MambaLayer`，完整官方源文件原样保存；[来源与 SHA256](../nnunetv2/utilities/brats2021_audit/vendor/provenance.json)固定到 commit `7b819768826f1d36e3cd81348cefdc3cbb16d3cd`。不重写 selective scan，不以卷积伪装缺失依赖的 Mamba。

官方层执行三种空间排列的全体素序列扫描并相加，**不是三组独立轴向直线扫描，也没有新增反向序列**。共享同一个 Mamba 实例：

\[
U(F)=F+\sum_{j=1}^{3}P_j^{-1}M(\operatorname{LN}(P_jF)).
\]

其 selective SSM 的核心是输入相关的状态更新 `h_t=Abar_t h_(t-1)+Bbar_t x_t, y_t=C_t h_t+D x_t`，参数由官方 `mamba_ssm.Mamba` 处理；局部卷积与门控也保留在原类内部。不同排列让体素在序列中的邻接关系不同，帮助减少单一 raster 顺序偏置。这提供长程上下文假设，不能保证解决 NCR 模态混淆。

外部仅作

\[
F'=F+\tanh(\alpha)[U(F)-F]/3,\quad \alpha_0=10^{-3}.
\]

放在 bottleneck，实际形状 `B×320×4×5×7`，每次扫描 140 tokens。三路平均和小残差用于控制插入幅度。主干先按原 nnU-Net 初始化，再构造 Mamba，保护 `dt_bias` 的特殊初始化；Mamba 部分强制 FP32 运算，输出转回主干 dtype。

本地 `lightmunet` 的 PyTorch `2.0.1+cu117`、mamba_ssm、causal_conv1d 已实际完成 CUDA 检查，无需为这次实验升级环境。源文件还导入 MONAI/einops。迁移机器时必须先运行 preflight，不能只看包是否安装。S1 是 nnU-Net+官方 MambaLayer 的单点适配，不是完整 SegMamba-V2 基准复现。

[SegMamba-V2 官方源文件](https://github.com/ge-xing/SegMamba-V2/blob/7b819768826f1d36e3cd81348cefdc3cbb16d3cd/brats23/models_segmamba/segmambav2.py)为直接代码来源；[MambaHoME](https://github.com/gmum/MambaHoME)为后续层级专家研究参考。MS 仅把当前轻量 MoE 和当前 bottleneck Mamba 合在一起，并不实现 MambaHoME。M-Net 的切片序列训练不纳入此次已具备 3D 上下文的主线。

## 执行路线（沿用 11.txt 的环境和 CLI）

激活并设置环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lightmunet
cd /root/nnUNet
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export nnUNet_raw=/root/autodl-tmp/data/nnUNet_raw
export nnUNet_preprocessed=/root/autodl-tmp/data/nnUNet_preprocessed
export nnUNet_results=/root/autodl-tmp/data/nnUNet_results
export nnUNet_compile=False BRATS_AUDIT_SEED=0
```

当前已有 400 例预处理数据。直接复用现有 plans 和 split，结构消融无需重新执行 plan_and_preprocess。若数据被重新转换或预处理，先重新 QC，并重建相应采样索引；旧索引会在 plans/split/dataset 或病例文件大小/mtime 改变时拒绝使用。文件 stat 校验不等价于完整 NIfTI QC；原始 500 例完整性审计与本次训练坐标准备是两个操作。

索引已为当前 fold 0 生成；以下命令会校验并复用：

```bash
python -m nnunetv2.utilities.brats2021_audit.sampling --prepare
python -m nnunetv2.utilities.brats2021_audit.preflight --mamba --full-patch
```

先执行 E0，之后逐项运行。以下每行是一个独立的 200 epoch 实验，不要在单张 GPU 上同时启动：

```bash
bash scripts/brats2021_audit/run_experiment.sh baseline
bash scripts/brats2021_audit/run_experiment.sh hard
bash scripts/brats2021_audit/run_experiment.sh hardloss
bash scripts/brats2021_audit/run_experiment.sh gate
bash scripts/brats2021_audit/run_experiment.sh moe_uniform
bash scripts/brats2021_audit/run_experiment.sh moe_no_prior
bash scripts/brats2021_audit/run_experiment.sh moe
bash scripts/brats2021_audit/run_experiment.sh frequency
bash scripts/brats2021_audit/run_experiment.sh mamba
```

等价直接命令及后台启动示例：

```bash
nnUNet_compile=False nnUNetv2_train 137 3d_fullres 0 --npz -tr nnUNetTrainer_BraTSMoE200
nohup bash /root/nnUNet/scripts/brats2021_audit/run_experiment.sh moe > /root/nnUNet/train_BraTSMoE200_seed0.log 2>&1 &
# 同一实验恢复：
bash scripts/brats2021_audit/run_experiment.sh moe --c
```

结果目录由 trainer 类名隔离。脚本 `loss` 对应 H0L，`moe_mamba` 对应后期 MS。首轮都从头训练；直接使用旧 baseline 的 pretrained_weights 属于另一个微调对照，新增参数可能不满足 nnU-Net 原生严格预训练加载规则，不能假设加模块后所有旧 checkpoint 都能直接加载。

每个实验结束，CLI 默认对 `checkpoint_final.pth` 做 80 例完整验证，`--npz` 保存概率。以 final 为首轮统一比较口径。`checkpoint_best` 基于训练过程中的区域 pseudo Dice 选出，未专门按互斥 NCR 选出；如改用 best，应对所有实验统一重新验证，并保存独立结果目录。

互斥类评估示例（先运行 E0 和对应模型）：

```bash
TR=nnUNetTrainer_BraTSMoE200
RUN="$nnUNet_results/Dataset137_BraTS2021/${TR}__nnUNetPlans__3d_fullres/fold_0"
BASE="$nnUNet_results/Dataset137_BraTS2021/nnUNetTrainer_BraTSAuditBaseline200__nnUNetPlans__3d_fullres/fold_0/validation"
python -m nnunetv2.utilities.brats2021_audit.evaluate \
  --gt "$nnUNet_raw/Dataset137_BraTS2021/labelsTr" \
  --pred "$RUN/validation" --baseline "$BASE" \
  --split "$nnUNet_preprocessed/Dataset137_BraTS2021/splits_final.json" \
  --output "$RUN/exclusive_audit"
```

锁定模型后，100 例的预测/区域评价命令如下。使用 `-f 0` 避免默认寻找其他 folds，`-chk` 使用 checkpoint 文件名；整个实验统一使用 final：

```bash
nnUNetv2_predict \
  -i "$nnUNet_raw/Dataset137_BraTS2021/imagesTs" \
  -o "$RUN/predictions_test" -d 137 -c 3d_fullres -f 0 \
  -tr "$TR" -chk checkpoint_final.pth
nnUNetv2_evaluate_folder \
  "$nnUNet_raw/Dataset137_BraTS2021/labelsTs" "$RUN/predictions_test" \
  -djfile "$nnUNet_raw/Dataset137_BraTS2021/dataset.json" \
  -pfile "$nnUNet_preprocessed/Dataset137_BraTS2021/nnUNetPlans.json" \
  -o "$RUN/predictions_test/summary.json"
```

首轮不加阈值搜索后处理。之后如沿用 `11.txt` 的 brats_postprocessing，阈值只能从训练/验证侧选出并锁定；使用 Dataset137 对应模型的 threshold_file，不能复用示例中的 Dataset139/SSA 阈值。原始预测与后处理结果分目录报告。

## 实验约束和判定

1. 固定 fold 0 的 320/80、4 通道顺序、区域标签、原 Z-score、patch `128×160×112`、batch 2、200 epoch、每 epoch 250/50 次 train/val iteration、SGD 初始 LR .01、weight decay 3e-5、原增强与 polyLR。所有 trainer 限定单 GPU、非级联 3d_fullres。原 GaussianBlur 和模拟低分辨率增强保留。
2. H1/H2/H0L 分开比较；M1、M2 不默认叠加；F1、S1 单独比较 E0。MS 只在 M2 与 S1 分别有效后运行，不能把 MS 结果解释成两个模块各自的效果。
3. 初始随机种子默认 0，写入 `audit_experiment.json`。公共主干在 adapter 之前初始化，同种子可得到相同主干权重。但 nnU-Net 的多进程增强仍使用 NonDetMultiThreadedAugmenter、worker seeds=None、CuDNN benchmark；这不是逐位可复现承诺。重复实验建议预先固定 0/1/2 三个种子，各种子重跑 E0。通过不同 `nnUNet_results` 根目录隔离种子，避免覆盖。
4. 主看独立 NCR Dice 均值、P10、非空均值、固定体积分层和空 NCR 的 FP mm³；同时看 ED/ET/WT/TC、NCR→ED 和 NCR→ET 混淆。脚本以双空 Dice=1，另报告非空结果与空类 FP；因此可能与旧审计对双空的取值不同。比较时必须让 E0 和候选经过同一脚本。配对 bootstrap 按病例重采样，不能按体素假装增加样本量。
5. NCR 均值 +2 个百分点且 P10 改善可作为预设继续门槛；建议同时要求 ED/ET/WT 均值下降不超过 .5 个百分点，并检查配对区间和实际错误病例。门槛是研究选择，不是保证收益。扫描多个模型带来的选择偏差应通过其他 folds/种子确认。
6. 频率方向进一步判断边界收益时，需要统一 HD95、2 mm Surface Dice、空集约定及分层；当前新增 evaluate 脚本只计算 Dice/体积/混淆/配对区间。病灶级检出、Brier/ECE 和空间路由诊断应在候选通过首轮后补充，不能用现有脚本结果声称这些指标已改善。
7. 每次运行保存 plans、split 哈希、代码哈希、版本、epoch/patch/batch、训练日志、GPU 峰值及最终逐例预测。采样索引受 split/源文件校验约束；不从验证预测或历史失败名单回灌训练，不对 100 例继续逐项选参数。

## 已执行验证及其范围

- 合成标签映射、概率非负/和为 1、层级违例梯度、空 NCR、深监督损失通过。
- 10,000 次样本提议比例符合 .5/.125/.25/.125（原分支/ED/NCR/ET），空 NCR 仍在原分支；强制类别中心在增强前 crop 中存在。
- 11 个 trainer 已通过原生发现与初始化；真实训练 fold 的采样、数据增强、五尺度嵌套 target 已检查。
- 所有结构变体的小 patch CUDA 前向/反向、严格 state_dict 重载、关闭深监督的推理通过，官方 Mamba 源文件哈希吻合。
- 真实计划 batch 2 前后向通过。峰值 PyTorch allocated 显存：E0 5.15 GiB，M1 5.21 GiB，M2 5.67 GiB，F1 7.75 GiB，S1 5.18 GiB。这是合成单步（统一测试辅助损失）的统计，不含正式训练优化器状态、数据队列、进程及长期碎片；不是承诺的训练峰值。一次计时有 warm-up 差异，不用于速度排名。
- E0/M2/F1/S1 已完成原生 Gaussian 重叠滑窗、三轴镜像与导出标签范围检查，合成完整输入 `4×129×160×113`，采用真实 patch；尚未测正式训练后的病例性能。

机器可读记录：[小 patch](brats2021_preflight.json)、[真实 patch](brats2021_full_patch_preflight.json)、[集成检查](brats2021_integration.json)。没有启动训练循环或 optimizer.step，没有修改任何已有实验 checkpoint。
