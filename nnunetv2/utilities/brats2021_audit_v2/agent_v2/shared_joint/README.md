# SharedJoint LegacyV2：真正主干作为第五通信节点

入口：`nnUNetTrainer_BraTSAgentSharedJointLegacyV2Loss300`。

## 与之前两个五专家模型的区别

`GeneralistLegacyV2` 和 `GeneralistFullSizeLegacyV2` 都有一套独立第五专家权重。
本类只有主干和四个单模态专家，共五套图像分割网络；`semantic.experts` 长度严格为4，
不创建、不复制、不别名注册 `semantic.experts[4]`。

主干只输入原来的四MRI通道（缺失通道按显式mask置零），不新增四个可用性图像通道。
可用性mask仍用于采样、专家跳过和router屏蔽，主干作为Joint节点始终可用。

每次forward只运行一次主干encoder/decoder，得到未通信的 $z_J$ 和decoder level3状态 $h_J$。
第5节点直接引用这两个保留梯度的张量，不detach，不进行第二次主干forward。
通过临时forward hook从官方decoder取得状态；调用后移除hook，不复制解码器实现。

## 通信与输出

五个节点顺序为 `T1,T1ce,T2,FLAIR,Joint`。前四节点的状态64通道，Joint状态256通道；
各自用1×1投影进入16通道通信空间。复用原LegacyV2的peer-only QKV、三区域注意力、
差异反馈及残差修正，初始化仍为零输出修正头、固定rho=0.1。

\[
(\tilde z_1,\ldots,\tilde z_4,\tilde z_J)
=C_{Legacy}(z_1,\ldots,z_4,z_J;h_1,\ldots,h_4,h_J),
\]
\[
z_{mix,c}=\sum_{m\in\{1,2,3,4,J\}}w_{c,m}\tilde z_{m,c},\qquad
z_{final,c}=(1-\beta_c)z_{J,c}+\beta_c z_{mix,c}.
\]

**旁路保留的是未通信的原始主干输出**，不是再算一个主干。主干既能向四专家发送信息，
也能接收专家信息；最终梯度经原主干旁路和通信融合分支共同回到同一套主干参数。

通信前 $\tilde z_J=z_J$，因此原主干有效系数是
$(1-\beta_c)+\beta_c w_{c,J}$。例如初始化五节点等权、beta=0.2时，主干系数0.84，
每个单模态专家0.04。这里是预定残差融合的两条计算路径，不是两套网络或重复优化器参数。

## 训练保持与必要的监督去重

300轮、seed、数据划分、增强、optimizer、模态采样、主干DS及ExclusiveLoss沿用原LegacyV2。

- epoch0–49：主干与四专家独立分支预热，通信和融合关闭；所有输入全模态。
- epoch50–99：启用原模态丢弃与五节点融合，通信仍关闭。
- epoch100–299：启用原QKV通信及原refined辅助监督ramp。

令 $L_{main}$ 为原最高分辨率融合输出＋主干粗尺度DS损失，$L_{local,4}$ 为可用四专家
本地监督均值，$L_{fusion}$ 为五节点融合监督，$L_{refined,5}$ 为五节点通信后监督。

\[
L=\begin{cases}
L_{main}+L_{local,4},&e<50,\\
L_{main}+0.25L_{local,4}+0.25L_{fusion}+\lambda(e)L_{refined,5},&e\ge50.
\end{cases}
\]

$\lambda$沿用原实现：100轮前为0，通信后按原20轮ramp到0.1。
第五节点原始输出已经是主干，不能再假装独立专家重复加一次raw-Joint local loss。
因此删除原独立Generalist那项local supervision，仅对四专家求均值；通信后Joint输出
经过新的修正，仍参与refined辅助监督。这是共享主干后的必要监督去重，
**不声称与五独立专家模型的每个loss项逐项相同**，也没有增加新损失或新系数。

## 运行

```bash
cd /data02/chujingyi/nnUNet
export PYTHONPATH=/data02/chujingyi/nnUNet
export nnUNet_raw=/data02/chujingyi/data/nnUNet_raw
export nnUNet_preprocessed=/data02/chujingyi/data/nnUNet_preprocessed
export nnUNet_results=/data02/chujingyi/data/nnUNet_results
unset BRATS_AGENT_INITIAL_WEIGHTS
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 137 3d_fullres 0 \
  -tr nnUNetTrainer_BraTSAgentSharedJointLegacyV2Loss300
```

这是新类从头训练的消融。旧Small/FullSize Generalist类和checkpoint完整保留，不能用它们
的checkpoint直接当新类续训。使用与原实验相同的BRATS_AUDIT_SEED；默认0。
当前没有启动GPU训练，显存和病例效果须实际实验，不能以参数减少推断Dice一定提高。

```bash
python scripts/brats2021_audit_v2/test_shared_joint_legacy.py \
  --report /data02/chujingyi/brats_dataset_audit/shared_joint_interaction_ceiling_20261030/verification_shared_joint.json
```

检查使用CPU真实网络宽度和小空间合成patch，覆盖15模态、主干一次forward、
真实张量身份、双向通信梯度、预热监督去重、epoch collate、checkpoint、
现有缺失模态predictor和trainer发现。测试不会更新患者模型参数。
