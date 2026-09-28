# MoE + ExclusiveLoss 与模块日志（2026-09-17）

下一项：`nnUNetTrainer_BraTSMoENoPriorLoss200`。若有收益，再运行 `nnUNetTrainer_BraTSMoEUniformLoss200`，区分学习路由与额外专家容量带来的收益。两个类均在原统一注册文件中，继承 `nnUNetTrainer_BraTSExclusiveLoss200`，保留损失、原采样、200 epochs 和优化器设置。block/network 复用已有实现。无先验版为 `mode=moe_no_prior`，均匀融合对照为 `mode=moe_uniform`。

## GPU 执行

沿用 `11.txt` 的 lightmunet 环境、nnUNet 路径和线程配置。挂载 GPU 后，先执行：

```bash
nnUNet_compile=False nnUNetv2_train 137 3d_fullres 0 --npz \
  -tr nnUNetTrainer_BraTSMoENoPriorLoss200
```

有收益后再单独执行对照：

```bash
nnUNet_compile=False nnUNetv2_train 137 3d_fullres 0 --npz \
  -tr nnUNetTrainer_BraTSMoEUniformLoss200
```

也已在现有脚本加入 `moe_no_prior_loss` 和 `moe_uniform_loss` 两个入口：

```bash
bash /root/nnUNet/scripts/brats2021_audit/run_experiment.sh moe_no_prior_loss
```

该脚本保留用户原先设置的结束后关机行为（成功、失败均关机）。直接使用上面的 nnUNetv2_train 命令没有额外的关机步骤。修正了原脚本 `lossgate` 分支中 `trainer = ...` 的 shell 赋值错误。

## 日志位置与含义

每个实验自己的 `fold_0/training_log_*.txt` 和 `fold_0/module_diagnostics.jsonl` 均记录新增模块诊断，不写入别的实验目录。GPU 训练路径、AMP unscale、梯度裁剪、optimizer.step 沿用原 trainer；未增加 CPU 训练模式或 Mamba 替代实现。

- `MODULE_GRAD_FEATURE`：默认每 50 个训练 step 采一次，每 epoch 汇总。记录模块总梯度、非零梯度比例、缺失梯度张量数、实际参数更新范数及更新/参数范数、残差系数和特征变化。梯度位于 AMP unscale 和全局裁剪之后。AMP 跳步另有字段。
- `grad_groups`：进一步区分 MoE 的 `router`、`experts.0` 到 `experts.4`、`project`、`scale`。专家顺序为 T1/T1ce/T2/FLAIR/joint。同时记录 L2 与 RMS，避免仅凭大模块较大的 L2 判断学习能力。初始若某个早期 router 层梯度为零，不等于整个模块断梯度；结合后续 epoch 和实际参数更新判断。
- `MODULE_ROUTING`：每 epoch 保存 3×5 平均专家权重，行是 q0/q1/q2，列是五个专家。query 没有独立类别监督，不保证对应 ED/NCR/ET；全 patch 平均含背景，也不是因果贡献。
- `MODULE_ABLATION`：第 1 个 epoch，以及第 25、50、75……个 epoch，在当前第一个验证 batch 上逐个旁路新增模块。`delta_dice_pp_full_minus_bypass` 是完整模型 Dice 减去旁路后的 Dice，正值表示该模块有利于当前 patch；同时保存 WT/TC/ET/ED/NCR 的 FP、FN 和 BCE 差值。仅旁路整个 MoE，不声称量化了每位专家的因果贡献。

特征改变量表示模块改动了多少，梯度表示优化信号，旁路差值表示当前模型在该 batch 上对模块的依赖：三者不能混为一谈。旁路诊断不等于重训消融，也不等于完整病例滑窗评估；两个预测都为空的 patch Dice 记为 null。验证探针在 eval/no_grad 下运行，保存并恢复随机数状态，结果不加入常规验证指标或 checkpoint 选择。

可选调节开销：

```bash
export BRATS_DIAG_EVERY=50
export BRATS_ABLATE_EVERY=25
```

设为 0 分别关闭梯度/特征采样或旁路诊断；原有 MoE 每 epoch 路由统计仍保留。正式比较继续固定 split、采样、loss 权重、训练时长和推理 checkpoint 规则；保留非空 NCR 与空 NCR 假阳性的分项评估。

## 检查范围

本次完成 Python 语法、trainer 继承/模式配置、日志 mixin 接入以及 shell 语法检查。未运行训练、网络前后向或 CUDA/AMP 检查。原有 GPU 诊断检查脚本未改成 CPU 模式。
