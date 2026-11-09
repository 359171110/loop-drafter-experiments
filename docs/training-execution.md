# Training execution schedule

2026-11-09，H20，使用运行中的四个方案 checkpoint 对同一组 8 条请求测速；每条最多 64 anchors。保持原 BF16 autocast、CE、优化器和全局预算，仅改变物理 microbatch。表中计时为前向／反向与梯度累积，不包含数据读取、优化器 step 和 checkpoint 写盘。

| 模式 | microbatch 16 / 秒 | microbatch 64 / 秒 | 加速比 |
|---|---:|---:|---:|
| none | 1.009 | 0.539 | 1.87× |
| final | 1.305 | 0.443 | 2.95× |
| final_capacity | 1.531 | 0.661 | 2.32× |
| trajectory | 1.436 | 0.565 | 2.54× |

单次成对测试有运行波动；前一轮测试的加速比为 1.78–2.37×。最大 CE 差异约 8.83e-6，梯度相对 L2 差异约 0.29%–0.33%，并非逐位相同。FP32 小模型测试核验了有效标签归一化与梯度累积的数学等价性。

本轮四个方案统一采用：第 1–6400 步 microbatch 16，第 6401–8192 步 microbatch 64。每次更新仍为 8 requests，所有 anchor、采样顺序、学习率及总更新次数保持相同。

复现本轮 v1 时，在启动训练前执行：

```bash
mkdir -p /experiments/loop_drafter/loop_memory_v1_data_v2_scratch_seed0
cp scripts/loop_drafter/training_execution_v1.json \
  /experiments/loop_drafter/loop_memory_v1_data_v2_scratch_seed0/training_execution.json
```

trainer 从输出目录的父目录读取该文件。原始 settings.microbatch 保留注册值 16，checkpoint 的 execution_plan 保存实际分段调度，metrics 的 execution_microbatch 记录每一步实际值。恢复时继承调度并拒绝改变已登记调度；首次引入调度时不得追溯到已完成步数之前。评测要求四个方案的 execution_plan 相同。

正在运行的任务不会热加载新源码。部署时应保存原 source_snapshot/source_identity 到版本归档，明确登记执行优化的源码修订，再恢复同一次训练的 checkpoint；模型结构、初始化与数据身份不变。自动等待器恢复时沿用已有 GPU 分配。

原始测试记录见 [training-microbatch-benchmark.json](training-microbatch-benchmark.json)。
