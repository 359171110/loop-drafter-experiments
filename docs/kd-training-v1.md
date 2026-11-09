# Loop drafter：纯 KD、首位置加权、循环覆盖

本版保留 `ouro_ssd_loop_memory_v1` 的 trajectory 架构，从 seed 0 随机初始化训练一个版本。沿用已采集的 16,384 条轨迹、请求顺序、AdamW 与学习率计划、四个 epoch 和 8,192 次更新。

- 目标：纯 `KL(teacher || student)`，温度 1，teacher top128 加聚合 tail；student 使用完整词表归一化。CE 仅记录，不参与梯度。
- 五个 child 的权重：`[2,1,1,1,1]`。分母是全局更新的有效位置权重和，多卡梯度求和后不再除 world size。
- 软标签：缓存 BF16 最终轮状态转 FP32，与冻结 LM head 相乘。第一个 child 的 teacher 状态位于处理完 bonus 的位置；这些未来状态只用于标签。它是缓存重建的近似 teacher 分布。
- Anchor：每请求每轮最多 128 个。按请求 ID 和 seed 固定随机排列，跨 epoch 连续取下一段；完整覆盖后才循环。每轮请求内无重复。
- 全局每步 8 条请求，microbatch 64；支持 1/2/4 卡。两卡各 4 条请求。
- 当前数据有 4,769,137 个可用首位置；四轮全部覆盖，共 7,405,096 次首位置监督，相比原 64-anchor 配方的 3,993,528 次增加约 85.4%。不同末尾 anchor 的有效深度数也会不同，因此这是完整配方比较。
- 每 100 步保存恢复点，每轮保留 checkpoint；评测固定最终轮。仅恢复同一新实验，不能用旧 CE checkpoint 继续训练。

运行入口读取原 v1 run 中的 `recipe.json`、完整采集结果和最终 trajectory checkpoint。所有数据路径由原 recipe 提供：

```bash
python -m scripts.loop_drafter.run_kd_experiment \
  --run /experiments/loop_drafter/loop_memory_v1_kd_first2_cycle128_seed0 \
  --baseline-run /experiments/loop_drafter/loop_memory_v1_data_v2_scratch_seed0 \
  --gpus 5,6
```

`--prepare-only` 注册配置、代码快照和预算，不启动 GPU。`--stop-after-steps 3` 用于独立短运行目录，只验证执行路径，不进入评测。正式运行请使用新的输出目录并省略此参数。重启原正式命令会读取该目录的恢复点。

启动器等待指定卡连续两次空闲并持有项目 GPU lease，不终止其他任务。训练完成后自动在同一 GPU 上按请求轮换测量 AR、原 v1 trajectory CE、新版 KD；core 为 GSM8K 300 / HumanEval 164 / ShareGPT 300，ARC-Challenge 300 独立报告，无 MBPP。保持冻结 input IDs、FP32 四轮 target、BF16 drafter、batch 1、greedy、最多生成 256 tokens、输入上限 2048。记录首 child 接收率、平均接受长度、速度及 token 一致性。GPU 监控记录若发现外部进程，汇总结果标为共享环境诊断。

当前比较用于判断整版是否有效。损失、权重和覆盖度的独立消融留待整版取得收益之后。

训练执行优化：KD 默认将每条轨迹的 token 和隐藏状态传到 GPU 一次，再在 GPU 上取历史窗口，避免每个 microbatch 在 CPU 展开重叠窗口并重复传输。实际训练在第 4,600 步保存断点后切换该管道，并将物理 microbatch 提到 128；模型、优化器、请求顺序、anchor、全局 batch 和学习率继续原进度。运行目录保留切换前源码快照及变更记录。

同一断点、同四条请求的单卡前后向测量（不含优化器和双卡通信，预热后 3 次中位数）：原 CPU 管道 microbatch 64 为 0.803 秒，GPU 管道 64 为 0.455 秒，GPU 管道 128 为 0.158 秒。该测量不能直接代表双卡完整训练加速比。输入张量逐元素一致；loss 从 2.129929232 变为 2.129929318。GPU 管道 128 的梯度相对 L2 差为 0.383%，原路径重复运行自身为 0.092%，因此不保证浮点逐位复现。峰值分配显存约 3.93 GiB。边界批次、teacher 对齐、跨卡归一化、断点恢复共 5 项测试通过。

在新运行启动前，可在运行目录写入 `training_execution.json`，从指定步启用较大 microbatch；新实验从头使用可设 `from_step: 0`：

```json
{"schema":"loop_training_execution_v1","microbatch":128,"from_step":4600}
```

恢复既有实验时，代码快照需要保留并登记这次执行变更；不要直接覆盖旧快照。完整小规模测量见 [原始记录](kd-device-staging-benchmark.json)。
