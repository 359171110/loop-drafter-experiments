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
