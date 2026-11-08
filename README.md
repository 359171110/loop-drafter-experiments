# Loop drafter experiments

Ouro-1.4B 四轮 loop target 的独立 drafter 实验代码。当前冻结架构为 `ouro_ssd_loop_memory_v1`，数据协议为 `data_v2_16384`，训练协议为 `scratch_no_mbpp_v1`。

## 当前状态

包含模型、训练、评测、数据准备工具、多卡调度、空闲 GPU 自动训练入口及 CPU 测试。新版 16K 提示已在原实验环境冻结，但完整新版轨迹尚未采集完成。现有 `capture*.py` 保留历史采集实现；遵守新版 frozen input_ids、raw completion 与长度分档的采集适配器仍需补齐，不能直接把历史采集命令当作新版数据流水线。

Git 仓库仅存代码和协议。目标模型、冻结提示、轨迹、frozen_io 和 checkpoint 需另行传输；克隆仓库本身不会启动任务。

## 环境

原运行环境为 Linux、Python 3.12、PyTorch 2.11.0+cu126。先按服务器驱动安装合适的 PyTorch CUDA wheel，再安装依赖：

```bash
python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

无本地模型时，7 项 target 集成测试会跳过；设置 `OURO_TEST_MODEL=/path/to/Ouro-1.4B` 可用真实模型代码和小型随机权重在 CPU 上验证。

目标模型目录须包含 Ouro 自定义模型代码和原 tokenizer 文件。运行命令均从仓库根目录执行。

## 固定实验

- v1 使用 SSD 主干、真实 token prefix 修正和 loop memory；比较 `none / final / final_capacity / trajectory` 四个模式。
- 四个模式均从头初始化，seed 0，沿用目标模型冻结 embedding / LM head，不载入旧 drafter checkpoint。只有同一运行中断恢复才允许 resume。
- 16,384 条训练提示：数学 5,120、代码 5,120、推理 4,096、对话 2,048；输入最多 2,048 tokens，目标输出最多 512 tokens，遇 EOS 停止。
- 4 epochs，8 requests/update，每请求最多 64 anchors，共 8,192 updates；全局有效 child token 的均匀 CE，无 KD。
- 核心评测 GSM8K 300、HumanEval 全部 164、ShareGPT 300；ARC-Challenge 300 独立报告。MBPP 不参与评测。
- 评测 batch 1、greedy、最多生成 256 tokens，FP32 target / BF16 drafter；比较速度及接受长度，记录 token 一致性，不做答案准确率评分。

完整参数见 [固定协议](scripts/loop_drafter/data_protocol_v2.json)。历史实验脚本保留供依赖和追溯，当前实验应使用下面的 v1 入口。

## 新服务器配置与数据

```bash
python configure.py --root /experiments/loop_drafter --model /models/Ouro-1.4B
```

生成被 Git 忽略的 `local/protocol.json`，不修改固定模板。根目录需要以下材料：

```text
data_v2/prompts/train.jsonl
data_v2/capture/manifest.jsonl
data_v2/capture/data_index.json
data_v2/capture/config.json
capture_fp32/frozen_io.pt
evaluation_v2/prompts.jsonl
evaluation_v2/sources.json
evaluation_reasoning_v1/prompts.jsonl
evaluation_reasoning_v1/sources.json
```

还需复制 manifest 引用的全部轨迹文件，以及数据选择、重叠排查和采集计划等审计材料。完整 16K 轨迹尚未就绪时不能训练。

**跨服务器请优先保留原材料的绝对路径**（可用挂载或软链接），并让 `--root`、`--model` 对应实际路径。`configure.py` 只改 recipe，不迁移数据内部路径。评测条目中的 `tokenizer_path`、sources 中的文件身份、轨迹 index 中的绝对路径和 mtime 都会校验；普通复制到另一目录后直接运行可能失败。迁移应保留时间戳，并在正式训练前通过 runner 校验。若必须重定位，应单独重建路径元数据及对应身份，保留原提示、input_ids、顺序与审计记录；仓库目前没有自动重定位工具。

版本对比使用同一份冻结提示和物化轨迹，不在每台服务器重新抽样。不同 GPU 的绝对速度须分开报告。

## 计划、训练和评测

评测提示及其 sources 文件需先到位；仅缺轨迹时会输出等待状态。

```bash
python -m scripts.loop_drafter.run_loop_memory_experiment \
  --protocol local/protocol.json --root /experiments/loop_drafter --gpus 0,1,2,3
```

完整材料通过校验后，指定 GPU 从头训练四个模式：

```bash
python -m scripts.loop_drafter.run_loop_memory_experiment \
  --protocol local/protocol.json --root /experiments/loop_drafter \
  --gpus 0,1,2,3 --execute --train-only
```

支持 1、2、4、8 张 GPU；8 卡时每模式 2 个 rank，较少卡时分批执行，全局训练预算保持一致。

也可明确启动等待器，在采集完整且 GPU 满足条件后自动开始训练：

```bash
python -m scripts.loop_drafter.auto_train_when_ready \
  --protocol local/protocol.json --root /experiments/loop_drafter
```

等待器不会启动数据采集或评测。它检查无其他计算进程、显存余量和利用率，连续两次满足条件后选择 8/4/2/1 卡，并检查项目内 GPU lease。启动后不要修改其 protocol 文件。

要执行含评测的完整流水线，在上述 runner 命令中去掉 `--train-only`。已有同运行 checkpoint 按协议恢复；正式计时需使用无其他计算负载的 GPU。核心集和 ARC 分开汇总，混用 GPU 的结果只作诊断。

## 代码入口

- `scripts/loop_drafter/loop_memory_draft.py`：冻结 v1 模型。
- `train_loop_memory.py` / `evaluate_loop_memory.py`：训练与评测。
- `run_loop_memory_experiment.py` / `auto_train_when_ready.py`：协议校验、多卡运行与自动等待。
- `prepare_data_v2.py`：数据准备；依赖本地源数据和历史冻结材料，不会自动下载完整数据集。
- `scripts/fest_hybrid`、`scripts/fest_hybrid_prefix_tree`：主干依赖。
