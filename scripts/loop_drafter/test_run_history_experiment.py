import json

import pytest

from .run_history_experiment import METHODS, merge_evaluation


def prepare_shards(tmp_path):
    prompts = [{"id": "p0", "source": "math"}, {"id": "p1", "source": "chat"}]
    (tmp_path / "panels").mkdir()
    for i, prompt in enumerate(prompts):
        (tmp_path / "panels" / f"worker{i}.jsonl").write_text(json.dumps(prompt) + "\n")
        shard = tmp_path / "shards" / f"worker{i}"
        shard.mkdir(parents=True)
        rows = []
        for method in METHODS:
            rows.append(prompt | {"method": method, "tokens": [1, 2],
                                  "prefill_seconds": .1,
                                  "decode_seconds": (i + 1) * (2 if method == "ar" else 1),
                                  "rounds": [] if method == "ar" else [
                                      {"committed_tokens": 2, "accepted_children": 1}]})
        (shard / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (shard / "summary.json").write_text(json.dumps({"hardware": {"cuda_visible_devices": str(i)}}))
    return prompts


def test_merge_pairs_methods_with_their_own_gpu_reference(tmp_path):
    prompts = prepare_shards(tmp_path)
    result = merge_evaluation(tmp_path, prompts, ["0", "1"])
    row = next(r for r in result["results"] if r["source"] == "__all__" and r["method"] == "history")
    assert row["requests"] == row["exact_matches"] == 2
    assert row["decode_latency_speedup"] == 2.
    assert row["mean_committed_tokens_per_verify"] == 2.
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 8
    assert {(r["id"], r["physical_gpu"]) for r in rows} == {("p0", "0"), ("p1", "1")}


def test_merge_rejects_duplicate_records(tmp_path):
    prompts = prepare_shards(tmp_path)
    path = tmp_path / "shards" / "worker0" / "requests.jsonl"
    with path.open("a") as stream:
        stream.write(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        merge_evaluation(tmp_path, prompts, ["0", "1"])
