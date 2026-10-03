import importlib.util
import multiprocessing
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/experiment_store.py"
SPEC = importlib.util.spec_from_file_location("experiment_store", SOURCE)
store_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(store_module)
JsonRecordStore = store_module.JsonRecordStore


def _write_shard(path, shard):
    store = JsonRecordStore(path)
    for index in range(25):
        store.put(f"records/{shard}/{index:02d}", {"shard": shard, "index": index})
    store.close()


def test_round_trip_prefix_and_update(tmp_path):
    store = JsonRecordStore(tmp_path / "records.sqlite3")
    store.put("generation/a/01", {"value": 1, "unicode": "实验"})
    store.put("generation/a/02", {"value": 2})
    store.put("quality/a/01", {"score": float("inf")})
    store.put("generation/a/01", {"value": 3})

    assert store.get("generation/a/01") == {"value": 3}
    assert store.contains("generation/a/02")
    assert store.count() == 3
    assert store.count("generation") == 2
    assert [key for key, _ in store.items("generation/a")] == [
        "generation/a/01",
        "generation/a/02",
    ]
    assert store.delete("generation/a/02")
    assert not store.delete("generation/a/02")
    with pytest.raises(KeyError):
        store.get("generation/a/02")


@pytest.mark.parametrize("key", ["", ".", "../x", "/absolute", "a/../b"])
def test_rejects_unsafe_keys(tmp_path, key):
    store = JsonRecordStore(tmp_path / "records.sqlite3")
    with pytest.raises(ValueError):
        store.put(key, {})


def test_concurrent_process_writers(tmp_path):
    path = tmp_path / "records.sqlite3"
    # Production prepare creates the schema before launching workers.
    JsonRecordStore(path).close()
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_write_shard, args=(path, shard)) for shard in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(20)
        assert process.exitcode == 0

    store = JsonRecordStore(path)
    assert store.count("records") == 100
    assert store.get("records/3/24") == {"shard": 3, "index": 24}
    store.checkpoint()
    store.close()
    assert [item.name for item in tmp_path.iterdir()] == ["records.sqlite3"]
