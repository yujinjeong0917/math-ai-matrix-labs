"""파인튜닝 트랙 2장 검증. `uv run pytest -q finetune/ch02_huggingface_map` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈로 다시 계산한다.
GPU·네트워크·API 키 없이 돈다.
"""

import json
import struct
from pathlib import Path

import pytest

import ft_ch02_hfmap as H

HERE = Path(__file__).parent


# ---------- 출처 파일 ----------

def test_sources_dated_and_pinned():
    s = H.load_sources()
    assert s["checked"] == "2026-10-11" and H.SOURCES_FILE.name == "sources_2026-10-11.json"
    for repo in s["repos"].values():
        assert len(repo["sha"]) == 40                        # 커밋 해시는 40자 전체로 고정
        assert repo["url_card"].startswith("https://huggingface.co/")
    for lib in s["libraries"]:
        assert lib["url"].startswith("https://") and lib["version"] and lib["quote"]
        assert "—" not in lib["quote"]                  # 엠대시가 섞인 인용은 쓰지 않는다


# ---------- 첫 화면 손계산: 헤더 한 줄 ----------

def test_hand_embed_row():
    n, h, meta = H.read_safetensors_header(H.BASE / "safetensors_headers" / "model-00001-of-00004.header.bin")
    t = h["model.embed_tokens.weight"]
    assert meta == {"format": "pt"} and n == 9368
    assert t["dtype"] == "BF16" and t["shape"] == [152064, 3584]
    assert 152064 * 3584 == 544_997_376
    assert t["data_offsets"] == [0, 544_997_376 * 2] == [0, 1_089_994_752]


# ---------- 가중치 파일 네 개 ----------

def test_shards_add_up_and_match_index_and_api():
    src = H.load_sources()["repos"]["qwen2.5-7b"]
    shards = H.shard_headers()
    assert len(shards) == 4
    params = nbytes = 0
    names = set()
    for f, s in shards.items():
        st = H.header_stats(s["header"])
        assert st["contiguous"] and st["size_matches_shape"] and st["dtypes"] == {"BF16": st["tensors"]}
        assert src["shard_file_sizes"][f] == 8 + s["n"] + st["bytes"]       # 파일 = 8바이트 + 헤더 + 값
        params += st["params"]
        nbytes += st["bytes"]
        names |= set(s["header"])
    idx = H.load_json(H.BASE / "model.safetensors.index.json")
    assert len(names) == 339 == len(idx["weight_map"]) and set(idx["weight_map"]) == names
    assert nbytes == idx["metadata"]["total_size"] == 15_231_233_024
    assert params == src["api_safetensors_total"] == 7_615_616_512 == nbytes // 2


def test_same_count_as_chapter5():
    # 5장이 config.json으로 센 값과 같아야 한다(5장 실습 폴더가 있으면 대조)
    p = HERE.parent / "ch05_gpu_memory" / "data" / "specs_2026-10-11.json"
    if not p.exists():
        pytest.skip("5장 실습 폴더가 없어요")
    assert H.load_json(p)["model"]["safetensors_total"] == 7_615_616_512


# ---------- 모델 카드 ----------

def test_front_matter():
    b = H.read_front_matter(H.BASE / "README_excerpt.md")
    i = H.read_front_matter(H.INST / "README_excerpt.md")
    ours = H.read_front_matter(H.DATA / "our_adapter_example" / "README.md")
    assert b["license"] == i["license"] == "apache-2.0"
    assert "base_model" not in b and i["base_model"] == "Qwen/Qwen2.5-7B"
    assert b["language"] == ["en"] and i["tags"] == ["chat"]
    assert ours["library_name"] == "peft" and ours["base_model"] == "Qwen/Qwen2.5-7B"
    assert "- Number of Parameters: 7.61B" in H.card_body_lines(H.BASE / "README_excerpt.md")


def test_front_matter_rejects_unclosed(tmp_path):
    p = tmp_path / "README.md"
    p.write_text("---\nlicense: mit\n", encoding="utf-8")
    with pytest.raises(ValueError):
        H.read_front_matter(p)


# ---------- config.json ----------

def test_config_readings_and_diff():
    b, i = H.load_json(H.BASE / "config.json"), H.load_json(H.INST / "config.json")
    r = H.config_readings(b)
    assert r["head_dim"] == 3584 // 28 == 128 and r["q_heads_per_kv_head"] == 7 and r["tied"] is False
    d = H.dict_diff(b, i)
    assert d["eos_token_id"] == [151643, 151645]
    assert d["max_position_embeddings"] == [131072, 32768]
    assert set(d) == {"eos_token_id", "max_position_embeddings", "transformers_version", "use_mrope"}


def test_vocab_rows_exceed_tokenizer():
    tc = H.load_sources()["tokenizer_json_counts"]
    assert tc["vocab"] + tc["added_tokens"] == tc["total"] == 151665
    st = H.special_tokens(H.load_json(H.BASE / "tokenizer_config.json"))
    assert max(st["table"]) == 151664 == tc["total"] - 1    # 번호는 0부터라 마지막 번호 = 개수 - 1
    assert H.load_json(H.BASE / "config.json")["vocab_size"] - tc["total"] == 399


# ---------- tokenizer_config / generation_config ----------

def test_eos_differs_base_vs_instruct():
    sb = H.special_tokens(H.load_json(H.BASE / "tokenizer_config.json"))
    si = H.special_tokens(H.load_json(H.INST / "tokenizer_config.json"))
    assert (sb["eos"], sb["eos_id"]) == ("<|endoftext|>", 151643)
    assert (si["eos"], si["eos_id"]) == ("<|im_end|>", 151645)
    assert sb["pad"] == si["pad"] == "<|endoftext|>" and sb["bos"] is None and sb["add_bos_token"] is False
    gb, gi = H.load_json(H.BASE / "generation_config.json"), H.load_json(H.INST / "generation_config.json")
    assert gb["do_sample"] is False and gb["eos_token_id"] == 151643
    assert gi["do_sample"] is True and gi["eos_token_id"] == [151645, 151643] and gi["temperature"] == 0.7


def test_default_system_from_template_text():
    tb = H.load_json(H.BASE / "tokenizer_config.json")["chat_template"]
    ti = H.load_json(H.INST / "tokenizer_config.json")["chat_template"]
    assert H.default_system_in_template(tb) == H.QWEN25_DEFAULT_SYSTEM["base"]
    assert H.default_system_in_template(ti) == H.QWEN25_DEFAULT_SYSTEM["instruct"]


# ---------- chat_template ----------

def test_stdlib_render_matches_jinja2_fixture():
    fx = H.load_fixture()
    msgs = H.load_json(H.DATA / "messages_example.json")["cases"]
    assert len(fx["cases"]) == 8
    for key, case in fx["cases"].items():
        model, name, agp = key.split("|")
        ds = H.QWEN25_DEFAULT_SYSTEM["base" if model == "base" else "instruct"]
        assert H.render_qwen25(msgs[name], agp == "True", ds) == case["text"], key


def test_render_hand_example():
    s = H.render_qwen25([{"role": "user", "content": "안녕"}], add_generation_prompt=True)
    assert s == ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                 "<|im_start|>user\n안녕<|im_end|>\n<|im_start|>assistant\n")
    with pytest.raises(NotImplementedError):
        H.render_qwen25([{"role": "tool", "content": "{}"}])


def test_token_fixture_consistent():
    fx = H.load_fixture()
    c, p = fx["cases"], fx["pieces"]
    assert fx["tokenizers"] == "0.23.3" and fx["measured"] == "2026-10-11"
    assert c["inst|with_system|False"]["n_tokens"] == c["base|with_system|False"]["n_tokens"] == 123
    # add_generation_prompt=True는 '<|im_start|>assistant\n' 세 토큰을 더 붙인다
    for m in ("base", "inst"):
        for case in ("with_system", "no_system"):
            assert c[f"{m}|{case}|True"]["n_tokens"] - c[f"{m}|{case}|False"]["n_tokens"] == 3
    # system을 안 주면 기본 문장이 들어간다: 차이 = 기본 system 칸 토큰 - 우리 system 칸 토큰
    assert c["inst|no_system|False"]["n_tokens"] - c["base|no_system|False"]["n_tokens"] == \
        p["default_system_inst"] - p["default_system_base"] == 10
    assert p["im_start"] == p["im_end"] == p["endoftext"] == 1
    assert p["learned_span_in_context"] == 55 and p["assistant_content"] == 52


# ---------- 라이브러리 지도 ----------

def test_dependency_layers():
    deps = dict(H.load_sources()["requires_dist"])
    deps.pop("transformers_extra_torch")
    layers = H.topo_order(deps)
    assert layers[-1] == ["peft", "trl"] and layers[1] == ["transformers"]
    assert "accelerate" in layers[0] and "tokenizers" in layers[0]


def test_topo_detects_cycle():
    with pytest.raises(ValueError):
        H.topo_order({"a": ["b"], "b": ["a"]})


def test_every_stage_has_a_library():
    m = H.stage_map(H.load_sources())
    assert all(v["libs"] for v in m.values())
    assert m["train_loop"]["libs"] == ["TRL", "accelerate"]


# ---------- 작은 safetensors 직접 쓰기 ----------

def test_write_and_read_toy(tmp_path):
    p = tmp_path / "toy.safetensors"
    n = H.write_safetensors(p, {"w": ("F16", [2, 2], [0.5, -1.0, 0.1, 2.0])}, {"format": "pt"})
    raw = p.read_bytes()
    assert struct.unpack("<Q", raw[:8])[0] == n and n % 8 == 0 and raw[8:9] == b"{"
    _, h, meta = H.read_safetensors_header(p)
    assert h["w"] == {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]} and meta == {"format": "pt"}
    assert H.read_tensor(p, "w") == [0.5, -1.0, struct.unpack("<e", struct.pack("<e", 0.1))[0], 2.0]
    assert len(raw) == 8 + n + 8


def test_results_file_matches_if_present():
    p = HERE / "results" / "ch02.json"
    if not p.exists():
        pytest.skip("experiments.py를 먼저 돌려요")
    r = json.loads(p.read_text(encoding="utf-8"))
    assert r["e1_shards"]["params"] == 7_615_616_512 and r["e5_template"]["all_match"] is True
    assert r["e5_template"]["learned_share"] == round(55 / 123, 4)
