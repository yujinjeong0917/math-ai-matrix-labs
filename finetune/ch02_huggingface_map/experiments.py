"""파인튜닝 트랙 2장 실험. `uv run python finetune/ch02_huggingface_map/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 첫 가중치 파일 헤더의 model.embed_tokens.weight 한 줄 읽기
E1 가중치 파일 네 개의 헤더: 텐서 수, 숫자 수, 바이트, 파일 크기와 맞는지, index.json·Hub API와 맞는지
E2 모델 카드 머리말: base, instruct, 우리 어댑터 예시
E3 config.json 읽기와 두 모델 비교, 토큰 사전 크기와 vocab_size의 차이
E4 tokenizer_config.json과 generation_config.json: eos, pad, bos, 기본 system 문장, 생성 설정
E5 chat_template: 표준 라이브러리 렌더러를 jinja2 결과와 대조, 토큰 수(라이브러리로 한 번 잰 값)
E6 라이브러리 지도: 단계별 담당, 의존 관계 층
E7 작은 safetensors 파일을 직접 써서 다시 읽기

표준 라이브러리만 쓴다. 네트워크 없이 data/ 의 파일만 읽는다.
"""

import json
import platform
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import ft_ch02_hfmap as H  # noqa: E402

OUT = HERE / "results" / "ch02.json"


def e0_hand():
    n, h, meta = H.read_safetensors_header(H.BASE / "safetensors_headers" / "model-00001-of-00004.header.bin")
    t = h["model.embed_tokens.weight"]
    rows, cols = t["shape"]
    return {"header_len": n, "metadata": meta, "name": "model.embed_tokens.weight", "dtype": t["dtype"],
            "shape": t["shape"], "numbers": rows * cols, "bytes": rows * cols * 2, "data_offsets": t["data_offsets"],
            "bytes_match_offsets": rows * cols * 2 == t["data_offsets"][1] - t["data_offsets"][0],
            "first_three": [{"name": k, "shape": h[k]["shape"], "data_offsets": h[k]["data_offsets"]}
                            for k in sorted(h, key=lambda k: h[k]["data_offsets"][0])[:3]]}


def e1_shards():
    src = H.load_sources()["repos"]["qwen2.5-7b"]
    idx = H.load_json(H.BASE / "model.safetensors.index.json")
    shards = H.shard_headers()
    rows, names = [], set()
    for f, s in shards.items():
        st = H.header_stats(s["header"])
        size = src["shard_file_sizes"][f]
        rows.append({"file": f, "header_len": s["n"], **st, "file_size": size,
                     "file_size_matches": size == 8 + s["n"] + st["bytes"]})
        names |= set(s["header"])
    total_params = sum(r["params"] for r in rows)
    total_bytes = sum(r["bytes"] for r in rows)
    wm = idx["weight_map"]
    placed_ok = all(name in shards[wm[name]]["header"] for name in wm)
    return {"rows": rows, "tensors": len(names), "params": total_params, "bytes": total_bytes,
            "file_bytes": sum(r["file_size"] for r in rows),
            "header_bytes": sum(8 + r["header_len"] for r in rows),
            "index_total_size": idx["metadata"]["total_size"], "index_tensors": len(wm),
            "index_matches_headers": set(wm) == names and placed_ok,
            "api_total": src["api_safetensors_total"], "api_matches": total_params == src["api_safetensors_total"],
            "gb_weights": round(total_bytes / 1e9, 4), "gb_if_fp32": round(total_params * 4 / 1e9, 4)}


def e2_cards():
    out = {}
    for name, p in (("base", H.BASE / "README_excerpt.md"), ("instruct", H.INST / "README_excerpt.md"),
                    ("our_adapter", H.DATA / "our_adapter_example" / "README.md")):
        out[name] = H.read_front_matter(p)
    out["base_spec_lines"] = H.card_body_lines(H.BASE / "README_excerpt.md")
    out["instruct_spec_lines"] = H.card_body_lines(H.INST / "README_excerpt.md")
    return out


def e3_config():
    b, i = H.load_json(H.BASE / "config.json"), H.load_json(H.INST / "config.json")
    tc = H.load_sources()["tokenizer_json_counts"]
    st = H.special_tokens(H.load_json(H.BASE / "tokenizer_config.json"))
    return {"base_readings": H.config_readings(b), "diff_base_vs_instruct": H.dict_diff(b, i),
            "keys": len(b), "tokenizer_total": tc["total"], "vocab_size": b["vocab_size"],
            "unused_rows": b["vocab_size"] - tc["total"], "max_special_id": max(st["table"]),
            "vocab_size_div_128": b["vocab_size"] / 128}


def e4_tokenizer_gen():
    tb, ti = H.load_json(H.BASE / "tokenizer_config.json"), H.load_json(H.INST / "tokenizer_config.json")
    sb, si = H.special_tokens(tb), H.special_tokens(ti)
    gb, gi = H.load_json(H.BASE / "generation_config.json"), H.load_json(H.INST / "generation_config.json")
    drop = {"added_tokens_decoder", "chat_template"}
    return {"base": {k: v for k, v in sb.items() if k != "table"}, "instruct": {k: v for k, v in si.items() if k != "table"},
            "n_special": len(sb["table"]),
            "tokenizer_config_diff": H.dict_diff({k: v for k, v in tb.items() if k not in drop},
                                                 {k: v for k, v in ti.items() if k not in drop}),
            "chat_template_same": tb["chat_template"] == ti["chat_template"],
            "chat_template_lines": len(ti["chat_template"].splitlines()),
            "default_system": {"base": H.default_system_in_template(tb["chat_template"]),
                               "instruct": H.default_system_in_template(ti["chat_template"])},
            "generation_config": {"base": gb, "instruct": gi},
            "generation_config_diff": H.dict_diff(gb, gi)}


def e5_template():
    fx = H.load_fixture()
    msgs = H.load_json(H.DATA / "messages_example.json")["cases"]
    checks = {}
    for key, case in fx["cases"].items():
        model, name, agp = key.split("|")
        ds = H.QWEN25_DEFAULT_SYSTEM["base" if model == "base" else "instruct"]
        mine = H.render_qwen25(msgs[name], agp == "True", ds)
        checks[key] = {"matches_jinja2": mine == case["text"], "n_tokens": case["n_tokens"], "chars": len(mine),
                       "im_start": H.count_marker(mine, "<|im_start|>"), "im_end": H.count_marker(mine, "<|im_end|>")}
    p = fx["pieces"]
    full = fx["cases"]["inst|with_system|False"]["n_tokens"]
    return {"jinja2": fx["jinja2"], "tokenizers": fx["tokenizers"], "checks": checks,
            "all_match": all(c["matches_jinja2"] for c in checks.values()),
            "text_inst_with_system": fx["cases"]["inst|with_system|False"]["text"],
            "default_system_tokens": {"base": p["default_system_base"], "instruct": p["default_system_inst"]},
            "assistant_content_tokens": p["assistant_content"], "learned_span_tokens": p["learned_span_in_context"],
            "full_tokens": full, "learned_share": round(p["learned_span_in_context"] / full, 4),
            "not_learned_tokens": full - p["learned_span_in_context"],
            "gen_prompt_extra_tokens": fx["cases"]["inst|with_system|True"]["n_tokens"] - full}


def e6_map():
    s = H.load_sources()
    deps = dict(s["requires_dist"])
    deps.pop("transformers_extra_torch")
    return {"stages": H.stage_map(s), "dep_layers": H.topo_order(deps),
            "versions": {l["name"]: l["version"] for l in s["libraries"]}}


def e7_toy_safetensors():
    tensors = {"embed.weight": ("F16", [3, 2], [0.5, -1.0, 0.25, 2.0, 0.1, 0.0]),
               "norm.weight": ("F32", [2], [1.0, 1.0])}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "toy.safetensors"
        n = H.write_safetensors(p, tensors, {"format": "pt"})
        n2, h, meta = H.read_safetensors_header(p)
        back = H.read_tensor(p, "embed.weight")
        size = p.stat().st_size
    return {"header_len": n, "header": h, "metadata": meta, "file_size": size,
            "file_size_check": size == 8 + n + 6 * 2 + 2 * 4, "embed_back": back}


def main():
    res = {"meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                    "checked": "2026-10-11", "sources": H.SOURCES_FILE.name, "fixture": H.FIXTURE_FILE.name},
           "e0_hand": e0_hand(), "e1_shards": e1_shards(), "e2_cards": e2_cards(), "e3_config": e3_config(),
           "e4_tokenizer_gen": e4_tokenizer_gen(), "e5_template": e5_template(), "e6_map": e6_map(),
           "e7_toy": e7_toy_safetensors()}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
