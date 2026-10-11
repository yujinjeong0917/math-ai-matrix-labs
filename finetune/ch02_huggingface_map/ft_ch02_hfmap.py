"""파인튜닝 트랙 2장: Hugging Face 저장소 파일을 네트워크 없이 읽는다.

- 모델 카드(README.md) 맨 위 YAML 머리말 읽기(이 장의 파일에 필요한 만큼만 읽는 작은 파서)
- config.json, generation_config.json, tokenizer_config.json 읽고 두 모델 비교
- safetensors 헤더 읽기(앞 8바이트 = 헤더 길이, 그다음 JSON), 작은 safetensors 파일 쓰기
- Qwen2.5 chat_template의 도구(tools)를 쓰지 않는 갈래를 표준 라이브러리로 다시 짜기
- 라이브러리 의존 관계를 위상 정렬로 줄 세우기

표준 라이브러리만 쓴다.
"""

import json
import struct
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
BASE = DATA / "qwen2.5-7b"
INST = DATA / "qwen2.5-7b-instruct"
SOURCES_FILE = DATA / "sources_2026-10-11.json"
FIXTURE_FILE = DATA / "render_fixture_2026-10-11.json"

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_sources():
    return load_json(SOURCES_FILE)


def load_fixture():
    return load_json(FIXTURE_FILE)


# ---------- 모델 카드 ----------

def _scalar(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def read_front_matter(path):
    """README.md 맨 위 '---'와 '---' 사이를 읽어 사전으로 돌려준다.

    이 장의 카드에 나오는 두 모양만 읽는다: `키: 값` 과 `키:` 다음 줄들의 `- 항목`.
    중첩된 사전 같은 다른 YAML 문법은 읽지 않는다(실제로는 PyYAML 같은 파서를 쓴다).
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta, key = {}, None
    for line in lines[1:]:
        if line.strip() == "---":
            return meta
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.lstrip().startswith("- ") and key is not None:
            meta[key].append(_scalar(line.lstrip()[2:]))
            continue
        k, _, v = line.partition(":")
        key = k.strip()
        meta[key] = _scalar(v) if v.strip() else []
    raise ValueError("머리말이 '---'로 닫히지 않았어요")


def card_body_lines(path, prefix="- "):
    """머리말 뒤 본문에서 prefix로 시작하는 줄(예: 사양 목록)을 돌려준다."""
    text = Path(path).read_text(encoding="utf-8")
    body = text.split("---", 2)[2]
    return [ln.strip() for ln in body.splitlines() if ln.strip().startswith(prefix)]


# ---------- config.json 읽기 ----------

def config_readings(cfg):
    """config.json에서 바로 셀 수 있는 값들."""
    h, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    return {
        "head_dim": h // nh,
        "q_heads_per_kv_head": nh // nkv,
        "embed_shape": [cfg["vocab_size"], h],
        "embed_numbers": cfg["vocab_size"] * h,
        "tied": cfg["tie_word_embeddings"],
    }


def dict_diff(a, b):
    """두 사전에서 값이 다르거나 한쪽에만 있는 키를 {키: [a값, b값]}으로."""
    out = {}
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "(없음)"), b.get(k, "(없음)")
        if va != vb:
            out[k] = [va, vb]
    return out


def special_tokens(tok_cfg):
    """tokenizer_config.json의 특수 토큰 번호 표와 eos/pad/bos 이름."""
    table = {int(i): v["content"] for i, v in tok_cfg["added_tokens_decoder"].items()}
    by_name = {v: k for k, v in table.items()}
    return {
        "table": table,
        "eos": tok_cfg["eos_token"], "eos_id": by_name.get(tok_cfg["eos_token"]),
        "pad": tok_cfg["pad_token"], "pad_id": by_name.get(tok_cfg["pad_token"]),
        "bos": tok_cfg["bos_token"], "add_bos_token": tok_cfg.get("add_bos_token"),
        "model_max_length": tok_cfg["model_max_length"],
    }


# ---------- safetensors ----------

def read_safetensors_header(path):
    """앞 8바이트(부호 없는 리틀엔디언 64비트 정수 N)와 이어지는 N바이트 JSON을 읽는다."""
    raw = Path(path).read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n].decode("utf-8"))
    meta = header.pop("__metadata__", None)
    return n, header, meta


def numel(shape):
    k = 1
    for s in shape:
        k *= s
    return k


def header_stats(header):
    """텐서 개수, 숫자 개수, 바이트, 형식, 구멍 없이 이어지는지."""
    items = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    pos, contiguous, size_ok = 0, True, True
    params = nbytes = 0
    dtypes = {}
    for name, t in items:
        b, e = t["data_offsets"]
        if b != pos:
            contiguous = False
        pos = e
        k = numel(t["shape"])
        if k * DTYPE_BYTES[t["dtype"]] != e - b:
            size_ok = False
        params += k
        nbytes += e - b
        dtypes[t["dtype"]] = dtypes.get(t["dtype"], 0) + 1
    return {"tensors": len(items), "params": params, "bytes": nbytes, "end": pos,
            "dtypes": dtypes, "contiguous": contiguous, "size_matches_shape": size_ok}


def write_safetensors(path, tensors, metadata=None):
    """작은 safetensors 파일을 쓴다. tensors: {이름: (dtype, shape, 값 목록)}. F16은 struct 'e', F32는 'f'."""
    fmt = {"F16": "e", "F32": "f"}
    header, chunks, pos = {}, [], 0
    for name, (dtype, shape, values) in tensors.items():
        assert len(values) == numel(shape)
        data = struct.pack("<" + fmt[dtype] * len(values), *values)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [pos, pos + len(data)]}
        chunks.append(data)
        pos += len(data)
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)          # 공백으로 8의 배수에 맞춘다(형식이 허용하는 채우기)
    Path(path).write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(chunks))
    return len(hb)


def read_tensor(path, name):
    """헤더의 data_offsets로 텐서 값을 꺼낸다(F16·F32만)."""
    n, header, _ = read_safetensors_header(path)
    t = header[name]
    b, e = t["data_offsets"]
    raw = Path(path).read_bytes()[8 + n + b: 8 + n + e]
    fmt = {"F16": "e", "F32": "f"}[t["dtype"]]
    return list(struct.unpack("<" + fmt * (len(raw) // struct.calcsize(fmt)), raw))


def shard_headers(repo=BASE):
    out = {}
    for p in sorted((repo / "safetensors_headers").glob("*.header.bin")):
        n, h, meta = read_safetensors_header(p)
        out[p.name.replace(".header.bin", ".safetensors")] = {"n": n, "header": h, "meta": meta}
    return out


# ---------- chat_template ----------

QWEN25_DEFAULT_SYSTEM = {
    "base": "You are a helpful assistant.",
    "instruct": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
}


def render_qwen25(messages, add_generation_prompt=False, default_system=QWEN25_DEFAULT_SYSTEM["instruct"]):
    """Qwen2.5 chat_template에서 tools가 없고 tool_calls·tool 메시지도 없는 갈래만 옮긴 것.

    1) 첫 메시지가 system이면 그 내용으로, 아니면 기본 system 문장으로 system 칸을 연다.
    2) 메시지마다 '<|im_start|>역할\\n내용<|im_end|>\\n'. 첫 system은 1)에서 이미 썼으니 건너뛴다.
    3) add_generation_prompt면 끝에 '<|im_start|>assistant\\n'을 붙인다.
    """
    out = []
    if messages and messages[0]["role"] == "system":
        out.append("<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n")
    else:
        out.append("<|im_start|>system\n" + default_system + "<|im_end|>\n")
    for i, m in enumerate(messages):
        if m["role"] == "system" and i == 0:
            continue
        if m["role"] in ("user", "assistant", "system") and not m.get("tool_calls"):
            out.append("<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>" + "\n")
        else:
            raise NotImplementedError("tool_calls·tool 메시지 갈래는 옮기지 않았어요")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
    return "".join(out)


def default_system_in_template(chat_template):
    """템플릿 글에서 '<|im_start|>system\\n...<|im_end|>' 기본 문장을 찾아 돌려준다."""
    key = "'<|im_start|>system\\nYou are"
    i = chat_template.index(key) + len("'<|im_start|>system\\n")
    j = chat_template.index("<|im_end|>", i)
    return chat_template[i:j]


def count_marker(text, marker):
    return text.count(marker)


# ---------- 라이브러리 지도 ----------

def topo_order(deps):
    """의존 관계 {이름: [먼저 있어야 하는 것들]}를 아래층부터 줄 세운다. 같은 층은 이름 순."""
    deps = {k: list(v) for k, v in deps.items() if not k.startswith("_")}
    for v in list(deps.values()):
        for d in v:
            deps.setdefault(d, [])
    layers, done = [], set()
    while len(done) < len(deps):
        layer = sorted(k for k, v in deps.items() if k not in done and all(d in done for d in v))
        if not layer:
            raise ValueError("의존 관계에 고리가 있어요")
        layers.append(layer)
        done |= set(layer)
    return layers


def stage_map(sources):
    """단계마다 맡는 라이브러리 이름."""
    out = {s["id"]: {"label": s["label"], "chapter": s["chapter"], "libs": []} for s in sources["stages"]}
    for lib in sources["libraries"]:
        out[lib["stage"]]["libs"].append(lib["name"])
    return out
