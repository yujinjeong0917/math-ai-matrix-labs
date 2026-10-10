"""8장 실행기: 7장 그래프(prepare -> train -> evaluate)를 돌리면서 메타데이터를 남기고, 캐시를 쓴다.

7장 pipelines_ch07_dag.py의 실행 부분만 필요한 만큼 다시 썼다(장끼리 동시에 작성되므로 import하지 않는다).
그래프 정의·컴파일은 7장에서 했으니, 여기서는 KFP IR과 같은 정보를 가진 dict(PIPELINE)를 바로 쓴다.

캐시 키(cache_key)는 세 가지 정책을 비교한다.
- "kfp-2.17.2": kubeflow/pipelines 태그 2.17.2의 backend/src/v2/cacheutils/cache.go GenerateCacheKey가 담는
  필드 묶음을 따랐다. 입력 결과 파일은 내용이 아니라 MLMD artifact ID(숫자)로, 이미지는 태그가 붙은 이름 문자열로,
  명령은 컴포넌트 몸통 문자열로 들어간다. 컨테이너 환경 변수는 들어가지 않는다.
  필드 묶음을 따른 것이지, 지문이 실제 KFP와 바이트까지 같다는 뜻은 아니다.
- "kfp-master": 위와 같고 환경 변수(env)를 더한다. PR #14340(2026-09-15 master에 병합, 2.17.2 이후)의 변경이다.
- "content": 비교용 가상 정책. 입력 결과 파일의 내용 지문과 이미지 다이제스트(내용 해시)를 쓴다.
  Bazel 원격 캐시처럼 "내용으로 주소를 매기는" 쪽이다. KFP에 이런 옵션이 있다는 뜻이 아니다.
캐시가 맞으면(같은 지문이 같은 파이프라인 이름·namespace에 있으면) 그 실행의 출력 artifact를 ID 그대로 다시 쓰고,
실행은 CACHED로 적는다(2.17.2 driver/cache.go의 reuseCachedOutputs와 같은 동작).
"""

import hashlib
import json
import time
import uuid
from pathlib import Path

import pipelines_ch08_steps as S

# KFP 가벼운 컴포넌트는 함수 몸통을 명령 문자열에 그대로 넣는다(7장 results/kfp_ir_2.17.0.json의 command 참고).
# 몸통이 부르는 모듈(pipelines_ch08_steps, pipelines_ch08_data)의 코드는 이미지 안에 있고, 명령 문자열에는 없다.
PIPELINE = {
    "name": "churn-ch08",
    "params": {"seed": 1, "n_rows": 4000, "shift": 1.0, "data_version": "", "impl": "numpy", "threshold": 0.5},
    "tasks": {
        "prepare": {
            "command": "import pipelines_ch08_steps as S\nS.prepare(seed, n_rows, shift, train_data.path, valid_data.path, data_version)",
            "params": ["seed", "n_rows", "shift", "data_version"],
            "inputs": {},
            "outputs": {"train_data": "system.Dataset", "valid_data": "system.Dataset"},
        },
        "train": {
            "command": "import pipelines_ch08_steps as S\nS.train(train_data.path, model.path, impl)",
            "params": ["impl"],
            "inputs": {"train_data": ("prepare", "train_data")},
            "outputs": {"model": "system.Model"},
        },
        "evaluate": {
            "command": "import pipelines_ch08_steps as S\nS.evaluate(model.path, valid_data.path, metrics.path, threshold)",
            "params": ["threshold"],
            "inputs": {"model": ("train", "model"), "valid_data": ("prepare", "valid_data")},
            "outputs": {"metrics": "system.Metrics"},
        },
    },
    "order": ["prepare", "train", "evaluate"],
}

POLICIES = ("kfp-2.17.2", "kfp-master", "content")


class Registry:
    """이미지 레지스트리 흉내. 태그는 옮겨 붙일 수 있고, 다이제스트는 내용에서 나오므로 바뀌지 않는다."""

    def __init__(self, repo="registry.example.com/churn"):
        self.repo, self.tags, self.blobs = repo, {}, {}

    def push(self, tag, content):
        digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]
        self.blobs[digest] = dict(content)
        self.tags[tag] = digest
        return digest

    def ref(self, tag=None, digest=None):
        return f"{self.repo}@sha256:{digest}" if digest else f"{self.repo}:{tag}"

    def resolve(self, ref):
        if "@sha256:" in ref:
            d = ref.split("@sha256:")[1]
        else:
            d = self.tags[ref.rsplit(":", 1)[1]]
        return d, self.blobs[d]


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cache_key(policy, inputs, params, outputs, image_ref, image_digest, command, env):
    """inputs: 키 -> artifact dict(id, sha). 정책별로 지문에 들어가는 필드 묶음을 만든다."""
    if policy in ("kfp-2.17.2", "kfp-master"):
        container = {"image": image_ref, "cmdArgs": [command], "pvcNames": []}
        if policy == "kfp-master" and env:
            container["env"] = dict(env)
        return {
            "inputArtifactNames": {k: {"artifactNames": [str(a["id"])]} for k, a in inputs.items()},
            "inputParameterValues": dict(params),
            "outputArtifactsSpec": {k: {"type": {"schemaTitle": t}} for k, t in outputs.items()},
            "outputParametersSpec": {},
            "containerSpec": container,
        }
    if policy == "content":
        return {"inputs": {k: a["sha"] for k, a in inputs.items()}, "params": dict(params), "outputs": dict(outputs),
                "image_digest": image_digest, "command": command, "env": dict(env or {})}
    raise ValueError(policy)


def fingerprint(key):
    return _sha(key)


def run(store, registry, root, image_ref, args=None, policy="kfp-2.17.2", caching=True, env=None,
        namespace="team-a", run_id=None, pipeline=PIPELINE):
    """한 번의 run. caching은 True/False 또는 task별 dict, env는 task별 환경 변수 dict."""
    args = {**pipeline["params"], **(args or {})}
    run_id = run_id or "run-" + uuid.uuid4().hex[:8]
    produced, log = {}, []
    t_run = time.perf_counter()
    for name in pipeline["order"]:
        spec = pipeline["tasks"][name]
        use_cache = caching.get(name, True) if isinstance(caching, dict) else bool(caching)
        task_env = dict((env or {}).get(name, {}))
        params = {k: args[k] for k in spec["params"]}
        inputs = {k: store.artifact(produced[src]) for k, src in spec["inputs"].items()}
        digest, image = registry.resolve(image_ref)  # task가 뜨는 순간 이미지 이름을 레지스트리에서 찾는다
        key = cache_key(policy, inputs, params, spec["outputs"], image_ref, digest, spec["command"], task_env)
        fp = fingerprint(key) if use_cache else ""
        props = {"params": params, "image_ref": image_ref, "image_digest": digest, "env": task_env, "policy": policy}
        rec = {"task": name, "fingerprint": fp[:12]}
        t0 = time.perf_counter()
        hit = store.get_cache(fp, pipeline["name"], namespace) if use_cache else None
        if hit is not None:
            outs = store.outputs_of(hit)
            eid = store.put_execution(run_id, pipeline["name"], namespace, name, "CACHED", fp, props, cached_from=hit)
            for k, a in inputs.items():
                store.put_event(a["id"], eid, "INPUT", k)
            for k, aid in outs.items():
                store.put_event(aid, eid, "OUTPUT", k)
                produced[(name, k)] = aid
            rec["status"] = "CACHED"
        else:
            paths = {k: Path(root) / run_id / name / k for k in spec["outputs"]}
            ctx = {"image": image, "env": task_env}
            if name == "prepare":
                meta = S.prepare(params["seed"], params["n_rows"], params["shift"], paths["train_data"], paths["valid_data"],
                                 params["data_version"], ctx=ctx)
            elif name == "train":
                meta = S.train(inputs["train_data"]["uri"], paths["model"], params["impl"], ctx=ctx)
            else:
                meta = S.evaluate(inputs["model"]["uri"], inputs["valid_data"]["uri"], paths["metrics"], params["threshold"], ctx=ctx)
            eid = store.put_execution(run_id, pipeline["name"], namespace, name, "COMPLETE", fp, props)
            for k, a in inputs.items():
                store.put_event(a["id"], eid, "INPUT", k)
            for k, t in spec["outputs"].items():
                aid = store.put_artifact(t, paths[k], S.file_sha(paths[k]), meta.get(k, {}))
                store.put_event(aid, eid, "OUTPUT", k)
                produced[(name, k)] = aid
            if use_cache:
                store.put_cache(fp, pipeline["name"], namespace, eid)
            rec["status"] = "RAN"
        rec["seconds"] = time.perf_counter() - t0
        rec["execution_id"] = eid
        log.append(rec)
    out = {"run_id": run_id, "tasks": log, "seconds": time.perf_counter() - t_run,
           "artifacts": {f"{t}.{k}": aid for (t, k), aid in produced.items()}}
    return out


def metrics_of(store, run_out):
    return store.artifact(run_out["artifacts"]["evaluate.metrics"])["props"]


def data_of(store, run_out):
    return store.artifact(run_out["artifacts"]["prepare.train_data"])["props"]
