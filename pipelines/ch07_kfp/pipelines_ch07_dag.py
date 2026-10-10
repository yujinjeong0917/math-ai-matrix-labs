"""7장 최소 구현: 입력·출력이 선언된 컴포넌트, 그래프(DAG) 컴파일, 위상 정렬 실행기. 표준 라이브러리만 쓴다.

KFP v2를 흉내 낸 것이지 KFP가 아니다. 무엇을 같게 만들었는지만 적는다(실제 kfp 2.17.0 컴파일 결과와 비교는 테스트에서).
- @component: 함수의 타입 표시에서 파라미터(int/float/str)와 결과 파일 입력(Input[...])·출력(Output[...])을 읽는다.
- @pipeline: 함수를 "실행"하지 않고, 자리표시자를 넣어 한 번 불러서 어떤 task가 어떤 출력을 받는지만 모은다.
  없는 출력 이름을 쓰면 KeyError, 타입이 안 맞으면 TypeError가 이때(아무것도 실행하기 전에) 난다.
- compile(): KFP IR과 같은 모양의 dict. root.dag.tasks[이름]에 componentRef, dependentTasks,
  inputs.artifacts[키].taskOutputArtifact{producerTask, outputArtifactKey}, inputs.parameters[키].componentInputParameter.
- run(): 위상 정렬 순서로 task를 하나씩 돈다. 출력 위치는 <pipeline_root>/<run_id>/<task>/<출력 이름>으로
  실행마다 새로 정해 주고(KFP local 실행의 배치와 같은 모양), 각 task가 실제로 읽은 입력의 위치와 지문을 기록한다.
  task가 실패하면 그 뒤에 기대는 task는 돌리지 않는다.
일부러 뺀 것: 캐시(8장), 재시도, 조건·반복, 컨테이너. isolation="subprocess"는 task마다 새 파이썬 프로세스를 띄우는
정도까지만 흉내 낸다(컨테이너 기동 비용은 들어 있지 않다).
"""

import hashlib
import importlib
import inspect
import json
import subprocess
import sys
import time
import uuid
from collections import deque
from pathlib import Path

# ---------- 타입 표시 ----------


class Artifact:
    schema = "system.Artifact"

    def __init__(self, name, uri, metadata=None):
        self.name, self.uri, self.metadata = name, str(uri), dict(metadata or {})

    @property
    def path(self):
        return self.uri

    def log_metric(self, key, value):  # kfp의 Metrics.log_metric과 같은 이름. 여기서는 metadata에만 적는다
        self.metadata[key] = value


class Dataset(Artifact):
    schema = "system.Dataset"


class Model(Artifact):
    schema = "system.Model"


class Metrics(Artifact):
    schema = "system.Metrics"


class _Marker:
    def __init__(self, direction, cls):
        self.direction, self.cls = direction, cls


class Input:
    def __class_getitem__(cls, item):
        return _Marker("in", item)


class Output:
    def __class_getitem__(cls, item):
        return _Marker("out", item)


PARAM_TYPES = {int: "NUMBER_INTEGER", float: "NUMBER_DOUBLE", str: "STRING", bool: "BOOLEAN"}

# ---------- 컴포넌트와 파이프라인 정의 ----------

_BUILDER = []  # @pipeline을 컴파일하는 동안만 task 목록이 들어 있다


class PipelineParam:
    def __init__(self, name, ptype):
        self.name, self.ptype = name, ptype


class TaskOutput:
    def __init__(self, task, key, cls):
        self.task, self.key, self.cls = task, key, cls


class _Outputs(dict):
    def __init__(self, task, decl):
        super().__init__({k: TaskOutput(task, k, c) for k, c in decl.items()})
        self.task = task

    def __missing__(self, key):
        raise KeyError(f"{key!r}: task '{self.task.name}'에는 이 출력이 없다. 있는 출력: {sorted(self)}")


class Task:
    def __init__(self, comp, name, args):
        self.comp, self.name, self.args = comp, name, args
        self.outputs = _Outputs(self, comp.out_artifacts)


class Component:
    def __init__(self, fn, image):
        self.fn, self.name, self.image = fn, fn.__name__, image
        self.module, self.qualname = fn.__module__, fn.__qualname__
        self.params, self.in_artifacts, self.out_artifacts, self.defaults = {}, {}, {}, {}
        for k, p in inspect.signature(fn).parameters.items():
            a = p.annotation
            if isinstance(a, _Marker):
                (self.in_artifacts if a.direction == "in" else self.out_artifacts)[k] = a.cls
            elif a in PARAM_TYPES:
                self.params[k] = a
            else:
                raise TypeError(f"{self.name}.{k}: 타입 표시가 없거나 지원하지 않는 타입이다")
            if p.default is not inspect.Parameter.empty:
                self.defaults[k] = p.default

    def __call__(self, **kwargs):
        if not _BUILDER:
            raise RuntimeError("컴포넌트는 @pipeline 안에서만 부를 수 있다(여기서는 실행하지 않고 그래프만 만든다)")
        tasks = _BUILDER[-1]
        for k, v in kwargs.items():
            if k in self.in_artifacts:
                if not isinstance(v, TaskOutput):
                    raise TypeError(f"{self.name}.{k}: 결과 파일 입력에는 다른 task의 출력을 이어야 한다")
                want, got = self.in_artifacts[k], v.cls
                if not (want is Artifact or issubclass(got, want)):
                    raise TypeError(f"{self.name}.{k}: {got.schema}를 {want.schema} 입력에 이을 수 없다")
            elif k in self.params:
                if isinstance(v, TaskOutput):
                    raise TypeError(f"{self.name}.{k}: {v.cls.schema}를 파라미터({PARAM_TYPES[self.params[k]]})에 이을 수 없다")
                if isinstance(v, PipelineParam) and v.ptype is not self.params[k]:
                    raise TypeError(f"{self.name}.{k}: 파라미터 타입이 다르다")
            else:
                raise TypeError(f"{self.name}: 모르는 입력 {k!r}")
        missing = [k for k in list(self.in_artifacts) + list(self.params) if k not in kwargs and k not in self.defaults]
        if missing:
            raise TypeError(f"{self.name}: 이어지지 않은 입력 {missing}")
        name = self.name
        used = {t.name for t in tasks}
        i = 2
        while name in used:
            name, i = f"{self.name}-{i}", i + 1
        t = Task(self, name, kwargs)
        tasks.append(t)
        return t


def component(fn=None, *, base_image="python:3.12"):
    """kfp의 @dsl.component와 같은 인자 이름(base_image)을 쓴다."""
    if fn is None:
        return lambda f: Component(f, base_image)
    return Component(fn, base_image)


class Pipeline:
    def __init__(self, fn, name):
        self.fn, self.name = fn, name or fn.__name__.replace("_", "-")
        self.params = {k: (p.annotation, p.default) for k, p in inspect.signature(fn).parameters.items()}
        self.build()  # kfp의 @dsl.pipeline처럼 정의하는 순간 한 번 불러서, 연결 실수를 실행 전에 드러낸다

    def build(self):
        _BUILDER.append([])
        try:
            self.fn(**{k: PipelineParam(k, t) for k, (t, _) in self.params.items()})
            return _BUILDER[-1]
        finally:
            _BUILDER.pop()


def pipeline(fn=None, *, name=None):
    if fn is None:
        return lambda f: Pipeline(f, name)
    return Pipeline(fn, name)


# ---------- 컴파일 ----------


def _defs(arts, params=None, defaults=None):
    out = {}
    if arts:
        out["artifacts"] = {k: {"artifactType": {"schemaTitle": c.schema}} for k, c in arts.items()}
    if params:
        out["parameters"] = {k: {"parameterType": PARAM_TYPES[t], **({"defaultValue": defaults[k]} if k in defaults else {})}
                             for k, t in params.items()}
    return out


def compile(pl):  # noqa: A001  (kfp의 compiler.compile과 같은 이름을 일부러 썼다)
    tasks = pl.build()
    comps, execs, dag = {}, {}, {}
    for t in tasks:
        c = t.comp
        comps[f"comp-{c.name}"] = {
            "executorLabel": f"exec-{c.name}",
            "inputDefinitions": _defs(c.in_artifacts, c.params, c.defaults),
            "outputDefinitions": _defs(c.out_artifacts),
        }
        execs[f"exec-{c.name}"] = {"container": {"image": c.image}, "python": {"module": c.module, "function": c.qualname}}
        arts, params, deps = {}, {}, set()
        for k, v in t.args.items():
            if isinstance(v, TaskOutput):
                arts[k] = {"taskOutputArtifact": {"producerTask": v.task.name, "outputArtifactKey": v.key}}
                deps.add(v.task.name)
            elif isinstance(v, PipelineParam):
                params[k] = {"componentInputParameter": v.name}
            else:
                params[k] = {"runtimeValue": {"constant": v}}
        spec = {"componentRef": {"name": f"comp-{c.name}"}, "taskInfo": {"name": t.name}, "inputs": {}}
        if arts:
            spec["inputs"]["artifacts"] = arts
        if params:
            spec["inputs"]["parameters"] = params
        if deps:
            spec["dependentTasks"] = sorted(deps)
        dag[t.name] = spec
    pdefs = {k: {"parameterType": PARAM_TYPES[t], "defaultValue": d} for k, (t, d) in pl.params.items()}
    return {
        "pipelineInfo": {"name": pl.name},
        "components": comps,
        "deploymentSpec": {"executors": execs},
        "root": {"dag": {"tasks": dag}, "inputDefinitions": {"parameters": pdefs}},
    }


def edges(ir):
    """(만든 task, 출력 이름, 받는 task, 입력 이름) 목록."""
    out = []
    for name, spec in ir["root"]["dag"]["tasks"].items():
        for k, v in spec.get("inputs", {}).get("artifacts", {}).items():
            src = v["taskOutputArtifact"]
            out.append((src["producerTask"], src["outputArtifactKey"], name, k))
    return sorted(out)


def toposort(ir):
    """Kahn 알고리즘. 이름순으로 깨서 순서가 항상 같다. 사이클이 있으면 ValueError."""
    tasks = ir["root"]["dag"]["tasks"]
    indeg = {n: len(s.get("dependentTasks", [])) for n, s in tasks.items()}
    children = {n: [] for n in tasks}
    for n, s in tasks.items():
        for d in s.get("dependentTasks", []):
            children[d].append(n)
    ready = deque(sorted(n for n, d in indeg.items() if d == 0))
    order = []
    while ready:
        n = ready.popleft()
        order.append(n)
        for c in sorted(children[n]):
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if len(order) != len(tasks):
        raise ValueError("사이클이 있어 순서를 정할 수 없다")
    return order


# ---------- 실행 ----------


def _sha(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.is_file() else None


def _call(module, function, kwargs, isolation):
    if isolation == "inprocess":
        fn = getattr(importlib.import_module(module), function)
        fn = getattr(fn, "fn", fn)
        return fn(**kwargs)
    code = (
        "import json,sys,importlib;from pipelines_ch07_dag import Artifact\n"
        "a=json.loads(sys.argv[1]);m=importlib.import_module(a['module']);f=getattr(m,a['function']);f=getattr(f,'fn',f)\n"
        "kw={k:(Artifact(v['name'],v['uri']) if isinstance(v,dict) and v.get('_artifact') else v) for k,v in a['kwargs'].items()}\n"
        "f(**kw)\n"
    )
    enc = {k: ({"_artifact": True, "name": v.name, "uri": v.uri} if isinstance(v, Artifact) else v) for k, v in kwargs.items()}
    here = str(Path(__file__).parent)
    r = subprocess.run([sys.executable, "-c", code, json.dumps({"module": module, "function": function, "kwargs": enc})],
                       cwd=here, capture_output=True, text=True, env={**_env(), "PYTHONPATH": here})
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f"exit {r.returncode}")


def _env():
    import os
    return dict(os.environ)


def run_steps(ir, arguments, pipeline_root, run_id=None, isolation="inprocess"):
    """task를 하나 끝낼 때마다 그 기록을 내놓는 생성기. 두 run을 사람이 정한 순서로 섞어 돌릴 때 쓴다."""
    run_id = run_id or uuid.uuid4().hex[:8]
    tasks, comps = ir["root"]["dag"]["tasks"], ir["components"]
    execs = ir["deploymentSpec"]["executors"]
    pdefs = ir["root"]["inputDefinitions"]["parameters"]
    params = {k: arguments.get(k, d["defaultValue"]) for k, d in pdefs.items()}
    produced, failed = {}, set()
    for name in toposort(ir):
        spec = tasks[name]
        comp = comps[spec["componentRef"]["name"]]
        ex = execs[comp["executorLabel"]]["python"]
        rec = {"run_id": run_id, "task": name, "inputs": {}, "outputs": {}}
        if any(d in failed for d in spec.get("dependentTasks", [])):
            failed.add(name)
            rec["status"] = "SKIPPED"
            yield rec
            continue
        kwargs = {}
        for k, v in spec.get("inputs", {}).get("parameters", {}).items():
            kwargs[k] = params[v["componentInputParameter"]] if "componentInputParameter" in v else v["runtimeValue"]["constant"]
        for k, v in spec.get("inputs", {}).get("artifacts", {}).items():
            src = v["taskOutputArtifact"]
            art = produced[(src["producerTask"], src["outputArtifactKey"])]
            kwargs[k] = art
            rec["inputs"][k] = {"uri": art.uri, "sha": _sha(art.uri)}
        for k, d in comp.get("outputDefinitions", {}).get("artifacts", {}).items():
            uri = Path(pipeline_root) / run_id / name / k
            uri.parent.mkdir(parents=True, exist_ok=True)
            kwargs[k] = Artifact(k, uri)
        t0 = time.perf_counter()
        try:
            _call(ex["module"], ex["function"], kwargs, isolation)
            rec["status"] = "SUCCEEDED"
            for k in comp.get("outputDefinitions", {}).get("artifacts", {}):
                produced[(name, k)] = kwargs[k]
                rec["outputs"][k] = {"uri": kwargs[k].uri, "sha": _sha(kwargs[k].uri)}
        except Exception as e:  # noqa: BLE001
            failed.add(name)
            rec["status"] = "FAILED"
            rec["error"] = str(e)[:200]
        rec["seconds"] = time.perf_counter() - t0
        yield rec


def run(ir, arguments, pipeline_root, run_id=None, isolation="inprocess"):
    log = list(run_steps(ir, arguments, pipeline_root, run_id, isolation))
    state = "SUCCEEDED" if all(r["status"] == "SUCCEEDED" for r in log) else "FAILED"
    out = {"run_id": log[0]["run_id"] if log else run_id, "state": state, "tasks": log}
    Path(pipeline_root, out["run_id"]).mkdir(parents=True, exist_ok=True)
    Path(pipeline_root, out["run_id"], "run_log.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out
