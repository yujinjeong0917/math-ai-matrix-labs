"""8장 최소 메타데이터 저장소: 표준 라이브러리 sqlite3만 쓴다.

ML Metadata(MLMD)의 데이터 모델을 흉내 냈다(TFX 가이드 "ML Metadata", 확인 2026-10-08).
- artifact: 결과 파일 하나(데이터, 모델, 지표). 위치(uri), 내용 지문(sha256), 속성.
- execution: task 한 번의 실행. 파라미터, 이미지, 상태(COMPLETE / CACHED).
- event: artifact와 execution 사이의 관계. INPUT(실행이 읽음) 또는 OUTPUT(실행이 내놓음).
MLMD 문서의 말대로 event를 모두 보면 "any artifact"에서 "all of its upstream inputs"까지 거꾸로 따라갈 수 있다.
MLMD도 SQLite를 기본 구현으로 제공한다. 여기 코드는 MLMD가 아니고, 같은 모양을 작게 만든 것이다.

cache 표는 KFP 2.17.2 백엔드의 캐시 조회를 흉내 낸다. 지문(fingerprint)이 같고, 파이프라인 이름과
namespace가 같은 실행 가운데 가장 최근 것 하나를 돌려준다(backend/src/v2/cacheutils/cache.go의 GetExecutionCache).
"""

import json
import sqlite3
from collections import deque

SCHEMA = """
CREATE TABLE artifacts (id INTEGER PRIMARY KEY, type TEXT, uri TEXT, sha TEXT, props TEXT);
CREATE TABLE executions (id INTEGER PRIMARY KEY, run_id TEXT, pipeline TEXT, namespace TEXT, task TEXT,
                         state TEXT, fingerprint TEXT, cached_from INTEGER, props TEXT);
CREATE TABLE events (artifact_id INTEGER, execution_id INTEGER, kind TEXT, key TEXT);
CREATE TABLE cache (fingerprint TEXT, pipeline TEXT, namespace TEXT, execution_id INTEGER, seq INTEGER);
CREATE INDEX ev_art ON events(artifact_id);
CREATE INDEX ev_exe ON events(execution_id);
CREATE INDEX cache_fp ON cache(fingerprint, pipeline, namespace);
"""


class Store:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self._seq = 0

    # ---------- 쓰기 ----------
    def put_artifact(self, type_, uri, sha, props):
        cur = self.db.execute("INSERT INTO artifacts(type, uri, sha, props) VALUES (?,?,?,?)",
                              (type_, str(uri), sha, json.dumps(props, sort_keys=True)))
        return cur.lastrowid

    def put_execution(self, run_id, pipeline, namespace, task, state, fingerprint, props, cached_from=None):
        cur = self.db.execute(
            "INSERT INTO executions(run_id, pipeline, namespace, task, state, fingerprint, cached_from, props) VALUES (?,?,?,?,?,?,?,?)",
            (run_id, pipeline, namespace, task, state, fingerprint, cached_from, json.dumps(props, sort_keys=True)))
        return cur.lastrowid

    def put_event(self, artifact_id, execution_id, kind, key):
        assert kind in ("INPUT", "OUTPUT")
        self.db.execute("INSERT INTO events VALUES (?,?,?,?)", (artifact_id, execution_id, kind, key))

    def put_cache(self, fingerprint, pipeline, namespace, execution_id):
        self._seq += 1
        self.db.execute("INSERT INTO cache VALUES (?,?,?,?,?)", (fingerprint, pipeline, namespace, execution_id, self._seq))

    # ---------- 읽기 ----------
    def get_cache(self, fingerprint, pipeline, namespace):
        row = self.db.execute(
            "SELECT execution_id FROM cache WHERE fingerprint=? AND pipeline=? AND namespace=? ORDER BY seq DESC LIMIT 1",
            (fingerprint, pipeline, namespace)).fetchone()
        return row[0] if row else None

    def artifact(self, aid):
        r = self.db.execute("SELECT id, type, uri, sha, props FROM artifacts WHERE id=?", (aid,)).fetchone()
        return {"id": r[0], "type": r[1], "uri": r[2], "sha": r[3], "props": json.loads(r[4])}

    def execution(self, eid):
        r = self.db.execute("SELECT id, run_id, pipeline, namespace, task, state, fingerprint, cached_from, props "
                            "FROM executions WHERE id=?", (eid,)).fetchone()
        return {"id": r[0], "run_id": r[1], "pipeline": r[2], "namespace": r[3], "task": r[4], "state": r[5],
                "fingerprint": r[6], "cached_from": r[7], "props": json.loads(r[8])}

    def outputs_of(self, eid):
        return dict(self.db.execute("SELECT key, artifact_id FROM events WHERE execution_id=? AND kind='OUTPUT'", (eid,)).fetchall())

    def inputs_of(self, eid):
        return dict(self.db.execute("SELECT key, artifact_id FROM events WHERE execution_id=? AND kind='INPUT'", (eid,)).fetchall())

    def producers(self, aid):
        """이 artifact를 OUTPUT으로 가진 실행들. 캐시로 다시 쓴 실행(CACHED)도 OUTPUT으로 걸리므로 여러 개일 수 있다."""
        return [r[0] for r in self.db.execute(
            "SELECT execution_id FROM events WHERE artifact_id=? AND kind='OUTPUT' ORDER BY execution_id", (aid,))]

    def counts(self):
        return {t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("artifacts", "executions", "events")}

    # ---------- 리니지 ----------
    def ancestors_sql(self, aid):
        """모델 artifact의 조상 집합: 재귀 CTE 한 번. (artifact id 집합, execution id 집합)을 돌려준다."""
        rows = self.db.execute("""
            WITH RECURSIVE up(kind, id) AS (
                SELECT 'a', ?
                UNION
                SELECT 'e', ev.execution_id FROM up JOIN events ev ON up.kind='a' AND ev.artifact_id=up.id AND ev.kind='OUTPUT'
                UNION
                SELECT 'a', ev.artifact_id FROM up JOIN events ev ON up.kind='e' AND ev.execution_id=up.id AND ev.kind='INPUT'
            ) SELECT kind, id FROM up""", (aid,)).fetchall()
        return ({i for k, i in rows if k == "a"}, {i for k, i in rows if k == "e"})

    def ancestors_bfs(self, aid):
        """같은 일을 파이썬 너비 우선 탐색으로. 테스트에서 재귀 CTE와 결과를 맞춰 본다."""
        arts, exes, q = {aid}, set(), deque([("a", aid)])
        while q:
            kind, i = q.popleft()
            if kind == "a":
                for e in self.producers(i):
                    if e not in exes:
                        exes.add(e)
                        q.append(("e", e))
            else:
                for a in self.inputs_of(i).values():
                    if a not in arts:
                        arts.add(a)
                        q.append(("a", a))
        return arts, exes

    def descendants_sql(self, aid):
        """반대 방향: 이 artifact를 읽은 실행과 그 실행이 낸 artifact 전부(문제 데이터의 영향 범위)."""
        rows = self.db.execute("""
            WITH RECURSIVE down(kind, id) AS (
                SELECT 'a', ?
                UNION
                SELECT 'e', ev.execution_id FROM down JOIN events ev ON down.kind='a' AND ev.artifact_id=down.id AND ev.kind='INPUT'
                UNION
                SELECT 'a', ev.artifact_id FROM down JOIN events ev ON down.kind='e' AND ev.execution_id=down.id AND ev.kind='OUTPUT'
            ) SELECT kind, id FROM down""", (aid,)).fetchall()
        return ({i for k, i in rows if k == "a"}, {i for k, i in rows if k == "e"})

    def trace(self, model_artifact_id):
        """운영 중인 모델 하나에서 출발해 '어떤 코드·데이터·파라미터로 만들었나'를 한 장으로 정리한다."""
        arts, exes = self.ancestors_sql(model_artifact_id)
        model = self.artifact(model_artifact_id)
        prods = [self.execution(e) for e in self.producers(model_artifact_id)]
        made = [p for p in prods if p["state"] == "COMPLETE"]
        reused = [p for p in prods if p["state"] == "CACHED"]
        train = made[0]
        data_id = self.inputs_of(train["id"])["train_data"]
        data = self.artifact(data_id)
        prep = self.execution([e for e in self.producers(data_id) if self.execution(e)["state"] == "COMPLETE"][0])
        return {
            "model": {"id": model["id"], "sha": model["sha"][:12]},
            "made_by": {"run_id": train["run_id"], "execution_id": train["id"], "params": train["props"]["params"],
                        "image_ref": train["props"]["image_ref"], "image_digest": train["props"]["image_digest"]},
            "reused_by_runs": [p["run_id"] for p in reused],
            "train_data": {"id": data_id, "sha": data["sha"][:12], "dataset_hash": data["props"]["dataset_hash"],
                           "source_version": data["props"]["source_version"],
                           "source_version_from": data["props"]["source_version_from"],
                           "seed": data["props"]["seed"], "shift": data["props"]["shift"], "rows": data["props"]["rows"]},
            "prepare": {"run_id": prep["run_id"], "params": prep["props"]["params"], "image_ref": prep["props"]["image_ref"],
                        "image_digest": prep["props"]["image_digest"], "env": prep["props"]["env"]},
            "ancestor_artifacts": len(arts), "ancestor_executions": len(exes),
        }
