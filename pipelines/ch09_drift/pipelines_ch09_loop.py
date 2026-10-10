"""9장 감시·재학습·되돌리기 루프 시뮬레이션. Jenkins 서버와 KFP 클러스터는 띄우지 않는다.

역할 분담(Jenkinsfile과 같은 경계)
- Jenkins 예약 작업(cron): 배치가 끝날 때마다 드리프트 검사, 경보면 KFP 재학습 run 제출, 정답이 도착하면 판정, 되돌리기.
- KFP run: 재학습 하나만 한다(retrain()). 여기서는 함수 호출로 흉내 낸다.
- 레지스트리: 모델 바이트의 sha256을 다이제스트로 쓴다. 되돌리기는 이름(태그)이 아니라 이전 다이제스트로 한다.

시간 규칙(교육용 단순화)
- 배치 t가 끝나면 그 배치의 입력은 바로 보이지만, 정답(이탈 여부)은 label_delay 배치 뒤에 도착한다.
  즉 배치 t가 끝난 시점에 정답을 아는 배치는 t - label_delay 이하뿐이다.
- 경보가 울리면 KFP run이 학습한 새 모델이 다음 배치(t+1)부터 전부 받는다.
- 새 모델이 처음 받은 배치의 정답이 도착하면, 같은 요청에 대한 이전 모델의 그림자 예측과 decide()로 비교한다.
  rollback이면 그다음 배치부터 이전 다이제스트로 돌아간다.
- 판정이 끝나기 전(대기 중)에는 경보가 울려도 새 run을 내지 않는다.

재학습 데이터 정책
- "labeled": 정답이 도착한 가장 최근 window개 배치로만 학습.
- "naive":  가장 최근 window개 배치로 학습하되, 아직 정답이 없는 행을 '이탈 안 함(0)'으로 채운다.
            정답이 늦게 오는 것을 잊은 파이프라인이다.
"""

import hashlib

import numpy as np

import pipelines_ch09_data as D
import pipelines_ch09_drift_np as K
import pipelines_ch09_model_np as M
from pipelines_ch09_release_gate import GUARD, decide


def digest(model):
    return "sha256:" + hashlib.sha256(M.model_bytes(model)).hexdigest()


class Registry:
    def __init__(self):
        self.models = {}

    def push(self, model):
        d = digest(model)
        self.models[d] = model
        return d


def concat(tables):
    return {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}


def errors(model, table):
    return {"n": len(table[D.LABEL]), "errors": int(np.sum(M.predict(model, table) != table[D.LABEL]))}


def retrain(batches, t, label_delay, window, policy):
    """KFP 재학습 run이 하는 일. 반환: (모델, 학습 데이터, 학습에 쓴 배치 번호들). 쓸 배치가 없으면 None."""
    if policy == "labeled":
        last = t - label_delay
        idx = [i for i in range(last - window + 1, last + 1) if i >= 0]
        if not idx:
            return None
        data = concat([batches[i] for i in idx])
    elif policy == "naive":
        idx = [i for i in range(t - window + 1, t + 1) if i >= 0]
        parts = []
        for i in idx:
            b = {k: v.copy() for k, v in batches[i].items()}
            if i > t - label_delay:  # 정답이 아직 안 온 배치
                b[D.LABEL] = np.zeros_like(b[D.LABEL])
            parts.append(b)
        data = concat(parts)
    else:
        raise ValueError(policy)
    return M.train(data), data, idx


def run_loop(batches, model0, ref0, *, monitor=True, policy="labeled", label_delay=3, window=2,
             alpha=0.05, guard=GUARD, pause_after_rollback=False):
    """pause_after_rollback: 한 번 되돌렸으면 사람이 볼 때까지 자동 재학습을 멈춘다(경보는 계속 기록)."""
    reg = Registry()
    serving = reg.push(model0)
    ref = ref0
    log = {"acc": [], "serving": [], "alarms": [], "runs": [], "decisions": []}
    pending = None  # {"digest", "prev", "prev_ref", "start"}
    paused = False
    for t, batch in enumerate(batches):
        model = reg.models[serving]
        log["acc"].append(M.accuracy(model, batch))
        log["serving"].append(serving)
        if not monitor:
            continue
        # 1) 정답이 도착한 배치로 대기 중인 새 모델 판정
        if pending is not None and t - label_delay >= pending["start"]:
            b = batches[pending["start"]]
            new_m, old_m = reg.models[pending["digest"]], reg.models[pending["prev"]]
            verdict = decide(errors(old_m, b), errors(new_m, b), guard)
            log["decisions"].append({"t": t, "judged_batch": pending["start"], "verdict": verdict,
                                     "acc_new": M.accuracy(new_m, b), "acc_old": M.accuracy(old_m, b)})
            if verdict == "rollback":
                serving, ref = pending["prev"], pending["prev_ref"]
                paused = pause_after_rollback
            pending = None
        # 2) 드리프트 검사(입력만 본다)
        chk = K.ks_check(ref, D.numeric_matrix(batch), alpha=alpha, names=D.NUMERIC)
        if chk["alarm"]:
            log["alarms"].append(t)
        # 3) 경보면 재학습 run 제출 -> 다음 배치부터 새 모델
        if chk["alarm"] and pending is None and not paused and t + 1 < len(batches):
            out = retrain(batches, t, label_delay, window, policy)
            if out is None:  # 정답이 도착한 배치가 아직 없다
                continue
            new_model, data, idx = out
            d = reg.push(new_model)
            log["runs"].append({"t": t, "train_batches": idx, "digest": d[:19]})
            pending = {"digest": d, "prev": serving, "prev_ref": ref, "start": t + 1}
            serving, ref = d, D.numeric_matrix(data)
    return log
