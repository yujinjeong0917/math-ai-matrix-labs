"""7장 Jenkins가 부르는 KFP run 제출 스크립트(실제 클러스터용). 이 실습 저장소에서는 실행하지 않는다.

인자 이름은 kfp 2.17.0의 kfp.Client.create_run_from_pipeline_package, wait_for_run_completion 시그니처를
kfp_local_check.py가 기록한 것(results/ch07_kfp_local.json의 client_signatures)과 테스트에서 맞춰 본다.
지표를 가져오는 부분은 클러스터마다 결과 파일 저장소(pipeline_root) 설정이 달라서 함수 하나로 비워 뒀다.
"""

import argparse
import json


def fetch_metrics(client, run_id):  # pragma: no cover - 클러스터의 결과 파일 저장소에 따라 다르다
    raise NotImplementedError("evaluate task의 metrics 결과 파일을 pipeline_root에서 읽어 오는 부분")


def main():  # pragma: no cover
    import kfp

    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default=None)
    a = ap.parse_args()
    client = kfp.Client(host=a.host)
    run = client.create_run_from_pipeline_package(
        pipeline_file=a.package,
        arguments={"seed": 1, "n_rows": 4000, "shift": 1.0},
        run_name=a.run_name,
        experiment_name="churn-ch07",
        enable_caching=False,
    )
    done = client.wait_for_run_completion(run_id=run.run_id, timeout=3600)
    if done.state != "SUCCEEDED":
        raise SystemExit(f"KFP run {run.run_id}: {done.state}")
    with open(a.out, "w") as f:
        json.dump(fetch_metrics(client, run.run_id), f)


if __name__ == "__main__":
    main()
