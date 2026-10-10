#!/usr/bin/env bash
# 7장 실패 예제: 세 단계를 파일 이름 규약으로 잇는 스크립트.
# 사용법: bash run_all.sh <작업 폴더> <seed> <shift>
# 일부러 'set -e'를 쓰지 않았다. 실패 예제 B에서 CH07_STRICT 값으로 세 가지 모드를 비교한다.
#   CH07_STRICT=""         아무 옵션 없음: 앞 단계가 실패해도 다음 단계로 간다.
#   CH07_STRICT="e"        set -e: 실패하면 멈춘다. 단, 로그를 남기려고 '| tee'로 이은 명령은 마지막(tee)의 성공만 본다.
#   CH07_STRICT="pipefail" set -eo pipefail: 파이프 중간의 실패도 잡는다.
case "${CH07_STRICT:-}" in
  e) set -e ;;
  pipefail) set -eo pipefail ;;
esac
W="$1"; SEED="$2"; SHIFT="$3"
PY="${PY:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$W"
"$PY" "$HERE/pipelines_ch07_steps.py" prepare --seed "$SEED" --shift "$SHIFT" --outdir "$W" 2>&1 | tee "$W/prepare.log"
"$PY" "$HERE/pipelines_ch07_steps.py" train --data "$W/train.npz" --out "$W/model.json"
"$PY" "$HERE/pipelines_ch07_steps.py" evaluate --model "$W/model.json" --data "$W/valid.npz" --out "$W/metrics.json"
