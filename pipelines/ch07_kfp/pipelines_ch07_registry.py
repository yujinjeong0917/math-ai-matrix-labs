"""7장 이미지 태그 시뮬레이션: Jenkins가 시험한 이미지와 KFP task가 실제로 돈 이미지가 같은가.

규칙(교육용 단순화)
- 커밋이 지수분포 간격(평균 commit_gap분)으로 들어온다. 커밋마다 Jenkins가 build_min분 동안 테스트·빌드하고,
  끝나면 이미지를 레지스트리에 올리며 태그 두 개를 붙인다: 커밋 SHA 태그(다시 안 바뀜)와 'latest'(새 이미지로 옮겨 감).
- 그 직후 KFP run을 제출한다. run은 queue_min분(지수분포 평균) 기다렸다가 task 세 개를 차례로 띄운다.
  task 사이 간격은 task_min분. task가 뜨는 순간 이미지 이름을 레지스트리에서 찾는다.
  ':latest'는 그 순간 latest가 가리키는 이미지, ':SHA'와 '@digest'는 제출할 때 적은 그 이미지.
- 쿠버네티스 문서대로 ':latest'의 기본 imagePullPolicy는 Always라서 매번 레지스트리를 본다고 둔다.
잰 것: 시험한 이미지와 다른 이미지로 돈 task가 하나라도 있는 run의 비율, 한 run 안에서 task마다 이미지가 다른 비율.
"""

import numpy as np


def simulate(ref_mode, commit_gap, n_commits=2000, build_min=8.0, queue_min=5.0, task_min=3.0, seed=0):
    rng = np.random.default_rng(seed)
    # 커밋 도착 시각: 지수분포 간격의 누적합(분 단위)
    commit_t = np.cumsum(rng.exponential(commit_gap, n_commits))
    # 빌드 시간이 모두 같아 이미지가 올라오는 순서는 커밋 순서와 같다(교육용 단순화)
    push_t = commit_t + build_min  # 빌드 시간이 같으니 올라가는 순서는 커밋 순서와 같다
    start_t = push_t + rng.exponential(queue_min, n_commits)
    wrong_runs = mixed_runs = 0
    for i in range(n_commits):
        images = []
        for k in range(3):
            t = start_t[i] + k * task_min
            if ref_mode == "latest":
                # task가 뜨는 시각 t에 latest가 가리키는 이미지 = 그 시각까지 올라온 마지막 이미지(번호). 자기 커밋 번호 i와 다르면 시험한 이미지가 아니다
                images.append(int(np.searchsorted(push_t, t, side="right") - 1))  # 그 순간 latest = 마지막으로 올라간 이미지
            else:
                # SHA/다이제스트로 제출하면 항상 자기 커밋의 이미지
                images.append(i)
        wrong_runs += any(im != i for im in images)
        mixed_runs += len(set(images)) > 1
    return {"wrong_image_run_rate": wrong_runs / n_commits, "mixed_image_run_rate": mixed_runs / n_commits}
