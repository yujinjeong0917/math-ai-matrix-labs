"""문의 내보내기 CSV(열: id, received_at, type, text)에서 유형별 건수를 센다.

스킬 본문이 "건수를 직접 세지 말고 이 스크립트를 돌려요"라고 시킨다.
스크립트 코드는 모델의 문맥에 들어가지 않고, 실행 결과(아래 한 줄)만 들어간다.
"""

import csv
import sys
from collections import Counter


def count(path):
    """CSV의 type 열 값별 행 수를 센다. 반환: Counter(유형 -> 건수)."""
    with open(path, encoding="utf-8") as f:
        return Counter(row["type"] for row in csv.DictReader(f))


if __name__ == "__main__":
    c = count(sys.argv[1])
    # 건수 많은 순(most_common)으로 한 줄 출력. 이 한 줄이 모델 문맥에 들어가는 전부다.
    print(", ".join(f"{k} {v}" for k, v in c.most_common()), f"/ 합계 {sum(c.values())}")
