def analyze(counts):
    if not counts or any(v < 0 for _, v in counts) or any(counts[i][1] < counts[i + 1][1] for i in range(len(counts) - 1)):
        raise ValueError("counts must be non-negative and non-increasing")
    rows, worst = [], None
    for i, (stage, count) in enumerate(counts):
        rate = None if i == 0 or counts[i - 1][1] == 0 else count / counts[i - 1][1]
        row = {"stage": stage, "count": count, "step_rate": rate}
        rows.append(row)
        if rate is not None and (worst is None or rate < worst["step_rate"]):
            worst = row
    return {"rows": rows, "investigate_first": None if worst is None else worst["stage"]}
