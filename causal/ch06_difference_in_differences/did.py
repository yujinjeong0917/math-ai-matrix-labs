def did(treated_before: float, treated_after: float, control_before: float, control_after: float) -> float:
    return (treated_after - treated_before) - (control_after - control_before)

def saturated_coefficients(treated_before: float, treated_after: float, control_before: float, control_after: float) -> dict[str, float]:
    # Y = beta0 + beta1*treated + beta2*after + beta3*treated*after
    beta0 = control_before
    beta1 = treated_before - control_before
    beta2 = control_after - control_before
    beta3 = did(treated_before, treated_after, control_before, control_after)
    return {"intercept": beta0, "treated": beta1, "after": beta2, "interaction": beta3}

def predict(coefficients: dict[str, float], treated: int, after: int) -> float:
    return (coefficients["intercept"] + coefficients["treated"] * treated
            + coefficients["after"] * after + coefficients["interaction"] * treated * after)
