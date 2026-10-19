"""Small stats helpers (no scipy on this machine): Welch t-test p-value via the regularised incomplete beta."""
import math
import statistics as st


def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
        de = d * c
        h *= de
        if abs(de - 1) < 1e-12:
            break
    return h


def _ibeta(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1 - bt * _betacf(b, a, 1 - x) / b


def welch_p(x, y):
    """two-sided Welch t-test p-value"""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float('nan')
    vx, vy = st.variance(x) / nx, st.variance(y) / ny
    if vx + vy == 0:
        return 1.0
    t = (st.mean(x) - st.mean(y)) / math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (nx - 1) + vy ** 2 / (ny - 1))
    return _ibeta(df / 2, 0.5, df / (df + t * t))


def paired_p(x, y):
    """two-sided paired t-test p-value"""
    d = [a - b for a, b in zip(x, y)]
    n = len(d)
    if n < 2:
        return float('nan')
    s = st.stdev(d)
    if s == 0:
        return 1.0
    t = st.mean(d) / (s / math.sqrt(n))
    return _ibeta((n - 1) / 2, 0.5, (n - 1) / ((n - 1) + t * t))
