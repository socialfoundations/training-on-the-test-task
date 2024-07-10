import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from utils import compute_compute, color_rule, palette

def get_features(results, rule=None):
    models = list(results.keys())
    compute = [compute_compute(model) for model in models]
    score = [results[model] for model in models]
    rule = rule if rule else color_rule
    feat_string = [int(rule(model)==palette[1]) for model in models]
    return compute, feat_string, score

def regress(compute, feat_string, score):
    logcompute = np.log10(compute)
    x = np.c_[feat_string, logcompute]
    fii = sm.OLS(score, sm.add_constant(x)).fit()
    summary = fii.summary2().tables
    coeffs = summary[1]['Coef.'].to_list()
    stderrs = summary[1]['Std.Err.'].to_list()
    ps = summary[1]['P>|t|'].to_list()
    r2 = summary[0][1][6]
    return {'coeffs': coeffs, 'stderrs': stderrs, 'ps': ps, 'r2': r2}

def regress_seg(compute, feat_string, score):
    knots = [22, 23]
    logcompute = np.log10(compute)
    x1 = np.minimum(logcompute, knots[0])
    x2 = np.maximum(0, np.minimum(logcompute - knots[0], knots[1] - knots[0]))
    x3 = np.maximum(0, logcompute - knots[1])
    x = np.c_[feat_string, x1, x2, x3]

    fii = sm.OLS(score, sm.add_constant(x)).fit()

    summary = fii.summary2().tables
    coeffs = summary[1]['Coef.'].to_list()
    stderrs = summary[1]['Std.Err.'].to_list()
    ps = summary[1]['P>|t|'].to_list()
    r2 = summary[0][1][6]
    return {'coeffs': coeffs, 'stderrs': stderrs, 'ps': ps, 'r2': r2}

def regress_hinge(logcompute, feat_string, score, c, r):
    logcompute = np.maximum(0, logcompute - c)
    x = np.c_[feat_string, logcompute]
    score = np.array(score) - r
    fii = sm.OLS(score, x).fit()
    summary = fii.summary2().tables
    coeffs = summary[1]['Coef.'].to_list()
    stderrs = summary[1]['Std.Err.'].to_list()
    ps = summary[1]['P>|t|'].to_list()
    r2 = summary[0][1][6]
    return {'coeffs': coeffs, 'stderrs': stderrs, 'ps': ps, 'r2': r2}

def get_hinge_regressor(c, f, a, r):
    c = np.log10(c)
    x = np.c_[f, c]
    y = a

    def hinge(x, a, c, theta):
        f, c_ = x[:, 0], x[:, 1]
        return r + a * np.maximum(c_ - c, 0) + theta * f 

    res = minimize(lambda p: np.sum((hinge(x, *p) - y)**2), [0., 22, 0])

    def get_line(f_t, n=100):
        c_t = [c_ for c_, f_ in zip(c, f) if f_ == f_t]
        c_min = min(c_t)
        c_max = max(c_t)
        c_t = np.linspace(c_min, c_max, n)
        return 10**c_t, hinge(np.c_[[f_t]*n, c_t], *res.x)

    p0 = get_line(0)
    p1 = get_line(1)

    ce = res.x[1]
    ols = regress_hinge(c, f, a, ce, r)
    return res.x, ols, (p0, p1)
