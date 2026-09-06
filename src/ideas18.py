"""Y-series (W22): every freeze-v2 member at 25 seeds, on request.

Existing 25-seed twins are reused (a1x25 for s62, x1 for s39, x11 for i11w, x4, a3x25); these
produce the missing ones. Tree-based members are seed-bagged (25 random states averaged).
"""
import numpy as np
from .ideas import _dl_out

IDEAS18 = {}
S = tuple(range(25))


def _avg(fn):
    """Average a forecaster over 25 seeds (for members whose seed lives in the model object)."""
    def run(tr, va):
        outs = [fn(tr, va, sd) for sd in S]
        base = outs[0].copy(); base["osi_pred"] = np.mean([o["osi_pred"].to_numpy() for o in outs], 0)
        return base
    return run


def y1_x9_25(tr, va):
    from .ideas5 import bienc_dual_forecaster
    return bienc_dual_forecaster(tr, va, seeds=S, cyclic=(40, 4))


def y2_ts2c_25(tr, va):
    from .ideas16 import PatchTSTKF, _run_arch
    from .ideas4 import scoring_weights
    return _run_arch(PatchTSTKF, tr, va, seeds=S, w=scoring_weights())


def y3_s35w_25(tr, va):
    from .ideas3 import mixup_forecaster
    from .ideas4 import scoring_weights
    return mixup_forecaster(tr, va, seeds=S, w=scoring_weights())


def y4_f2_25(tr, va):
    from .ideas10 import f2_member_upgrade
    return f2_member_upgrade(tr, va, seeds=S)


def y5_p1b_25(tr, va):
    from .stockflow import stockflow_forecaster
    return _avg(lambda a, b, sd: stockflow_forecaster(a, b, monotone=False, seed=sd))(tr, va)


def y6_p1c_25(tr, va):
    from .stockflow import stockflow_forecaster
    return _avg(lambda a, b, sd: stockflow_forecaster(a, b, dagger_rounds=0, seed=sd))(tr, va)


def y7_r6_25(tr, va):
    from .stockflow import stockflow_r6_forecaster
    return _avg(lambda a, b, sd: stockflow_r6_forecaster(a, b, seed=sd))(tr, va)


IDEAS18.update({
    "Y5 p1b @ 25 seeds": y5_p1b_25, "Y6 p1c @ 25 seeds": y6_p1c_25, "Y7 r6corr @ 25 seeds": y7_r6_25,
    "Y4 f2 @ 25 seeds": y4_f2_25, "Y1 x9 (s51+snapshots) @ 25 seeds": y1_x9_25,
    "Y2 ts2c @ 25 seeds": y2_ts2c_25, "Y3 s35w @ 25 seeds": y3_s35w_25,
})
