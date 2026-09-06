"""P1 — damage/restoration state-space hurdle model.

A different estimator, not a different architecture. Instead of predicting OSI, we
model the physical process as a stock-flow system and enforce it as a hard constraint:

    P_t = clip(P_{t-1} + damage_t - restore_t, 0, 1)

with two learned flows:

    damage_t   : hurdle,  P(damage>0) x E[sqrt(damage) | damage>0]^2
    restore_t  : P_{t-1} x restore_fraction_t      (bounded by the stock)

OSI is then reconstructed with the organizers' exact definition from the rolled-out
stock and flows. That reconstruction was verified against the provided `osi` column at
RMSE 3.4e-05 over the scored hours -- 256x below the current ensemble error -- so the
formulation itself costs nothing.

Two properties make this worth trying after ~75 architecture-level experiments:
  * the trajectory cannot be internally inconsistent (the identity is imposed, not
    learned), and
  * the learning problem becomes ~51,000 county-hour transitions rather than 239
    sequences, which is the project's strongest empirical lever.
"""
import numpy as np
import pandas as pd

from .data import FREEZE_H, PRED_HOURS, mask_after_freeze
from .features import county_static
from .ideas import load_static
from .models import SEED

EPS = 1e-6
GUST_THR = 30.0


# --------------------------------------------------------------- weather / static
def _weather_matrices(df):
    """Per-county x hour weather arrays, all legal at any timestamp."""
    piv = lambda c: df.pivot_table(index="fipsCode", columns="hour", values=c
                                   ).sort_index().reindex(columns=np.arange(216))
    g = piv("gust").to_numpy()
    out = {"gust": g}
    out["gust_lag1"] = np.pad(g, ((0, 0), (1, 0)), mode="edge")[:, :-1]
    out["gust_lag3"] = np.pad(g, ((0, 0), (3, 0)), mode="edge")[:, :-3]
    mx6 = np.stack([np.pad(g, ((0, 0), (k, 0)), mode="edge")[:, :216] for k in range(6)])
    out["gust_max6"] = mx6.max(0)
    mx24 = np.stack([np.pad(g, ((0, 0), (k, 0)), mode="edge")[:, :216] for k in range(24)])
    out["gust_max24"] = mx24.max(0)
    run = np.maximum.accumulate(np.nan_to_num(g), axis=1)
    out["gust_runmax"] = run
    out["gust_exceed"] = np.clip(g - np.pad(run, ((0, 0), (1, 0)))[:, :-1], 0, None)
    exc = np.clip(g - GUST_THR, 0, None)
    out["cum_exposure"] = np.nan_to_num(exc).cumsum(1)
    out["exposure_now"] = exc
    for c in ("mslma", "tp", "soil_moist", "t2m", "r2", "wind_speed_10m"):
        out[c] = piv(c).to_numpy()
    out["mslma_d24"] = out["mslma"] - np.pad(out["mslma"], ((0, 0), (24, 0)),
                                             mode="edge")[:, :-24]
    hod = np.tile(np.arange(216) % 24, (len(g), 1))
    out["hod_sin"], out["hod_cos"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
    out["hour_idx"] = np.tile(np.arange(216, dtype=float), (len(g), 1))
    # hours since this county's own strongest wind so far (storm-phase clock)
    peak_so_far = np.argmax.__call__ if False else None
    hp = np.zeros_like(g)
    running_peak = np.full(len(g), -1.0)
    peak_hour = np.zeros(len(g))
    for t in range(216):
        newer = np.nan_to_num(g[:, t]) > running_peak
        running_peak = np.where(newer, np.nan_to_num(g[:, t]), running_peak)
        peak_hour = np.where(newer, t, peak_hour)
        hp[:, t] = t - peak_hour
    out["hrs_since_peak"] = hp
    return {k: np.nan_to_num(v) for k, v in out.items()}


def _static_matrix(masked):
    st = county_static(masked)
    ext = load_static().reindex(st.index)
    cols = ["log_customers", "state_IN", "state_OH", "state_PA", "state_WV"]
    S = np.column_stack([st[c].to_numpy() for c in cols]
                        + [ext["log_pop"].to_numpy(), ext["log_density"].to_numpy(),
                           ext["rucc"].to_numpy(), ext["lat"].to_numpy(),
                           ext["lon"].to_numpy()])
    return np.nan_to_num(S), st.index.to_numpy(), cols + ["log_pop", "log_density",
                                                          "rucc", "lat", "lon"]


WCOLS = ["gust", "gust_lag1", "gust_lag3", "gust_max6", "gust_max24", "gust_runmax",
         "gust_exceed", "cum_exposure", "exposure_now", "mslma", "mslma_d24", "tp",
         "soil_moist", "t2m", "r2", "wind_speed_10m", "hod_sin", "hod_cos",
         "hour_idx", "hrs_since_peak"]
SCOLS = ["stock", "up1", "up3", "up6", "dn1", "dn3", "dn6", "cum_damage",
         "hrs_since_damage", "stock_x_gust"]


def _state_block(stock, up_h, dn_h, cum_dmg, hrs_dmg, gust):
    """State features from the stock-flow history (arrays over counties)."""
    return np.column_stack([
        stock, up_h[:, -1], up_h[:, -3:].mean(1), up_h[:, -6:].mean(1),
        dn_h[:, -1], dn_h[:, -3:].mean(1), dn_h[:, -6:].mean(1),
        cum_dmg, hrs_dmg, stock * gust])


class StockFlowModel:
    """Damage hurdle + restoration-rate model over county-hour transitions."""

    def __init__(self, monotone=True, seed=SEED):
        self.monotone, self.seed = monotone, seed

    # ---------------------------------------------------------------- training
    def fit(self, train_full, dagger_rounds=1):
        """Teacher-forced fit, then optional DAgger rounds that retrain on the
        model's OWN rollout states. Diagnosis motivating this: one-step accuracy is
        good (stock RMSE 0.0073) while free-running rollout degrades to 0.0193 and
        inflates cumulative damage 1.9x, i.e. a train/inference state mismatch."""
        import xgboost as xgb
        masked = mask_after_freeze(train_full)
        W = _weather_matrices(train_full)
        S, fips, self.scols = _static_matrix(masked)
        P = (train_full.pivot_table(index="fipsCode", columns="hour", values="P_t")
             .sort_index().reindex(columns=np.arange(216)).to_numpy())
        P = np.nan_to_num(P)
        dP = np.diff(P, axis=1, prepend=P[:, :1])
        up, dn = np.clip(dP, 0, None), np.clip(-dP, 0, None)

        rows, y_up, y_rf, pos = [], [], [], []
        cum = np.zeros(len(P))
        hrs = np.full(len(P), 99.0)
        up_h = np.zeros((len(P), 6))
        dn_h = np.zeros((len(P), 6))
        for t in range(1, 216):
            stock = P[:, t - 1]
            blk = _state_block(stock, up_h, dn_h, cum, hrs, W["gust"][:, t])
            wblk = np.column_stack([W[c][:, t] for c in WCOLS])
            rows.append(np.hstack([blk, wblk, S]))
            y_up.append(up[:, t])
            y_rf.append(np.where(stock > EPS, dn[:, t] / np.maximum(stock, EPS), np.nan))
            pos.append(stock > EPS)
            cum = cum + up[:, t]
            hrs = np.where(up[:, t] > 0, 0.0, hrs + 1)
            up_h = np.roll(up_h, -1, 1); up_h[:, -1] = up[:, t]
            dn_h = np.roll(dn_h, -1, 1); dn_h[:, -1] = dn[:, t]

        X = np.vstack(rows)
        yu = np.concatenate(y_up)
        yr = np.concatenate(y_rf)
        stock_pos = np.concatenate(pos)
        self.cols = SCOLS + WCOLS + self.scols
        mono = None
        if self.monotone:                       # stronger wind cannot reduce damage
            mono = tuple(1 if c in ("gust", "gust_max6", "gust_exceed", "exposure_now")
                         else 0 for c in self.cols)

        common = dict(n_estimators=400, learning_rate=0.06, max_depth=6, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
                      random_state=self.seed, n_jobs=4, tree_method="hist")
        if mono:
            common["monotone_constraints"] = mono
        self.clf = xgb.XGBClassifier(**common, eval_metric="logloss").fit(X, (yu > 0).astype(int))
        m = yu > 0
        self.reg = xgb.XGBRegressor(**common).fit(X[m], np.sqrt(yu[m]))
        # Duan smearing: E[X|X>0] vs (E[sqrt X|X>0])^2 -- one global scalar, in-fold
        pred_sqrt = self.reg.predict(X[m])
        self.smear = float(yu[m].mean() / max((pred_sqrt ** 2).mean(), 1e-12))

        rm = stock_pos & np.isfinite(yr)
        rcommon = dict(common)
        rcommon.pop("monotone_constraints", None)
        self.res = xgb.XGBRegressor(**rcommon).fit(X[rm], np.clip(yr[rm], 0, 1))

        for _ in range(dagger_rounds):
            Xd, yud, yrd, posd = self._rollout_states(train_full, up, dn, P, W, S)
            X2, yu2 = np.vstack([X, Xd]), np.concatenate([yu, yud])
            yr2 = np.concatenate([yr, yrd])
            pos2 = np.concatenate([stock_pos, posd])
            self.clf = xgb.XGBClassifier(**common, eval_metric="logloss").fit(
                X2, (yu2 > 0).astype(int))
            m2 = yu2 > 0
            self.reg = xgb.XGBRegressor(**common).fit(X2[m2], np.sqrt(yu2[m2]))
            ps = self.reg.predict(X2[m2])
            self.smear = float(yu2[m2].mean() / max((ps ** 2).mean(), 1e-12))
            rm2 = pos2 & np.isfinite(yr2)
            self.res = xgb.XGBRegressor(**rcommon).fit(X2[rm2], np.clip(yr2[rm2], 0, 1))
        return self

    def _rollout_states(self, train_full, up, dn, P, W, S):
        """Run the current model forward on the TRAINING counties and collect
        (features from the model's own states, TRUE flow targets)."""
        n = len(P)
        Pr = P[:, :FREEZE_H + 1].copy()
        cur = Pr[:, -1].copy()
        cum = np.clip(np.diff(Pr, axis=1, prepend=Pr[:, :1]), 0, None).sum(1)
        upm = np.clip(np.diff(Pr, axis=1, prepend=Pr[:, :1]), 0, None)
        dnm = np.clip(-np.diff(Pr, axis=1, prepend=Pr[:, :1]), 0, None)
        last = np.where(upm > 0, np.arange(FREEZE_H + 1), -99).max(1)
        hrs = np.where(last < 0, 99.0, FREEZE_H - last).astype(float)
        up_h, dn_h = upm[:, -6:].copy(), dnm[:, -6:].copy()
        rows, yu, yr, pos = [], [], [], []
        for t in range(FREEZE_H + 1, 216):
            blk = _state_block(cur, up_h, dn_h, cum, hrs, W["gust"][:, t])
            X = np.hstack([blk, np.column_stack([W[c][:, t] for c in WCOLS]), S])
            rows.append(X)
            yu.append(up[:, t])                      # TRUE targets at the model's state
            yr.append(np.where(cur > EPS, dn[:, t] / np.maximum(cur, EPS), np.nan))
            pos.append(cur > EPS)
            p = self.clf.predict_proba(X)[:, 1]
            mag = np.clip(self.reg.predict(X), 0, None) ** 2 * self.smear
            up_t = p * mag
            rf = np.clip(self.res.predict(X), 0, 1)
            dn_t = np.minimum(cur * rf, cur + up_t)
            cur = np.clip(cur + up_t - dn_t, 0, 1)
            cum = cum + up_t
            hrs = np.where(up_t > 1e-9, 0.0, hrs + 1)
            up_h = np.roll(up_h, -1, 1); up_h[:, -1] = up_t
            dn_h = np.roll(dn_h, -1, 1); dn_h[:, -1] = dn_t
        return (np.vstack(rows), np.concatenate(yu), np.concatenate(yr),
                np.concatenate(pos))

    # ---------------------------------------------------------------- rollout
    def predict(self, val_masked):
        W = _weather_matrices(val_masked)
        S, fips, _ = _static_matrix(val_masked)
        Pobs = (val_masked.pivot_table(index="fipsCode", columns="hour", values="P_t")
                .sort_index().reindex(columns=np.arange(216)).to_numpy())
        Pobs = np.nan_to_num(Pobs)
        n = len(fips)
        P = np.zeros((n, 216)); P[:, :FREEZE_H + 1] = Pobs[:, :FREEZE_H + 1]
        dP = np.diff(P[:, :FREEZE_H + 1], axis=1, prepend=P[:, :1])
        UP = np.zeros((n, 216)); DN = np.zeros((n, 216))
        UP[:, :FREEZE_H + 1] = np.clip(dP, 0, None)
        DN[:, :FREEZE_H + 1] = np.clip(-dP, 0, None)

        cum = UP[:, :FREEZE_H + 1].sum(1)
        last = np.where(UP[:, :FREEZE_H + 1] > 0, np.arange(FREEZE_H + 1), -99).max(1)
        hrs = np.where(last < 0, 99.0, FREEZE_H - last).astype(float)
        up_h = UP[:, FREEZE_H - 5:FREEZE_H + 1].copy()
        dn_h = DN[:, FREEZE_H - 5:FREEZE_H + 1].copy()

        for t in range(FREEZE_H + 1, 216):
            stock = P[:, t - 1]
            blk = _state_block(stock, up_h, dn_h, cum, hrs, W["gust"][:, t])
            X = np.hstack([blk, np.column_stack([W[c][:, t] for c in WCOLS]), S])
            p = self.clf.predict_proba(X)[:, 1]
            mag = np.clip(self.reg.predict(X), 0, None) ** 2 * self.smear
            up_t = p * mag
            rf = np.clip(self.res.predict(X), 0, 1)
            dn_t = np.minimum(stock * rf, stock + up_t)        # cannot exceed the stock
            P[:, t] = np.clip(stock + up_t - dn_t, 0, 1)
            UP[:, t], DN[:, t] = up_t, dn_t
            cum = cum + up_t
            hrs = np.where(up_t > 1e-9, 0.0, hrs + 1)
            up_h = np.roll(up_h, -1, 1); up_h[:, -1] = up_t
            dn_h = np.roll(dn_h, -1, 1); dn_h[:, -1] = dn_t
        return fips, P, UP, DN


def reconstruct_osi(P, UP, DN):
    """The organizers' exact definition from stock and gross flows."""
    def centered3(A):
        B = np.pad(A, ((0, 0), (1, 1)), mode="edge")
        return (B[:, :-2] + B[:, 1:-1] + B[:, 2:]) / 3.0

    D = np.zeros_like(P)
    for i in range(P.shape[1]):
        D[:, i] = P[:, max(0, i - 5):i + 1].mean(1)
    return np.clip(0.40 * P + 0.35 * centered3(UP) + 0.25 * D - 0.10 * centered3(DN), 0, None)


def stockflow_forecaster(train_full, val_masked, monotone=True, dagger_rounds=1, seed=None):
    m = (StockFlowModel(monotone=monotone) if seed is None else StockFlowModel(monotone=monotone, seed=seed)).fit(train_full, dagger_rounds=dagger_rounds)
    fips, P, UP, DN = m.predict(val_masked)
    osi = reconstruct_osi(P, UP, DN)
    sel = np.isin(np.arange(216), PRED_HOURS)
    return pd.DataFrame({"fipsCode": np.repeat(fips, len(PRED_HOURS)),
                         "hour": np.tile(PRED_HOURS, len(fips)),
                         "osi_pred": osi[:, sel].ravel()})


def stockflow_nomono_forecaster(train_full, val_masked):
    return stockflow_forecaster(train_full, val_masked, monotone=False)


def stockflow_teacher_only(train_full, val_masked):
    """Control: no DAgger, to isolate the effect of training on rollout states."""
    return stockflow_forecaster(train_full, val_masked, dagger_rounds=0)


class StockFlowDirectModel:
    """R8 -- non-recursive cumulative-flow formulation.

    Same stock-flow decomposition as StockFlowModel, but the flows for all 144
    future hours are predicted at once from the FROZEN h71 state + weather at t +
    statics -- the model never consumes its own predictions, removing the
    `predicted Q -> next model input` channel that causes rollout drift. The stock
    identity is applied afterwards as a deterministic pass (dn_t bounded by the
    available stock, P_t clipped), exactly P1's constraint but with no model in the
    loop. Ablation question: is P1's value in the decomposition or the recursion?

    restore="gross":    predict dn_t directly (the literal R8 formulation)
    restore="fraction": predict a restore fraction as P1 does, applied to the
                        deterministically evolved stock (keeps proportionality)
    """

    def __init__(self, monotone=False, restore="gross", seed=SEED):
        self.monotone, self.restore, self.seed = monotone, restore, seed

    @staticmethod
    def _frozen_state(P):
        """State block inputs at FREEZE_H from an observed prefix (true or val)."""
        pre = P[:, :FREEZE_H + 1]
        d = np.diff(pre, axis=1, prepend=pre[:, :1])
        pre_up, pre_dn = np.clip(d, 0, None), np.clip(-d, 0, None)
        last = np.where(pre_up > 0, np.arange(FREEZE_H + 1), -99).max(1)
        hrs = np.where(last < 0, 99.0, FREEZE_H - last).astype(float)
        return (P[:, FREEZE_H], pre_up[:, -6:], pre_dn[:, -6:], pre_up.sum(1), hrs)

    def _rows(self, P, W, S):
        stock71, up_h, dn_h, cum71, hrs71 = self._frozen_state(P)
        rows = []
        for t in range(FREEZE_H + 1, 216):
            blk = _state_block(stock71, up_h, dn_h, cum71, hrs71, W["gust"][:, t])
            rows.append(np.hstack([blk, np.column_stack([W[c][:, t] for c in WCOLS]), S]))
        return rows

    def fit(self, train_full):
        import xgboost as xgb
        masked = mask_after_freeze(train_full)
        W = _weather_matrices(train_full)
        S, fips, self.scols = _static_matrix(masked)
        P = (train_full.pivot_table(index="fipsCode", columns="hour", values="P_t")
             .sort_index().reindex(columns=np.arange(216)).to_numpy())
        P = np.nan_to_num(P)
        dP = np.diff(P, axis=1, prepend=P[:, :1])
        up, dn = np.clip(dP, 0, None), np.clip(-dP, 0, None)

        rows = self._rows(P, W, S)
        hours = np.arange(FREEZE_H + 1, 216)
        X = np.vstack(rows)
        yu = up[:, hours].T.ravel()
        yd = dn[:, hours].T.ravel()
        stock_prev = P[:, hours - 1].T.ravel()
        self.cols = SCOLS + WCOLS + self.scols
        mono = None
        if self.monotone:
            mono = tuple(1 if c in ("gust", "gust_max6", "gust_exceed", "exposure_now")
                         else 0 for c in self.cols)

        common = dict(n_estimators=400, learning_rate=0.06, max_depth=6, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
                      random_state=self.seed, n_jobs=4, tree_method="hist")
        if mono:
            common["monotone_constraints"] = mono
        self.clf = xgb.XGBClassifier(**common, eval_metric="logloss").fit(X, (yu > 0).astype(int))
        m = yu > 0
        self.reg = xgb.XGBRegressor(**common).fit(X[m], np.sqrt(yu[m]))
        pred_sqrt = self.reg.predict(X[m])
        self.smear = float(yu[m].mean() / max((pred_sqrt ** 2).mean(), 1e-12))

        rcommon = dict(common)
        rcommon.pop("monotone_constraints", None)
        rm = stock_prev > EPS                    # only hours with restorable stock
        if self.restore == "gross":
            self.res = xgb.XGBRegressor(**rcommon).fit(X[rm], yd[rm])
        else:
            rf = np.clip(yd[rm] / np.maximum(stock_prev[rm], EPS), 0, 1)
            self.res = xgb.XGBRegressor(**rcommon).fit(X[rm], rf)
        return self

    def predict(self, val_masked):
        W = _weather_matrices(val_masked)
        S, fips, _ = _static_matrix(val_masked)
        Pobs = (val_masked.pivot_table(index="fipsCode", columns="hour", values="P_t")
                .sort_index().reindex(columns=np.arange(216)).to_numpy())
        Pobs = np.nan_to_num(Pobs)
        n = len(fips)
        P = np.zeros((n, 216)); P[:, :FREEZE_H + 1] = Pobs[:, :FREEZE_H + 1]
        dP = np.diff(P[:, :FREEZE_H + 1], axis=1, prepend=P[:, :1])
        UP = np.zeros((n, 216)); DN = np.zeros((n, 216))
        UP[:, :FREEZE_H + 1] = np.clip(dP, 0, None)
        DN[:, :FREEZE_H + 1] = np.clip(-dP, 0, None)

        X = np.vstack(self._rows(P, W, S))       # all 144 hours at once, frozen inputs
        p = self.clf.predict_proba(X)[:, 1]
        mag = np.clip(self.reg.predict(X), 0, None) ** 2 * self.smear
        up_all = (p * mag).reshape(216 - FREEZE_H - 1, n)
        res_all = np.clip(self.res.predict(X), 0, None).reshape(216 - FREEZE_H - 1, n)

        # deterministic constraint pass: no model in this loop
        for i, t in enumerate(range(FREEZE_H + 1, 216)):
            stock = P[:, t - 1]
            up_t = up_all[i]
            dn_pred = res_all[i] if self.restore == "gross" else np.clip(res_all[i], 0, 1) * stock
            dn_t = np.minimum(dn_pred, stock + up_t)
            P[:, t] = np.clip(stock + up_t - dn_t, 0, 1)
            UP[:, t], DN[:, t] = up_t, dn_t
        return fips, P, UP, DN


def stockflow_direct_forecaster(train_full, val_masked, restore="gross"):
    m = StockFlowDirectModel(restore=restore).fit(train_full)
    fips, P, UP, DN = m.predict(val_masked)
    osi = reconstruct_osi(P, UP, DN)
    sel = np.isin(np.arange(216), PRED_HOURS)
    return pd.DataFrame({"fipsCode": np.repeat(fips, len(PRED_HOURS)),
                         "hour": np.tile(PRED_HOURS, len(fips)),
                         "osi_pred": osi[:, sel].ravel()})


def stockflow_direct_fraction_forecaster(train_full, val_masked):
    return stockflow_direct_forecaster(train_full, val_masked, restore="fraction")


# ------------------------------------------------------------------ R6 (W21)
def _rollout_scaled(m, df, c_dmg=1.0, c_res=1.0):
    """Closed-loop rollout with two flow scalars (ported from r6_flowbias.rollout)."""
    W = _weather_matrices(df)
    S, fips, _ = _static_matrix(df)
    Pobs = np.nan_to_num(df.pivot_table(index="fipsCode", columns="hour", values="P_t")
                         .sort_index().reindex(columns=np.arange(216)).to_numpy())
    n = len(fips)
    P = np.zeros((n, 216)); P[:, :FREEZE_H + 1] = Pobs[:, :FREEZE_H + 1]
    d = np.diff(P[:, :FREEZE_H + 1], axis=1, prepend=P[:, :1])
    UP = np.zeros((n, 216)); DN = np.zeros((n, 216))
    UP[:, :FREEZE_H + 1] = np.clip(d, 0, None); DN[:, :FREEZE_H + 1] = np.clip(-d, 0, None)
    cum = UP[:, :FREEZE_H + 1].sum(1)
    last = np.where(UP[:, :FREEZE_H + 1] > 0, np.arange(FREEZE_H + 1), -99).max(1)
    hrs = np.where(last < 0, 99.0, FREEZE_H - last).astype(float)
    uh, dh = UP[:, FREEZE_H - 5:FREEZE_H + 1].copy(), DN[:, FREEZE_H - 5:FREEZE_H + 1].copy()
    for tt in range(FREEZE_H + 1, 216):
        stock = P[:, tt - 1]
        X = np.hstack([_state_block(stock, uh, dh, cum, hrs, W["gust"][:, tt]),
                       np.column_stack([W[c][:, tt] for c in WCOLS]), S])
        p = m.clf.predict_proba(X)[:, 1]
        mag = np.clip(m.reg.predict(X), 0, None) ** 2 * m.smear
        up = c_dmg * p * mag
        rf = np.clip(m.res.predict(X), 0, 1)
        dn = np.minimum(c_res * stock * rf, stock + up)
        P[:, tt] = np.clip(stock + up - dn, 0, 1)
        UP[:, tt], DN[:, tt] = up, dn
        cum = cum + up
        hrs = np.where(up > 1e-9, 0.0, hrs + 1)
        uh = np.roll(uh, -1, 1); uh[:, -1] = up
        dh = np.roll(dh, -1, 1); dh[:, -1] = dn
    return fips, P, UP, DN


def stockflow_r6_forecaster(train_full, val_masked, seed=None):
    """R6 member (W21): p1b's model with two flow-bias scalars -- damage and restoration
    cumulative-flow ratios on the TRAINING counties' own rollout, clipped to [0.5, 2] -- the
    exact procedure r6_flowbias.py ran nested to produce oof/r6corr.csv."""
    m = (StockFlowModel(monotone=False) if seed is None else StockFlowModel(monotone=False, seed=seed)).fit(train_full, dagger_rounds=1)
    _, Ptr, UPtr, DNtr = _rollout_scaled(m, mask_after_freeze(train_full))
    Ttr = np.nan_to_num(train_full.pivot_table(index="fipsCode", columns="hour", values="P_t")
                        .sort_index().reindex(columns=np.arange(216)).to_numpy())
    dT = np.diff(Ttr, axis=1, prepend=Ttr[:, :1])
    c_d = float(np.clip(np.clip(dT, 0, None)[:, 72:].sum() / max(UPtr[:, 72:].sum(), 1e-9), 0.5, 2.0))
    c_r = float(np.clip(np.clip(-dT, 0, None)[:, 72:].sum() / max(DNtr[:, 72:].sum(), 1e-9), 0.5, 2.0))
    fips, P, UP, DN = _rollout_scaled(m, val_masked, c_d, c_r)
    osi = reconstruct_osi(P, UP, DN)
    sel = np.isin(np.arange(216), PRED_HOURS)
    return pd.DataFrame({"fipsCode": np.repeat(fips, len(PRED_HOURS)),
                         "hour": np.tile(PRED_HOURS, len(fips)), "osi_pred": osi[:, sel].ravel()})
