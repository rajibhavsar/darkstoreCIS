"""
engine.py  -  Monte-Carlo scenario engine for the Bengaluru dark-store model
==========================================================================
Implements Step 20 (coverage) and Step 21 (random demand, two-stage) of the
derivation document.

Stage 1  (here-and-now)  : choose which stores to open, once, on EXPECTED demand.
Stage 2  (wait-and-see)  : for every random-demand scenario s, evaluate the
                           network (coverage mode) or re-assign customers by
                           solving the capacity LP Q_s(y) (cost mode).

Random demand
-------------
    population_i^s = population_i  x  multiplier_i^s
    multiplier     = local (per hexagon)  x  ward (shared by a ward)  x  city (shared by all)
Each factor has mean 1, so the AVERAGE demand stays equal to the base data and
only the spread changes.

No Streamlit imports here: the engine can be used from a notebook or script.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Deployed copy: the data sits in simulation/data. On your laptop it falls back to ../data and ../tableau.
_LOCAL = os.path.join(HERE, "data")
_has_local = os.path.exists(os.path.join(_LOCAL, "hex_centroids.csv"))
DATA = _LOCAL if _has_local else os.path.join(ROOT, "data")
TABLEAU = _LOCAL if _has_local else os.path.join(ROOT, "tableau")

# study box (same as darkstore_model.py)
NORTH, SOUTH, EAST, WEST = 13.15, 12.82, 77.80, 77.44

# --------------------------------------------------------------------------
# Default parameters (every one is editable in the dashboard)
# --------------------------------------------------------------------------
DEFAULTS = dict(
    mode="coverage",            # "coverage" (Step 20) | "cost" (Step 21)
    # decisions
    promise=10, n_stores=10, method="exact", drop_shops=False,
    # time budget (Step 4)
    tau_p=2.0, tau_l=2.0, speed=1.0,
    # demand (Step 5)
    hh_size=3.6, phi=0.30, lam=3.0, weekday_share=0.70, peak_share=0.35, peak_hours=3,
    # operations (Step 7)
    orders_per_rider_hr=6.0, riders_small=5, riders_large=12,
    # randomness
    n_scen=500, seed=42, dist="lognormal",
    sigma_local=0.25, sigma_ward=0.10, sigma_city=0.0,
    # robustness extras
    n_pick=60,
    # costs (Steps 13-17)  Rs  - ILLUSTRATIVE, no real data yet
    c_proc=10.0, c_rider=2.0, c_late=5.0, c_lost=100.0,
    f_small=60_000.0, f_large=150_000.0,      # per store per week
    beta=1.0, tmax_extra=5.0, force_n=False,
)


def omega(P) -> float:
    """share of a weekly order count that lands in ONE peak hour"""
    return P["weekday_share"] / 5.0 * P["peak_share"] / P["peak_hours"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class City:
    """Static data: hexagons, candidate sites, road minutes. Loaded once."""

    def __init__(self):
        hx = pd.read_csv(os.path.join(DATA, "hex_centroids.csv")).reset_index(drop=True)
        sites = pd.read_csv(os.path.join(DATA, "real_candidates.csv"))
        times = pd.read_csv(os.path.join(DATA, "travel_times.csv"))

        inside = sites.lat.between(SOUTH, NORTH) & sites.lon.between(WEST, EAST)
        routed = sites.site_id.isin(times.site_id)
        sites = sites[inside & routed].reset_index(drop=True)
        # no landuse and no building tag => came from a shop tag (already trading)
        sites["shop_only"] = sites.osm_landuse.isna() & sites.osm_building.isna()

        hi = {h: i for i, h in enumerate(hx.hex_id)}
        si = {s: j for j, s in enumerate(sites.site_id)}
        t = times[times.site_id.isin(si)].copy()
        t["hi"] = t.hex_id.map(hi)
        t["si"] = t.site_id.map(si)
        t = t.dropna(subset=["hi"])
        R = np.full((len(hx), len(sites)), np.inf, dtype=np.float32)
        R[t.hi.astype(int).values, t.si.values] = t.drive_minutes.values.astype(np.float32)

        self.hexes, self.sites, self.R = hx, sites, R
        self.H, self.J = R.shape
        self.pop = hx.population.values.astype(np.float64)
        self.ward_code, self.ward_names = pd.factorize(hx.ward_no.astype(str))
        self.n_wards = len(self.ward_names)

        # competitor / access data (for the Competition tab)
        self.access = pd.read_csv(os.path.join(TABLEAU, "hexagons_bengaluru.csv"))
        self.competitors = pd.read_csv(os.path.join(DATA, "competitor_stores_bengaluru.csv"))

    def site_mask(self, drop_shops: bool):
        return ~self.sites.shop_only.values if drop_shops else np.ones(self.J, bool)


# --------------------------------------------------------------------------
# Random demand
# --------------------------------------------------------------------------
def _factor(rng, shape, sigma, dist):
    """random factor with mean exactly 1 and standard deviation ~sigma"""
    if sigma <= 0:
        return np.ones(shape)
    if dist == "lognormal":
        return np.exp(sigma * rng.standard_normal(shape) - sigma ** 2 / 2)
    if dist == "uniform":
        a = min(1.0, sigma * np.sqrt(3))
        return 1 + rng.uniform(-a, a, shape)
    if dist == "normal":                      # truncated at 5% of base
        return np.maximum(0.05, 1 + sigma * rng.standard_normal(shape))
    raise ValueError(dist)


def draw_multipliers(P, city: City):
    rng = np.random.default_rng(P["seed"])
    S, H = P["n_scen"], city.H
    local = _factor(rng, (S, H), P["sigma_local"], P["dist"])
    ward = _factor(rng, (S, city.n_wards), P["sigma_ward"], P["dist"])[:, city.ward_code]
    cityw = _factor(rng, (S, 1), P["sigma_city"], P["dist"])
    return local * ward * cityw


def demand_matrices(P, city: City, mult):
    """population -> households -> weekly orders -> peak-hour orders  (Step 5)"""
    pop_s = city.pop[None, :] * mult
    orders = pop_s / P["hh_size"] * P["phi"] * P["lam"]
    return pop_s, orders, orders * omega(P)


# --------------------------------------------------------------------------
# Stage-1 solvers
# --------------------------------------------------------------------------
def greedy_cover(w, Cf, n):
    """add the store that reaches the most new demand, n times (nested path)"""
    H, J = Cf.shape
    covered = np.zeros(H, bool)
    avail = np.ones(J, bool)
    order = []
    for _ in range(min(n, J)):
        g = (w * ~covered).astype(np.float32) @ Cf
        g[~avail] = -1
        j = int(np.argmax(g))
        order.append(j)
        avail[j] = False
        covered |= Cf[:, j] > 0
    return order


def exact_cover(w, Cf, n, time_limit=60):
    """best N stores at once: maximal covering location problem (integer programme)"""
    hrows = np.where(Cf.any(1))[0]
    scols = np.where(Cf.any(0))[0]
    n = min(n, len(scols))
    Cs = sparse.csr_matrix(Cf[np.ix_(hrows, scols)])
    nh, ns = Cs.shape
    c = np.concatenate([np.zeros(ns), -w[hrows]])
    A1 = sparse.hstack([-Cs, sparse.identity(nh, format="csr")]).tocsr()   # y_i - sum x_j <= 0
    A2 = np.concatenate([np.ones(ns), np.zeros(nh)])[None, :]              # sum x_j = N
    res = milp(c, constraints=[LinearConstraint(A1, -np.inf, 0), LinearConstraint(A2, n, n)],
               integrality=np.concatenate([np.ones(ns), np.zeros(nh)]),
               bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-4})
    if res.x is None:
        return greedy_cover(w, Cf, n), "greedy fallback"
    x = res.x[:ns]
    chosen = list(scols[np.argsort(-x)[:n]])
    status = "optimal" if res.status == 0 else "best found (time limit)"
    return chosen, status


def greedy_cost(d, ben, attr, caps_wk, fixed, nmax, force_n):
    """Open stores one at a time while each one earns more than it costs (Step 18).

    ben[i,j]  = c_lost - unit cost of serving i from j  (money saved per order, 0 if not worth it)
    attr[i,j] = 1 if site j could serve hexagon i
    A store can only take caps_wk[f] orders a week, so its gain is scaled down when it
    would attract more orders than that. After a store opens, the demand it absorbs is
    removed before the next one is judged (this is what stops one hot spot being counted twice).
    Returns (sites, is_large flags).
    """
    res = d.astype(np.float32).copy()
    order, large = [], []
    taken = np.zeros(ben.shape[1], bool)
    for _ in range(min(nmax, ben.shape[1])):
        value = res @ ben
        att = np.maximum(res @ attr, 1e-6)
        gains = []
        for f in (0, 1):
            gains.append(value * np.minimum(1.0, caps_wk[f] / att) - fixed[f])
        g = np.maximum(gains[0], gains[1])
        g[taken] = -np.inf
        j = int(np.argmax(g))
        if g[j] <= 0 and not (force_n and len(order) < nmax):
            break
        f = int(gains[1][j] > gains[0][j])
        frac = min(1.0, caps_wk[f] / att[j])
        res = res * (1 - frac * attr[:, j])
        taken[j] = True
        order.append(j)
        large.append(bool(f))
    return order, large


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class SimResult:
    P: dict
    chosen: pd.DataFrame            # one row per open store
    scen: pd.DataFrame              # one row per scenario
    hexdf: pd.DataFrame             # one row per hexagon
    util: np.ndarray                # scenarios x stores, peak-hour utilisation (1 = full)
    pick: pd.DataFrame | None       # site pick frequency under re-optimisation
    reopt: np.ndarray | None        # coverage % if stores were re-chosen per scenario
    meta: dict = field(default_factory=dict)
    scen_sites: list = field(default_factory=list)   # per re-optimised scenario: indices into sites_xy
    sites_xy: pd.DataFrame | None = None             # site_id, lat, lon of every candidate used
    reopt_base: np.ndarray | None = None             # fixed greedy network on the same scenarios (like-for-like)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def run(P: dict, city: City) -> SimResult:
    P = {**DEFAULTS, **P}
    t0 = time.time()
    keep = np.where(city.site_mask(P["drop_shops"]))[0]
    sites = city.sites.iloc[keep].reset_index(drop=True)
    R = city.R[:, keep] / P["speed"]                      # road minutes at this speed
    T = P["promise"]
    total_t = P["tau_p"] + R + P["tau_l"]                 # t_ij (Step 4)

    mult = draw_multipliers(P, city)
    pop_s, ord_s, pk_s = demand_matrices(P, city, mult)
    base_orders = city.pop / P["hh_size"] * P["phi"] * P["lam"]   # expected demand
    base_pk = base_orders * omega(P)

    if P["mode"] == "coverage":
        out = _run_coverage(P, city, sites, R, total_t, ord_s, pk_s, base_orders, base_pk)
    else:
        out = _run_cost(P, city, sites, R, total_t, ord_s, pk_s, base_orders, base_pk)

    chosen, scen, hex_service, util, status = out
    S = P["n_scen"]
    scen.insert(0, "scenario", np.arange(1, S + 1))
    scen["population"] = pop_s.sum(1)
    scen["total_orders_week"] = ord_s.sum(1)

    hexdf = city.hexes[["hex_id", "centroid_lat", "centroid_lon", "ward_name", "population"]].copy()
    hexdf["pop_sim_mean"] = pop_s.mean(0)
    hexdf["pop_sim_p5"] = np.percentile(pop_s, 5, axis=0)
    hexdf["pop_sim_p95"] = np.percentile(pop_s, 95, axis=0)
    hexdf["service_rate"] = hex_service
    hexdf["covered"] = hex_service > 0.5

    meta = dict(status=status, seconds=time.time() - t0,
                n_sites=len(sites), mult_mean=float(mult.mean()), mult_sd=float(mult.std()))
    res = SimResult(P, chosen, scen, hexdf, util, None, None, meta)
    _pick_frequency(res, P, city, sites, R, total_t, base_orders, ord_s, mult)
    meta["seconds"] = time.time() - t0
    return res


# ---- coverage mode (Step 20) ----------------------------------------------
def _run_coverage(P, city, sites, R, total_t, ord_s, pk_s, base_orders, base_pk):
    C = total_t <= P["promise"]                                    # hexagon i reached from site j in time
    Cf = C.astype(np.float32)
    if P["method"] == "exact":
        order, status = exact_cover(base_orders, Cf, P["n_stores"])
    else:
        order, status = greedy_cover(base_orders, Cf, P["n_stores"]), "greedy"
    cov = C[:, order].any(1)

    D = ord_s.sum(1)
    covered_pct = 100 * (ord_s @ cov) / D
    # who serves whom: nearest open store (for the capacity stress test)
    near = np.argmin(R[:, order], axis=1)
    onehot = sparse.csr_matrix((cov.astype(float), (np.arange(len(near)), near)), shape=(len(near), len(order)))
    load = pk_s @ onehot                                           # scenarios x stores
    cap = P["riders_large"] * P["orders_per_rider_hr"]
    util = load / cap
    scen = pd.DataFrame({
        "served_pct": covered_pct, "ontime_pct": covered_pct,
        "orders_served": ord_s @ cov, "peak_util_max": util.max(1),
        "stores_over_capacity": (util > 1).sum(1)})
    chosen = _chosen_table(sites, order, cap=np.full(len(order), cap), fmt=["standard"] * len(order),
                           base_load=(base_pk * cov) @ onehot)
    return chosen, scen, cov.astype(float), util, status


# ---- cost mode (Step 18 + 21) ------------------------------------------------
def _unit_cost(P, R, total_t):
    """c_proc + c_rider*2*r/(mu*beta) + c_late*lateness, per order, for allowed pairs; inf otherwise"""
    T = P["promise"]
    late = np.maximum(0, total_t - T)
    trans = P["c_rider"] * 2 * R / P["beta"]
    allowed = total_t <= T + P["tmax_extra"]                      # Step 6: the set L_i
    U = np.where(allowed, P["c_proc"] + trans + P["c_late"] * late, np.inf).astype(np.float32)
    return U, trans.astype(np.float32), late.astype(np.float32)


def _benefit(P, U):
    ben = np.maximum(P["c_lost"] - U, 0).astype(np.float32)
    return ben, (ben > 0).astype(np.float32)


def _caps_week(P):
    """weekly orders one store can handle = peak-hour capacity / share of the week in the peak hour"""
    w = omega(P)
    return [P["riders_small"] * P["orders_per_rider_hr"] / w, P["riders_large"] * P["orders_per_rider_hr"] / w]


def _fixed(P):
    return [P["f_small"], P["f_large"]]


def _run_cost(P, city, sites, R, total_t, ord_s, pk_s, base_orders, base_pk):
    U, trans, late = _unit_cost(P, R, total_t)
    cap_s = P["riders_small"] * P["orders_per_rider_hr"]            # orders per peak hour (Step 7)
    cap_l = P["riders_large"] * P["orders_per_rider_hr"]
    ben, attr = _benefit(P, U)
    order, large = greedy_cost(base_orders, ben, attr, _caps_week(P), _fixed(P), P["n_stores"], P["force_n"])
    status = "greedy cost search"
    if not order:
        raise ValueError("At these costs no store pays for itself - lower the fixed cost or raise the lost-order cost.")
    large = np.array(large)
    Uc = U[:, order]
    near = np.argmin(Uc, axis=1)
    ok = Uc[np.arange(len(near)), near] < P["c_lost"]
    oh = sparse.csr_matrix((ok.astype(float), (np.arange(len(near)), near)), shape=(len(near), len(order)))
    base_load = base_pk @ oh
    caps = np.where(large, cap_l, cap_s).astype(float)
    fixed_each = np.where(large, P["f_large"], P["f_small"])
    fixed = fixed_each.sum()

    # --- stage 2: one LP per scenario over the open stores only ---
    pair_mask = Uc < P["c_lost"]
    hi, cj = np.nonzero(pair_mask)
    uh = Uc[hi, cj].astype(float)
    th = trans[:, order][hi, cj].astype(float)
    lt = late[:, order][hi, cj].astype(float)
    ontime = lt <= 0
    uniq_h, inv = np.unique(hi, return_inverse=True)
    npair, nh, ns = len(hi), len(uniq_h), len(order)
    rows_hex = inv
    S = ord_s.shape[0]
    cols = np.arange(npair)
    scen = {k: np.zeros(S) for k in ["served_pct", "ontime_pct", "orders_served", "cost_total", "cost_fixed",
                                     "cost_proc", "cost_transport", "cost_late", "cost_lost",
                                     "peak_util_max", "stores_over_capacity"]}
    util = np.zeros((S, ns))
    hex_service = np.zeros(city.H)
    for s in range(S):
        d, dpk = ord_s[s], pk_s[s]
        D = d.sum()
        c = d[hi] * (uh - P["c_lost"])
        A = sparse.vstack([
            sparse.csr_matrix((np.ones(npair), (rows_hex, cols)), shape=(nh, npair)),
            sparse.csr_matrix((dpk[hi], (cj, cols)), shape=(ns, npair))]).tocsr()
        b = np.concatenate([np.ones(nh), caps])
        r = linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), method="highs")
        z = r.x if r.x is not None else np.zeros(npair)
        flow = d[hi] * z                                          # weekly orders on each pair
        served = flow.sum()
        proc = P["c_proc"] * served
        tr = (flow * th).sum()                                    # th already includes c_rider
        lat = (flow * lt).sum() * P["c_late"]
        lost = P["c_lost"] * (D - served)
        scen["orders_served"][s] = served
        scen["served_pct"][s] = 100 * served / D
        scen["ontime_pct"][s] = 100 * flow[ontime].sum() / D
        scen["cost_fixed"][s], scen["cost_proc"][s] = fixed, proc
        scen["cost_transport"][s], scen["cost_late"][s], scen["cost_lost"][s] = tr, lat, lost
        scen["cost_total"][s] = fixed + proc + tr + lat + lost
        u = np.bincount(cj, weights=dpk[hi] * z, minlength=ns) / caps
        util[s] = u
        scen["peak_util_max"][s] = u.max()
        scen["stores_over_capacity"][s] = int((u > 0.999).sum())
        hex_service += np.bincount(hi, weights=z, minlength=city.H)
    hex_service /= S
    chosen = _chosen_table(sites, order, caps, np.where(large, "large", "small"), base_load)
    return chosen, pd.DataFrame(scen), hex_service, util, status


def _chosen_table(sites, order, cap, fmt, base_load):
    ch = sites.iloc[order][["site_id", "lat", "lon"]].reset_index(drop=True)
    ch["format"] = fmt
    ch["capacity_orders_hr"] = cap
    ch["expected_peak_orders_hr"] = np.asarray(base_load).ravel()
    return ch


# ---- robustness: re-choose stores in every scenario --------------------------
def _pick_frequency(res, P, city, sites, R, total_t, base_orders, ord_s, mult):
    n_pick = int(min(P["n_pick"], P["n_scen"]))
    if n_pick <= 0:
        return
    counts = np.zeros(len(sites))
    reopt = np.zeros(n_pick)
    sets = []
    if P["mode"] == "coverage":
        # greedy in every scenario (fast enough for 500+); compared like-for-like with a greedy fixed network
        Cf = (total_t <= P["promise"]).astype(np.float32)
        base_cov = (Cf[:, greedy_cover(base_orders, Cf, P["n_stores"])] > 0).any(1)
        reopt_base = np.zeros(n_pick)
        for s in range(n_pick):
            o = greedy_cover(ord_s[s], Cf, P["n_stores"])
            sets.append([int(j) for j in o])
            counts[o] += 1
            reopt[s] = 100 * (ord_s[s] @ (Cf[:, o] > 0).any(1)) / ord_s[s].sum()
            reopt_base[s] = 100 * (ord_s[s] @ base_cov) / ord_s[s].sum()
        res.reopt, res.reopt_base = reopt, reopt_base
    else:
        U, _, _ = _unit_cost(P, R, total_t)
        ben, attr = _benefit(P, U)
        for s in range(n_pick):
            o, _ = greedy_cost(ord_s[s], ben, attr, _caps_week(P), _fixed(P), P["n_stores"], P["force_n"])
            counts[o] += 1
            sets.append([int(j) for j in o])
    res.scen_sites = sets
    res.sites_xy = sites[["site_id", "lat", "lon"]].reset_index(drop=True)
    res.pick = pd.DataFrame({"site_id": sites.site_id, "lat": sites.lat, "lon": sites.lon,
                             "pick_share": counts / n_pick})
    res.pick = res.pick[res.pick.pick_share > 0].sort_values("pick_share", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Quick deterministic coverage (expected demand, greedy) for what-if questions
# --------------------------------------------------------------------------
def base_coverage(P: dict, city: City, promise=None, n=None) -> float:
    """% of expected weekly demand reached by a greedy network of n stores at promise T"""
    P = {**DEFAULTS, **P}
    T = P["promise"] if promise is None else promise
    n = int(P["n_stores"] if n is None else n)
    keep = np.where(city.site_mask(P["drop_shops"]))[0]
    R = city.R[:, keep] / P["speed"]
    Cf = ((P["tau_p"] + R + P["tau_l"]) <= T).astype(np.float32)
    base = city.pop / P["hh_size"] * P["phi"] * P["lam"]
    order = greedy_cover(base, Cf, n)
    cov = (Cf[:, order] > 0).any(1)
    return float(100 * (base @ cov) / base.sum())


def sensitivity(P: dict, city: City) -> pd.DataFrame:
    """One-at-a-time what-ifs on the coverage of the current network (plan figure 11, tornado)."""
    P = {**DEFAULTS, **P}
    base = base_coverage(P, city)
    N, T = int(P["n_stores"]), P["promise"]
    cases = [
        ("Delivery promise", f"T = {max(4, T - 2)} min", dict(promise=max(4, T - 2)),
         f"T = {T + 2} min", dict(promise=T + 2)),
        ("Number of stores", f"N = {max(1, round(N * .75))}", dict(n_stores=max(1, round(N * .75))),
         f"N = {round(N * 1.25)}", dict(n_stores=round(N * 1.25))),
        ("Rider speed", "20% slower", dict(speed=P["speed"] * .8), "20% faster", dict(speed=P["speed"] * 1.2)),
        ("Last stretch (parking to door)", f"{P['tau_l'] + 2:g} min", dict(tau_l=P["tau_l"] + 2),
         f"{max(0, P['tau_l'] - 1):g} min", dict(tau_l=max(0, P["tau_l"] - 1))),
        ("Packing time", f"{P['tau_p'] + 1:g} min", dict(tau_p=P["tau_p"] + 1),
         f"{max(0, P['tau_p'] - 1):g} min", dict(tau_p=max(0, P["tau_p"] - 1))),
    ]
    rows = []
    for name, lo_l, lo_o, hi_l, hi_o in cases:
        rows.append(dict(factor=name, low_label=lo_l, high_label=hi_l,
                         low=base_coverage({**P, **lo_o}, city), high=base_coverage({**P, **hi_o}, city), base=base))
    df = pd.DataFrame(rows)
    df["swing"] = (df.high - df.low).abs()
    return df.sort_values("swing", ascending=True).reset_index(drop=True)


# --------------------------------------------------------------------------
# Sensitivity analysis
# --------------------------------------------------------------------------
def _pct(v, p, up, integer=False, lo=None, hi=None):
    x = v * (1 + p) if up else v * (1 - p)
    x = round(x) if integer else x
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def tornado(P: dict, city: City, metric: str = "coverage", pct: float = 0.2) -> pd.DataFrame:
    """One-at-a-time sensitivity: every input moved down and up by the same pct.

    metric "coverage": % of expected weekly demand reached by a greedy network (fast).
    metric "cost"    : expected weekly cost in Rs crore, with the network RE-CHOSEN for each case.
    """
    P = {**DEFAULTS, **P}
    if metric == "coverage":
        f = lambda Q: base_coverage(Q, city)
        specs = [("Delivery promise T", "promise", dict(integer=True, lo=4)),
                 ("Number of stores N", "n_stores", dict(integer=True, lo=1)),
                 ("Rider speed", "speed", {}),
                 ("Last stretch time", "tau_l", {}),
                 ("Packing time", "tau_p", {})]
    else:
        def f(Q):
            r = run({**Q, "n_scen": 1, "sigma_local": 0, "sigma_ward": 0, "sigma_city": 0, "n_pick": 0}, city)
            return float(r.scen.cost_total.mean() / 1e7)
        specs = [("Lost-order cost", "c_lost", {}), ("Rent (fixed cost)", "fixed", {}),
                 ("Rider cost per minute", "c_rider", {}), ("Processing per order", "c_proc", {}),
                 ("Lateness penalty", "c_late", {}), ("Orders per rider trip (β)", "beta", dict(lo=1.0)),
                 ("Rider speed", "speed", {}), ("Orders per rider-hour", "orders_per_rider_hr", {}),
                 ("Share of households ordering (φ)", "phi", dict(hi=1.0)),
                 ("Orders per household (λ)", "lam", {})]
    base = f(P)
    rows = []
    for name, key, kw in specs:
        out = []
        for up in (False, True):
            if key == "fixed":
                m = 1 + pct if up else 1 - pct
                Q = {**P, "f_small": P["f_small"] * m, "f_large": P["f_large"] * m}
                shown = f"{m:.0%} of ₹{P['f_small']:,.0f} / ₹{P['f_large']:,.0f}"
            else:
                v = _pct(P[key], pct, up, **kw)
                Q = {**P, key: v}
                shown = f"{v:g}"
            out.append((f(Q), shown))
        rows.append(dict(factor=name, low_label=out[0][1], high_label=out[1][1], low=out[0][0], high=out[1][0], base=base))
    df = pd.DataFrame(rows)
    df["swing"] = (df.high - df.low).abs()
    return df.sort_values("swing", ascending=True).reset_index(drop=True)


def cost_grid(P: dict, city: City, f_mults, lost_vals) -> pd.DataFrame:
    """How many stores does the cost model choose to open for each (rent multiplier, lost-order cost)?"""
    P = {**DEFAULTS, **P}
    keep = np.where(city.site_mask(P["drop_shops"]))[0]
    R = city.R[:, keep] / P["speed"]
    total_t = P["tau_p"] + R + P["tau_l"]
    U, _, _ = _unit_cost(P, R, total_t)
    base = city.pop / P["hh_size"] * P["phi"] * P["lam"]
    rows = []
    for m in f_mults:
        for cl in lost_vals:
            ben, attr = _benefit({**P, "c_lost": cl}, U)
            o, _ = greedy_cost(base, ben, attr, _caps_week(P), [P["f_small"] * m, P["f_large"] * m],
                               P["n_stores"], False)
            rows.append(dict(rent_mult=m, c_lost=cl, stores=len(o)))
    return pd.DataFrame(rows)


def volatility_sweep(P: dict, city: City, which: str, values, n_scen: int = 200) -> pd.DataFrame:
    """How do the average and the bad-tail outcomes change as demand noise grows?"""
    P = {**DEFAULTS, **P}
    rows = []
    for v in values:
        Q = {**P, which: v, "n_scen": n_scen, "n_pick": 0}
        r = run(Q, city)
        s = r.scen
        x = s.cost_total / 1e7 if P["mode"] == "cost" else s.served_pct
        rows.append(dict(sigma=v, mean=x.mean(), p5=np.percentile(x, 5), p95=np.percentile(x, 95)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Sweep: coverage vs number of stores for several promises  (plan figure 7)
# --------------------------------------------------------------------------
def coverage_sweep(P: dict, city: City, promises, n_max=30) -> pd.DataFrame:
    P = {**DEFAULTS, **P}
    keep = np.where(city.site_mask(P["drop_shops"]))[0]
    R = city.R[:, keep] / P["speed"]
    total_t = P["tau_p"] + R + P["tau_l"]
    mult = draw_multipliers(P, city)
    _, ord_s, _ = demand_matrices(P, city, mult)
    base = city.pop / P["hh_size"] * P["phi"] * P["lam"]
    D = ord_s.sum(1)
    rows = []
    for T in promises:
        Cf = (total_t <= T).astype(np.float32)
        order = greedy_cover(base, Cf, n_max)
        cum = np.zeros((city.H, n_max), bool)
        cov = np.zeros(city.H, bool)
        for k, j in enumerate(order):
            cov = cov | (Cf[:, j] > 0)
            cum[:, k] = cov
        pct = 100 * (ord_s @ cum) / D[:, None]                    # scenarios x N
        for k in range(len(order)):
            rows.append(dict(promise=T, stores=k + 1, mean=pct[:, k].mean(),
                             p5=np.percentile(pct[:, k], 5), p95=np.percentile(pct[:, k], 95),
                             base=100 * (base @ cum[:, k]) / base.sum()))
    return pd.DataFrame(rows)
