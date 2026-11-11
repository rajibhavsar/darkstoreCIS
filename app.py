"""
Bengaluru dark-store planner  -  random-demand scenario simulator (dashboard)

Run from this folder:
    py -m streamlit run app.py
"""
import json
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

import engine as E

st.set_page_config(page_title="Dark-Store Simulator", page_icon="📦", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------- style --
BLUE, BLUE_D, AMBER, RED, SLATE, MUTED = "#1D4ED8", "#1E3A8A", "#F59E0B", "#DC2626", "#1E293B", "#64748B"
PLAT = {"Blinkit": "#EAB308", "Zepto": "#7C3AED", "Instamart": "#EA580C"}
FONT = "Inter, Segoe UI, system-ui, sans-serif"

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }}
  [data-testid="stSidebar"] {{ background: #F8FAFC; border-right: 1px solid #E2E8F0; }}
  .hero {{ background: linear-gradient(120deg, {BLUE_D} 0%, {BLUE} 55%, #3B82F6 100%);
           color: white; padding: 22px 28px; border-radius: 14px; margin-bottom: 18px; }}
  .hero h1 {{ margin: 0; font-size: 1.65rem; font-weight: 700; letter-spacing: -.01em; color: white; }}
  .hero p  {{ margin: 4px 0 0; opacity: .88; font-size: .93rem; }}
  .pill {{ display:inline-block; background: rgba(255,255,255,.18); border-radius: 999px;
           padding: 2px 12px; font-size: .78rem; margin: 8px 6px 0 0; }}
  .kpi {{ background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px 16px;
          box-shadow: 0 1px 2px rgba(15,23,42,.04); height: 152px; box-sizing: border-box; overflow: hidden;
          display: flex; flex-direction: column; justify-content: space-between; }}
  .kpi .sub {{ display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }}
  .kpi .lbl {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: {MUTED}; font-weight: 600; }}
  .kpi .val {{ font-size: clamp(1.3rem, 1.9vw, 1.75rem); font-weight: 700; color: {SLATE}; line-height: 1.25; white-space: nowrap; }}
  .kpi .sub {{ font-size: .78rem; color: {MUTED}; }}
  .kpi.accent {{ border-left: 4px solid {BLUE}; }}  .kpi.warn {{ border-left: 4px solid {AMBER}; }}
  .kpi.bad {{ border-left: 4px solid {RED}; }}
  h3 {{ font-size: 1.05rem !important; color: {SLATE}; margin-top: .2rem !important; }}
  .note {{ background: #EFF6FF; border: 1px solid #BFDBFE; color: #1E3A8A; padding: 10px 14px;
           border-radius: 10px; font-size: .86rem; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
  .stTabs [data-baseweb="tab"] {{ background: #F1F5F9; border-radius: 8px 8px 0 0; padding: 8px 16px; }}
  .stTabs [aria-selected="true"] {{ background: white; border-bottom: 3px solid {BLUE}; }}
</style>""", unsafe_allow_html=True)


def kpi(label, value, sub="", tone="accent"):
    return f'<div class="kpi {tone}"><div class="lbl">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'


def style(fig, h=380, legend=True):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=36, b=10), font=dict(family=FONT, size=12, color=SLATE),
                      paper_bgcolor="white", plot_bgcolor="white", showlegend=legend,
                      legend=dict(orientation="h", y=-0.18), hoverlabel=dict(font_size=12))
    fig.update_xaxes(gridcolor="#EEF2F6", zerolinecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#EEF2F6", zerolinecolor="#E2E8F0")
    return fig


# ----------------------------------------------------------------- data --
@st.cache_resource(show_spinner="Loading Bengaluru data...")
def get_city():
    return E.City()


@st.cache_resource
def get_geojson():
    with open(os.path.join(E.TABLEAU, "hexagons_bengaluru.geojson"), encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def cached_run(P, _city_id=0):
    return E.run(P, get_city())


@st.cache_data(show_spinner=False)
def cached_sweep(P, promises, n_max):
    return E.coverage_sweep(P, get_city(), list(promises), n_max)


@st.cache_data(show_spinner=False)
def cached_cov(P, promise, n):
    return E.base_coverage(P, get_city(), promise=promise, n=n)


@st.cache_data(show_spinner="Running what-if cases...")
def cached_sens(P):
    return E.sensitivity(P, get_city())


@st.cache_data(show_spinner="Re-running the model for each input (about 10 s in cost mode)...")
def cached_tornado(P, metric, pct):
    return E.tornado(P, get_city(), metric, pct)


@st.cache_data(show_spinner="Solving the break-even grid...")
def cached_grid(P, f_mults, lost_vals):
    return E.cost_grid(P, get_city(), list(f_mults), list(lost_vals))


@st.cache_data(show_spinner="Re-running at each noise level...")
def cached_vol(P, which, values, n_scen):
    return E.volatility_sweep(P, get_city(), which, list(values), n_scen)


city = get_city()
gj = get_geojson()

# -------------------------------------------------------------- sidebar --
D = E.DEFAULTS


def set_vol():
    presets = {"Low": (0.10, 0.05, 0.0), "Medium": (0.25, 0.10, 0.0),
               "High": (0.40, 0.20, 0.10), "Festival / rain shock": (0.25, 0.10, 0.25)}
    a, b, c = presets[st.session_state["vol_preset"]]
    st.session_state.update(sig_local=a, sig_ward=b, sig_city=c)


with st.sidebar:
    st.markdown("### 📦 Simulation inputs")
    mode_lbl = st.radio("Model", ["Coverage (Step 20)", "Full cost model (Step 21)"], key="mode_lbl",
                        help="Coverage: choose N stores to reach the most demand in time. "
                             "Cost: choose how many stores pay for themselves, with rent, riders, lateness and lost orders.")
    mode = "coverage" if mode_lbl.startswith("Coverage") else "cost"

    with st.expander("1 · Network decision", expanded=True):
        st.caption("T and the number of stores are set in the bar at the top of the page.")
        if mode == "coverage":
            method = st.radio("Store-selection method", ["exact", "greedy"], horizontal=True,
                              format_func=lambda x: "Optimal (MILP)" if x == "exact" else "Greedy")
            force_n = False
        else:
            force_n = st.checkbox("Force exactly N_max stores", False,
                                  help="Off: the model stops opening stores once the next one costs more than it earns.")
            method = "greedy"
        drop_shops = st.checkbox("Exclude existing shops as candidate sites", False)

    with st.expander("2 · Random demand", expanded=True):
        st.caption("Each scenario multiplies every hexagon's population by a random factor (mean 1).")
        n_scen = st.slider("Number of scenarios", 50, 3000, 500, step=50)
        st.selectbox("Volatility preset", ["Low", "Medium", "High", "Festival / rain shock"], index=1,
                     key="vol_preset", on_change=set_vol)
        dist = st.selectbox("Distribution", ["lognormal", "uniform", "normal"],
                            format_func=lambda x: {"lognormal": "Lognormal (never negative)",
                                                   "uniform": "Uniform around 1", "normal": "Normal (truncated)"}[x])
        sig_local = st.slider("Per-hexagon σ", 0.0, 0.8, 0.25, 0.01, key="sig_local",
                              help="Independent noise in every hexagon.")
        sig_ward = st.slider("Per-ward σ (shared)", 0.0, 0.6, 0.10, 0.01, key="sig_ward",
                             help="One shared shock per ward: neighbouring hexagons move together.")
        sig_city = st.slider("City-wide σ (shared)", 0.0, 0.6, 0.0, 0.01, key="sig_city",
                             help="One shock for the whole city: festival, rain, strike.")
        seed = st.number_input("Random seed", 0, 99999, 42)

    with st.expander("3 · Time & demand assumptions"):
        tau_p = st.number_input("Packing time τp (min)", 0.0, 10.0, 2.0, 0.5)
        tau_l = st.number_input("Last stretch τl (min)", 0.0, 10.0, 2.0, 0.5)
        speed = st.slider("Rider speed multiplier μ", 0.6, 1.4, 1.0, 0.05)
        hh = st.number_input("Household size h", 1.0, 8.0, 3.6, 0.1)
        phi = st.slider("Share of households ordering φ", 0.05, 0.9, 0.30, 0.05)
        lam = st.slider("Orders / household / week λ", 0.5, 10.0, 3.0, 0.5)
        wk = st.slider("Weekday share of orders", 0.5, 0.9, 0.70, 0.05)
        pk = st.slider("Peak share of a day", 0.1, 0.6, 0.35, 0.05)
        pkh = st.slider("Peak window (hours)", 1, 6, 3)

    with st.expander("4 · Store capacity"):
        v = st.number_input("Orders per rider per hour v", 1.0, 15.0, 6.0, 0.5)
        rs = st.number_input("Riders · small store", 1, 40, 5)
        rl = st.number_input("Riders · large store", 1, 60, 12)

    if mode == "cost":
        with st.expander("5 · Costs (₹, illustrative)", expanded=True):
            st.caption("No real cost data yet: see Step 22 of the derivation. Read shapes, not levels.")
            f_s = st.number_input("Weekly fixed cost · small store", 0, 1_000_000, 60_000, 5_000)
            f_l = st.number_input("Weekly fixed cost · large store", 0, 1_000_000, 150_000, 5_000)
            c_proc = st.number_input("Processing per order", 0.0, 100.0, 10.0, 1.0)
            c_rider = st.number_input("Rider cost per minute", 0.0, 20.0, 2.0, 0.5)
            beta = st.slider("Orders per rider trip β", 1.0, 4.0, 1.0, 0.25)
            c_late = st.number_input("Lateness penalty per minute", 0.0, 50.0, 5.0, 1.0)
            c_lost = st.number_input("Lost-order cost", 0.0, 500.0, 100.0, 10.0)
            tmax = st.slider("Delivery cut-off: T + … min", 0, 10, 5)
    else:
        f_s, f_l, c_proc, c_rider, beta, c_late, c_lost, tmax = (D["f_small"], D["f_large"], D["c_proc"],
                                                                 D["c_rider"], D["beta"], D["c_late"], D["c_lost"],
                                                                 D["tmax_extra"])

    with st.expander("6 · Robustness check"):
        n_pick = st.slider("Scenarios to re-optimise", 0, 500, 500 if mode == "coverage" else 30, 10,
                           help="Re-choose the stores inside each of these scenarios (greedy). Feeds the live site-selection "
                                "map and the most-picked map. Cost mode is slower, so it defaults to 30.")

    run_clicked = st.button("▶  Run simulation", type="primary", width='stretch')

# ---- main-page control bar: the two decisions people change most ----
bar = st.columns([1.1, 1.1, 0.9, 2.5])
promise = bar[0].number_input("Delivery promise T (minutes)", 4, 40, 10, 1, key="T_in",
                              help="Any whole number of minutes. Coverage collapses below ~8 min and saturates above ~20.")
if mode == "coverage":
    n_stores = bar[1].number_input("Number of stores to open N", 1, 400, 10, 1, key="N_cov")
else:
    n_stores = bar[1].number_input("Maximum stores N_max", 10, 1000, 400, 10, key="N_cost")
bar[2].markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
run_clicked = bar[2].button("▶  Run", type="primary", width="stretch", key="run_bar") or run_clicked
bar[3].markdown("<div style='height:1.9rem;color:#64748B;font-size:.85rem'>Change T or N, then press Run "
                "(or the button in the sidebar). Results below update for the whole dashboard.</div>",
                unsafe_allow_html=True)

P = dict(mode=mode, promise=promise, n_stores=n_stores, method=method, drop_shops=drop_shops, force_n=force_n,
         tau_p=tau_p, tau_l=tau_l, speed=speed, hh_size=hh, phi=phi, lam=lam, weekday_share=wk, peak_share=pk,
         peak_hours=pkh, orders_per_rider_hr=v, riders_small=rs, riders_large=rl, n_scen=n_scen, seed=int(seed),
         dist=dist, sigma_local=sig_local, sigma_ward=sig_ward, sigma_city=sig_city, n_pick=n_pick,
         c_proc=c_proc, c_rider=c_rider, c_late=c_late, c_lost=c_lost, f_small=f_s, f_large=f_l, beta=beta,
         tmax_extra=tmax)

if run_clicked or "res" not in st.session_state:
    with st.spinner(f"Simulating {n_scen:,} demand scenarios..."):
        try:
            st.session_state["res"] = cached_run(P)
            st.session_state["P_run"] = P
            st.session_state.pop("err", None)
        except ValueError as ex:
            st.session_state["err"] = str(ex)

if "err" in st.session_state:
    st.error(st.session_state["err"])
if "res" not in st.session_state:
    st.stop()

res: E.SimResult = st.session_state["res"]
PR = st.session_state["P_run"]
scen, chosen, hexdf = res.scen, res.chosen, res.hexdf
is_cost = PR["mode"] == "cost"

# ----------------------------------------------------------------- hero --
st.markdown(f"""
<div class="hero">
  <h1>Bengaluru Dark-Store Planner · Random-Demand Simulator</h1>
  <p>Stores are chosen once on expected demand, then stress-tested against {PR['n_scen']:,} random demand scenarios.</p>
  <span class="pill">{'Cost model' if is_cost else 'Coverage model'}</span>
  <span class="pill">{PR['promise']}-min promise</span>
  <span class="pill">{len(chosen)} stores open</span>
  <span class="pill">σ local {PR['sigma_local']:.2f} · ward {PR['sigma_ward']:.2f} · city {PR['sigma_city']:.2f}</span>
  <span class="pill">solved in {res.meta['seconds']:.1f}s · {res.meta['status']}</span>
</div>""", unsafe_allow_html=True)

if PR != P:
    st.warning("Inputs changed since the last run: press **Run simulation** to update every chart.")

# ------------------------------------------------------------------ KPIs --
sp = scen.served_pct
p5, p95 = np.percentile(sp, 5), np.percentile(sp, 95)
full_share = 100 * (scen.stores_over_capacity > 0).mean()
cols = st.columns(5)
cols[0].markdown(kpi("Expected demand reached", f"{sp.mean():.1f}%",
                     f"between {p5:.1f}% and {p95:.1f}% in 90% of scenarios"), unsafe_allow_html=True)
cols[1].markdown(kpi("Worst 5% of scenarios", f"{p5:.1f}%", f"std. deviation {sp.std():.2f} pts",
                     "warn"), unsafe_allow_html=True)
if is_cost:
    ct = scen.cost_total
    cols[2].markdown(kpi("Expected weekly cost", f"₹{ct.mean() / 1e7:.2f} cr",
                         f"₹{ct.mean() / scen.orders_served.mean():.0f} per order served"), unsafe_allow_html=True)
    cols[3].markdown(kpi("Worst-5% weekly cost", f"₹{np.percentile(ct, 95) / 1e7:.2f} cr",
                         f"{100 * (np.percentile(ct, 95) / ct.mean() - 1):+.1f}% vs expected", "bad"), unsafe_allow_html=True)
else:
    cols[2].markdown(kpi("Stores open", f"{len(chosen)}", f"{100 * hexdf.covered.mean():.0f}% of hexagons reached"),
                     unsafe_allow_html=True)
    if res.reopt is not None:
        regret = max(0.0, res.reopt.mean() - res.reopt_base.mean())
        cols[3].markdown(kpi("Cost of a fixed network", f"{regret:.2f} pts",
                             "coverage lost vs re-choosing sites every scenario", "bad"), unsafe_allow_html=True)
    else:
        cols[3].markdown(kpi("Orders / week", f"{scen.total_orders_week.mean() / 1e6:.2f} M", "expected demand"),
                         unsafe_allow_html=True)
if is_cost:
    ont, srv = scen.ontime_pct.mean(), scen.served_pct.mean()
    cols[4].markdown(kpi("Orders delivered on time", f"{ont:.1f}%",
                         f"{srv - ont:.1f}% late · {100 - srv:.1f}% lost"), unsafe_allow_html=True)
else:
    people = float((hexdf.pop_sim_mean * hexdf.covered).sum())
    cols[4].markdown(kpi("Residents within reach", f"{people / 1e6:.2f} M",
                         f"of {hexdf.pop_sim_mean.sum() / 1e6:.1f} M residents"), unsafe_allow_html=True)
st.markdown('<div class="note" style="margin:10px 0 4px">Preliminary results: ordering rates, rider capacity and rupee costs are '
            'placeholders until the survey and rider interviews are in; 3 of 934 hexagons have no road times yet. '
            'Read the shape of the results, not the absolute levels.</div>', unsafe_allow_html=True)
st.write("")

tabs = st.tabs(["Summary", "Sites", "Market", "Trade-offs", "Sensitivity"] + (["Costs"] if is_cost else []))
t_over, t_sites, t_market, t_curve, t_sens = tabs[:5]
t_cost = tabs[5] if is_cost else None
# sections that share a tab are simply written one after the other
t_compmap = t_comp = t_market        # Market: competitor map, then access gaps
t_map = t_rob = t_sites              # Sites: live maps, map explorer, then stability

# ------------------------------------------------------- site selection --
HEX_TRACE = dict(geojson=gj, featureidkey="properties.hex_id", marker_line_width=.2, marker_line_color="white")
MAP_LAYOUT = dict(style="carto-positron", center=dict(lat=12.97, lon=77.59), zoom=9.5)


def live_map_html(res, height=600):
    """Browser-side animation: one re-optimised scenario per second, looping forever."""
    xy = res.sites_xy
    fig = go.Figure()
    fig.add_trace(go.Choroplethmap(locations=hexdf.hex_id, z=hexdf.population, colorscale=[[0, "#F1F5F9"], [1, "#BFD3F2"]],
                                   marker_opacity=.55, showscale=False, hoverinfo="skip", showlegend=False, **HEX_TRACE))
    fig.add_trace(go.Scattermap(lat=[], lon=[], mode="markers", name="picked so far (size = how often)",
                                marker=dict(size=[], color=AMBER, opacity=.6), hoverinfo="skip"))
    fig.add_trace(go.Scattermap(lat=[], lon=[], mode="markers", name="picked in this scenario",
                                marker=dict(size=13, color=RED), hoverinfo="skip"))
    fig.update_layout(map=MAP_LAYOUT, height=height, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="v", x=0.01, y=0.99, bgcolor="rgba(255,255,255,.85)"))
    js ="""
var gd = document.getElementById('{plot_id}');
var LAT=%s, LON=%s, SETS=%s, N=SETS.length, i=0, delay=1000, counts=new Array(LAT.length).fill(0), timer=null, playing=true;
var lbl=document.getElementById('lbl'), pp=document.getElementById('pp'), spd=document.getElementById('spd');
function step(){
  if(i===0){ counts.fill(0); }
  var s=SETS[i]; s.forEach(function(j){ counts[j]++; });
  var la=[],lo=[],sz=[];
  for(var j=0;j<counts.length;j++){ if(counts[j]>0){ la.push(LAT[j]); lo.push(LON[j]); sz.push(5+16*Math.sqrt(counts[j]/(i+1))); } }
  Plotly.restyle(gd,{lat:[la],lon:[lo],'marker.size':[sz]},[1]);
  Plotly.restyle(gd,{lat:[s.map(function(j){return LAT[j];})],lon:[s.map(function(j){return LON[j];})]},[2]);
  lbl.textContent='Scenario '+(i+1)+' / '+N+'  ·  '+s.length+' sites opened  ·  '+la.length+' different sites seen so far';
  i=(i+1)%%N;
}
function run(){ clearInterval(timer); if(playing){ timer=setInterval(step,delay); } }
pp.onclick=function(){ playing=!playing; pp.textContent=playing?'Pause':'Play'; run(); };
spd.onchange=function(){ delay=+spd.value; run(); };
step(); run();
""" % (json.dumps(xy.lat.round(5).tolist()), json.dumps(xy.lon.round(5).tolist()), json.dumps(res.scen_sites))
    controls = ("<div style=\"font-family:Inter,Segoe UI,sans-serif;font-size:14px;display:flex;align-items:center;gap:14px;"
                "margin:2px 0 8px\"><button id=\"pp\" style=\"background:#1D4ED8;color:#fff;border:0;border-radius:8px;"
                "padding:6px 16px;cursor:pointer;font-weight:600\">Pause</button>"
                "<label>Speed <select id=\"spd\"><option value=\"1000\" selected>1 s per scenario</option>"
                "<option value=\"500\">0.5 s</option><option value=\"250\">0.25 s</option><option value=\"2000\">2 s</option></select></label>"
                "<span id=\"lbl\" style=\"font-weight:600;color:#1E293B\"></span></div>")
    return controls + fig.to_html(full_html=False, include_plotlyjs="cdn", post_script=js, config={"scrollZoom": True})


with t_sites:
    if not res.scen_sites:
        st.info("Set 'Scenarios to re-optimise' above 0 in the sidebar (section 6) and run again.")
    else:
        n_anim = len(res.scen_sites)
        st.markdown(f"### Live site selection: {n_anim} scenarios, one per second")
        st.caption("In each scenario the demand is redrawn and the stores are re-chosen. Red = sites opened in that scenario; "
                   "amber = every site seen so far, bigger the more often it is chosen. The loop restarts after the last scenario.")
        if hasattr(st, "iframe"):                       # newer Streamlit
            st.iframe(live_map_html(res), height=680)
        else:                                           # older Streamlit
            components.html(live_map_html(res), height=680)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Most picked sites")
            pk_ = res.pick
            fig = go.Figure(go.Choroplethmap(locations=hexdf.hex_id, z=hexdf.population,
                                             colorscale=[[0, "#F1F5F9"], [1, "#DCE7F8"]], marker_opacity=.5,
                                             showscale=False, hoverinfo="skip", **HEX_TRACE))
            fig.add_trace(go.Scattermap(lat=pk_.lat, lon=pk_.lon, mode="markers",
                                        marker=dict(size=7 + 16 * pk_.pick_share, color=100 * pk_.pick_share,
                                                    colorscale="YlOrRd", cmin=0, cmax=100, opacity=.85,
                                                    colorbar=dict(title="% of<br>scenarios", thickness=10, len=.5)),
                                        hovertext=[f"{s}: picked in {100 * x:.0f}% of scenarios"
                                                   for s, x in zip(pk_.site_id, pk_.pick_share)], showlegend=False))
            fig.update_layout(map=MAP_LAYOUT, height=560, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown("### Final selected locations")
            fig = go.Figure(go.Choroplethmap(locations=hexdf.hex_id, z=hexdf.service_rate * 100,
                                             colorscale=[[0, "#F1F5F9"], [1, "#93B4F0"]], marker_opacity=.6,
                                             showscale=False, hoverinfo="skip", **HEX_TRACE))
            fmt_col = {"large": RED, "small": AMBER, "standard": RED}
            for f_, g in chosen.groupby("format"):
                fig.add_trace(go.Scattermap(lat=g.lat, lon=g.lon, mode="markers", name=f_ if is_cost else "selected store",
                                            marker=dict(size=13, color=fmt_col.get(f_, RED)),
                                            hovertext=[f"{r.site_id} · {r.format} · expected peak {r.expected_peak_orders_hr:.0f} orders/hr"
                                                       for r in g.itertuples()]))
            fig.update_layout(map=MAP_LAYOUT, height=560, margin=dict(l=0, r=0, t=0, b=0),
                              legend=dict(orientation="v", x=0.01, y=0.99, bgcolor="rgba(255,255,255,.85)"))
            st.plotly_chart(fig, width="stretch")
        core = int((res.pick.pick_share >= .9).sum())
        overlap = int(res.pick.head(len(chosen)).site_id.isin(chosen.site_id).sum())
        st.markdown(f'<div class="note"><b>{core}</b> sites are chosen in at least 90% of scenarios. '
                    f'<b>{overlap} of the {len(chosen)}</b> most-picked sites are also in the final network '
                    f'(the final network is chosen once on expected demand).</div>', unsafe_allow_html=True)

# ------------------------------------------------------- competitor map --
with t_compmap:
    comp = city.competitors
    st.markdown("### Competitor dark stores in Bengaluru")
    st.caption("Source: data/competitor_stores_bengaluru.csv (from your Google My Maps KML, collected Sep-Oct 2025).")
    c1, c2 = st.columns([1, 3])
    with c1:
        sel_p = st.multiselect("Platforms", list(PLAT), default=list(PLAT), key="cm_plat")
        show_us = st.checkbox("Show our selected sites", True, key="cm_us")
        show_pop = st.checkbox("Shade by population", True, key="cm_pop")
        rad = st.slider("'Near a competitor' radius (km)", 0.5, 3.0, 1.0, 0.25)
        cf = comp[comp.platform.isin(sel_p)]
        for p_ in sel_p:
            st.markdown(kpi(p_, f"{(comp.platform == p_).sum()}", "stores", "accent"), unsafe_allow_html=True)
        if len(cf):
            la1, lo1 = np.radians(chosen.lat.values)[:, None], np.radians(chosen.lon.values)[:, None]
            la2, lo2 = np.radians(cf.lat.values)[None, :], np.radians(cf.lon.values)[None, :]
            a_ = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
            dkm = 6371 * 2 * np.arcsin(np.sqrt(a_))
            near_share = 100 * (dkm.min(1) <= rad).mean()
            st.markdown(kpi("Our sites near a competitor", f"{near_share:.0f}%", f"within {rad} km of a selected platform",
                            "warn"), unsafe_allow_html=True)
    with c2:
        fig = go.Figure()
        if show_pop:
            fig.add_trace(go.Choroplethmap(locations=hexdf.hex_id, z=hexdf.population, colorscale=[[0, "#F8FAFC"], [1, "#C7D8F5"]],
                                           marker_opacity=.55, showscale=False, hoverinfo="skip", **HEX_TRACE))
        for p_ in sel_p:
            g = comp[comp.platform == p_]
            fig.add_trace(go.Scattermap(lat=g.lat, lon=g.lon, mode="markers", name=p_, marker=dict(size=8, color=PLAT[p_], opacity=.9),
                                        hovertext=g.store_name))
        if show_us:
            fig.add_trace(go.Scattermap(lat=chosen.lat, lon=chosen.lon, mode="markers", name="our selected sites",
                                        marker=dict(size=14, color=RED),
                                        hovertext=[f"{r.site_id} · {r.format}" for r in chosen.itertuples()]))
        fig.update_layout(map=MAP_LAYOUT, height=660, margin=dict(l=0, r=0, t=0, b=0),
                          legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, width="stretch")

# -------------------------------------------------------------- overview --
def build_brief():
    """plain-English findings written from the numbers on screen"""
    T_, N_ = PR["promise"], len(chosen)
    out = []
    if not is_cost:
        now = cached_cov(PR, T_, PR["n_stores"])
        gT = cached_cov(PR, T_ + 2, PR["n_stores"]) - now
        gN = cached_cov(PR, T_, PR["n_stores"] * 2) - now
        marg = now - cached_cov(PR, T_, max(1, PR["n_stores"] - 1))
        people = float((hexdf.pop_sim_mean * hexdf.covered).sum())
        out.append(("Where we land",
                    f"{N_} stores with a {T_}-minute promise reach <b>{sp.mean():.1f}%</b> of expected weekly demand "
                    f"(about {people / 1e6:.2f} M residents). Even in the worst 5% of demand scenarios coverage stays at "
                    f"<b>{p5:.1f}%</b>, so the plan is not fragile to demand swings."))
        if gT >= gN:
            lever = (f"Relaxing the promise by 2 minutes adds <b>{gT:.1f} points</b>; doubling the network to "
                     f"{PR['n_stores'] * 2} stores adds only <b>{gN:.1f}</b>. The delivery promise is the stronger lever.")
        else:
            lever = (f"Doubling the network to {PR['n_stores'] * 2} stores adds <b>{gN:.1f} points</b>; relaxing the promise by "
                     f"2 minutes adds <b>{gT:.1f}</b>. Store count is the stronger lever at this promise.")
        out.append(("What moves the result", lever))
        extra = ""
        if res.pick is not None and len(res.pick):
            core = int((res.pick.pick_share >= .9).sum())
            reg = max(0.0, res.reopt.mean() - res.reopt_base.mean()) if res.reopt is not None else 0
            extra = (f" {core} sites are chosen in at least 90% of scenarios, and re-choosing sites each week would add only "
                     f"{reg:.2f} points, so the network can be fixed up front.")
        out.append(("How much more is worth it",
                    f"The last store adds <b>{marg:.2f} points</b> of coverage.{extra}"))
    else:
        ct_ = scen.cost_total
        shares = {k: scen[c].mean() / ct_.mean() for k, c in
                  {"rent": "cost_fixed", "rider delivery": "cost_transport", "processing": "cost_proc",
                   "lateness": "cost_late", "lost orders": "cost_lost"}.items()}
        top = sorted(shares.items(), key=lambda kv: -kv[1])[:2]
        out.append(("Where we land",
                    f"The model opens <b>{N_}</b> stores (limit {PR['n_stores']}) at a {T_}-minute promise. Expected cost is "
                    f"<b>₹{ct_.mean() / 1e7:.2f} cr a week</b>, ₹{ct_.mean() / scen.orders_served.mean():.0f} per order served, "
                    f"with {scen.ontime_pct.mean():.1f}% of orders on time and {100 - scen.served_pct.mean():.1f}% lost."))
        out.append(("Where the money goes",
                    f"{top[0][0].capitalize()} is {100 * top[0][1]:.0f}% of cost and {top[1][0]} is {100 * top[1][1]:.0f}%. "
                    f"These are the two numbers to firm up first with real quotes."))
        out.append(("Demand risk",
                    f"In the worst 5% of demand weeks cost is <b>{100 * (np.percentile(ct_, 95) / ct_.mean() - 1):+.1f}%</b> "
                    f"versus the average (₹{np.percentile(ct_, 95) / 1e7:.2f} cr)."))
    return out


with t_over:
    brief = build_brief()
    st.markdown("### Executive summary")
    bc = st.columns(len(brief))
    for col, (ttl, body) in zip(bc, brief):
        col.markdown(f'<div class="kpi accent" style="height:auto;min-height:150px"><div class="lbl">{ttl}</div>'
                     f'<div style="font-size:.93rem;color:{SLATE};line-height:1.45;margin-top:6px">{body}</div></div>',
                     unsafe_allow_html=True)
    plain = "\n\n".join(f"{t_}: {re.sub('<[^>]+>', '', b_)}" for t_, b_ in brief)
    st.download_button("Download brief (text)", f"Dark-store simulation brief\n\n{plain}\n", "brief.txt")
    st.write("")
    a, b = st.columns([3, 2])
    with a:
        st.markdown("### How much demand is reached? (all scenarios)")
        fig = go.Figure(go.Histogram(x=sp, nbinsx=40, marker_color=BLUE, opacity=.9, name="scenarios"))
        fig.add_vline(x=sp.mean(), line_color=SLATE, line_dash="dash",
                      annotation_text=f"mean {sp.mean():.1f}%", annotation_position="top")
        fig.add_vline(x=p5, line_color=RED, line_dash="dot", annotation_text=f"P5 {p5:.1f}%",
                      annotation_position="top left")
        fig.update_xaxes(title="% of weekly orders reached in time")
        fig.update_yaxes(title="scenarios")
        st.plotly_chart(style(fig, 360, False), width='stretch')
    with b:
        st.markdown("### Does a busier week change coverage?")
        fig = go.Figure(go.Scatter(x=scen.total_orders_week / 1e6, y=sp, mode="markers",
                                   marker=dict(color=scen.total_orders_week, colorscale="Blues", size=6, opacity=.7),
                                   hovertemplate="orders %{x:.3f} M<br>reached %{y:.2f}%<extra></extra>"))
        fig.update_xaxes(title="total orders that week (millions)")
        fig.update_yaxes(title="% reached")
        st.plotly_chart(style(fig, 360, False), width='stretch')

    a, b = st.columns(2)
    with a:
        st.markdown("### What the random draw looks like")
        h = hexdf.sort_values("population").reset_index(drop=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h.index, y=h.pop_sim_p95, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=h.index, y=h.pop_sim_p5, fill="tonexty", fillcolor="rgba(29,78,216,.16)",
                                 line=dict(width=0), name="5th-95th percentile"))
        fig.add_trace(go.Scatter(x=h.index, y=h.population, line=dict(color=SLATE, width=2), name="base population"))
        fig.update_xaxes(title="hexagons, sorted by base population")
        fig.update_yaxes(title="population")
        st.plotly_chart(style(fig, 340), width='stretch')
    with b:
        st.markdown("### Scenario results (first 200)")
        keep = ["scenario", "population", "total_orders_week", "served_pct"] + (["cost_total"] if is_cost else ["peak_util_max"])
        st.dataframe(scen[keep].head(200).round(2), height=340, width='stretch', hide_index=True)

# ------------------------------------------------------------------ map --
with t_map, st.expander("Map explorer: shade the city by population, demand spread, service or competitor distance"):
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown("### Layers")
        layer = st.radio("Shade hexagons by", ["Base population", "Simulated demand spread (P95 − P5)",
                                               "Reached in time" if not is_cost else "Share of demand served",
                                               "Nearest competitor distance"])
        show_ours = st.checkbox("Our stores", True)
        show_pick = st.checkbox("Sites picked when re-optimising", False,
                                disabled=res.pick is None) if res.pick is not None else False
        comp_sel = st.multiselect("Competitor stores", list(PLAT), default=[])
        st.markdown('<div class="note">Hexagons are H3 cells (~0.76 km²). Shading uses the same data as the model.</div>',
                    unsafe_allow_html=True)
    with c2:
        m = hexdf.merge(city.access[["hex_id", "nearest_store_km"]], on="hex_id")
        if layer == "Base population":
            z, cs, ttl = m.population, "Blues", "population"
        elif layer.startswith("Simulated"):
            z, cs, ttl = m.pop_sim_p95 - m.pop_sim_p5, "Oranges", "P95 − P5"
        elif layer.startswith("Nearest"):
            z, cs, ttl = m.nearest_store_km, "Purples", "km"
        else:
            z, cs, ttl = m.service_rate * 100, [[0, "#F1F5F9"], [1, BLUE]], "% served"
        fig = go.Figure(go.Choroplethmap(geojson=gj, locations=m.hex_id, z=z, featureidkey="properties.hex_id",
                                         colorscale=cs, marker_opacity=.72, marker_line_width=.3,
                                         marker_line_color="white", colorbar=dict(title=ttl, thickness=12, len=.6),
                                         customdata=np.c_[m.ward_name, m.population],
                                         hovertemplate="%{customdata[0]}<br>population %{customdata[1]:,}<br>"
                                                       + ttl + " %{z:,.1f}<extra></extra>"))
        if comp_sel:
            cp = city.competitors[city.competitors.platform.isin(comp_sel)]
            for p_, g in cp.groupby("platform"):
                fig.add_trace(go.Scattermap(lat=g.lat, lon=g.lon, mode="markers", name=p_,
                                            marker=dict(size=6, color=PLAT[p_]), hovertext=g.store_name))
        if show_pick and res.pick is not None:
            pk_ = res.pick.head(80)
            fig.add_trace(go.Scattermap(lat=pk_.lat, lon=pk_.lon, mode="markers", name="re-optimised picks",
                                        marker=dict(size=8 + 14 * pk_.pick_share, color=AMBER, opacity=.8),
                                        hovertext=[f"{s}: chosen in {100 * x:.0f}% of scenarios" for s, x in
                                                   zip(pk_.site_id, pk_.pick_share)]))
        if show_ours:
            fig.add_trace(go.Scattermap(lat=chosen.lat, lon=chosen.lon, mode="markers", name="our stores",
                                        marker=dict(size=11, color=RED),
                                        hovertext=[f"{r.site_id} · {r.format}" for r in chosen.itertuples()]))
        fig.update_layout(map=dict(style="carto-positron", center=dict(lat=12.97, lon=77.59), zoom=9.7),
                          height=640, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------- curves --
with t_curve:
    PRc = {**PR, "n_stores": len(chosen)}      # in cost mode: the number of stores the model actually opened
    st.markdown("### Coverage vs number of stores, one line per promise")
    st.caption("Greedy network for each promise, evaluated on the same random demand. Band = 5th to 95th percentile of scenarios.")
    c1, c2 = st.columns([1, 3])
    with c1:
        T_run, N_run = PR["promise"], len(chosen)
        prom_sel = st.multiselect("Promises (min)", sorted({6, 8, 10, 12, 15, 20, T_run}),
                                  default=sorted({8, 10, 12, 15, T_run}))
        nmax = st.slider("Up to N stores", 5, 400, int(min(400, max(30, 2 * N_run))))
        target = st.slider("Target coverage (%)", 50, 95, 80)
    sweep = cached_sweep(PRc, tuple(prom_sel), nmax) if prom_sel else pd.DataFrame()
    with c2:
        if sweep.empty:
            st.info("Pick at least one promise.")
        else:
            fig = go.Figure()
            pal = ["#1D4ED8", "#0EA5E9", "#F59E0B", "#DC2626", "#7C3AED", "#64748B"]
            for i, (t_, g) in enumerate(sweep.groupby("promise")):
                col = pal[i % len(pal)]
                fig.add_trace(go.Scatter(x=g.stores, y=g.p95, line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=g.stores, y=g.p5, fill="tonexty", line=dict(width=0), showlegend=False,
                                         fillcolor="rgba(100,116,139,.10)",
                                         hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=g.stores, y=g["mean"], name=f"{t_}-min", mode="lines+markers",
                                         line=dict(color=col, width=2.5), marker=dict(size=5)))
            fig.add_hline(y=target, line_dash="dot", line_color=MUTED, annotation_text=f"target {target}%",
                          annotation_position="bottom right")
            if N_run <= nmax:
                fig.add_vline(x=N_run, line_dash="dash", line_color=SLATE, annotation_text=f"your plan: {N_run}",
                              annotation_position="top")
            fig.update_xaxes(title="number of stores")
            fig.update_yaxes(title="% of weekly orders reached in time", range=[0, 100])
            st.plotly_chart(style(fig, 420), width='stretch')
    if not sweep.empty:
        need = []
        for t_, g in sweep.groupby("promise"):
            hit = g[g["mean"] >= target]
            need.append({"Promise (min)": t_, f"Stores needed for {target}%": str(int(hit.stores.iloc[0])) if len(hit) else f"> {nmax}",
                         f"Coverage at N={nmax}": f"{g['mean'].iloc[-1]:.1f}%"})
        st.markdown("### Minutes vs stores trade-off")
        st.dataframe(pd.DataFrame(need), hide_index=True, width='content')

    # ---- compare options side by side -------------------------------------------------
    st.divider()
    st.markdown("## Compare options side by side")
    st.caption("Edit the table: each row is a plan (promise T, stores N). Same random demand as the run above; networks are built "
               "greedily, so Option A can sit a few tenths of a point below the optimal network in the headline.")
    default_opts = pd.DataFrame({"Option": ["A · current plan", "B · relax the promise", "C · double the stores"],
                                 "T (min)": [T_run, T_run + 2, T_run], "N (stores)": [N_run, N_run, min(400, 2 * N_run)]})
    opts = st.data_editor(default_opts, hide_index=True, num_rows="fixed", width="stretch", key="opts",
                          column_config={"T (min)": st.column_config.NumberColumn(min_value=4, max_value=40, step=1),
                                         "N (stores)": st.column_config.NumberColumn(min_value=1, max_value=400, step=1)})
    opts = opts.dropna()
    if len(opts):
        Ts = tuple(sorted({int(x) for x in opts["T (min)"]}))
        sw = cached_sweep(PRc, Ts, int(opts["N (stores)"].max()))
        rows = []
        for r in opts.itertuples(index=False):
            g = sw[(sw.promise == int(r[1])) & (sw.stores == int(r[2]))]
            if len(g):
                g = g.iloc[0]
                rows.append({"Option": r[0], "T (min)": int(r[1]), "N": int(r[2]), "Expected coverage %": g["mean"],
                             "Worst 5% %": g.p5, "Best 5% %": g.p95})
        cmp_df = pd.DataFrame(rows)
        if len(cmp_df):
            a_, b_ = st.columns([3, 2])
            with a_:
                fig = go.Figure(go.Bar(x=cmp_df.Option, y=cmp_df["Expected coverage %"], marker_color=BLUE,
                                       error_y=dict(type="data", symmetric=False,
                                                    array=cmp_df["Best 5% %"] - cmp_df["Expected coverage %"],
                                                    arrayminus=cmp_df["Expected coverage %"] - cmp_df["Worst 5% %"],
                                                    color=SLATE),
                                       text=[f"{v:.1f}%" for v in cmp_df["Expected coverage %"]], textposition="outside"))
                fig.update_yaxes(title="% of weekly orders reached in time", range=[0, 105])
                st.plotly_chart(style(fig, 330, False), width="stretch")
            with b_:
                st.dataframe(cmp_df.round(1), hide_index=True, width="stretch")
                st.caption("Whiskers show the 5th to 95th percentile of demand scenarios.")

    # ---- diminishing returns ---------------------------------------------------------
    st.divider()
    st.markdown("## Diminishing returns: what does each extra store add?")
    dr = cached_sweep(PRc, (T_run,), int(min(400, max(30, 2 * N_run))))
    dr = dr.assign(add=dr["mean"].diff().fillna(dr["mean"]))
    fig = go.Figure(go.Bar(x=dr.stores, y=dr["add"], marker_color=np.where(dr.stores == N_run, AMBER, BLUE),
                           hovertemplate="store %{x}<br>adds %{y:.2f} points<extra></extra>"))
    fig.update_xaxes(title=f"store number (promise {T_run} min)")
    fig.update_yaxes(title="extra coverage (percentage points)")
    st.plotly_chart(style(fig, 300, False), width="stretch")
    st.caption("Greedy order: the first stores take the densest areas. The amber bar is your plan's last store.")

# ----------------------------------------------------------- sensitivity --
with t_sens:
    st.markdown("## Sensitivity analysis")
    st.caption("How much does the answer move if an input is wrong? Three views: one input at a time, two decisions together, "
               "and how much demand noise the plan can absorb.")
    PRc = {**PR, "n_stores": len(chosen)} if not is_cost else PR
    metric = "cost" if is_cost else "coverage"
    unit = "₹ crore per week" if is_cost else "% of weekly orders reached"

    # ---- 1. tornado ----
    st.markdown("### 1 · One input at a time (tornado)")
    c1, c2 = st.columns([1, 3])
    with c1:
        rng_pct = st.slider("Move each input by ±", 5, 50, 20, 5, format="%d%%", key="sens_pct") / 100
        st.markdown('<div class="note">Every input is lowered and raised by the same percentage. '
                    + ("The network is <b>re-chosen</b> for each case, so this includes the plan adapting."
                       if is_cost else "Coverage is independent of ordering rates and household size: it depends only on where people live and how far riders get.")
                    + "</div>", unsafe_allow_html=True)
    tor = cached_tornado(PRc, metric, rng_pct)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(y=tor.factor, x=tor.low - tor.base, orientation="h", name=f"input −{rng_pct:.0%}",
                             marker_color=AMBER, text=tor.low_label, textposition="outside", cliponaxis=False,
                             customdata=tor.low, hovertemplate="%{y} lowered to %{text}<br>result %{customdata:.2f}<extra></extra>"))
        fig.add_trace(go.Bar(y=tor.factor, x=tor.high - tor.base, orientation="h", name=f"input +{rng_pct:.0%}",
                             marker_color=BLUE, text=tor.high_label, textposition="outside", cliponaxis=False,
                             customdata=tor.high, hovertemplate="%{y} raised to %{text}<br>result %{customdata:.2f}<extra></extra>"))
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title=f"change in result vs base of {tor.base.iloc[0]:.2f} ({unit})", zeroline=True)
        st.plotly_chart(style(fig, 80 + 46 * len(tor)), width="stretch")
    top = tor.sort_values("swing", ascending=False).head(3)
    st.markdown(f'<div class="note"><b>Reading it:</b> the result is most sensitive to <b>{top.factor.iloc[0]}</b> '
                f'(a swing of {top.swing.iloc[0]:.2f}), then <b>{top.factor.iloc[1]}</b> ({top.swing.iloc[1]:.2f}) and '
                f'<b>{top.factor.iloc[2]}</b> ({top.swing.iloc[2]:.2f}). Measure these first; the rest can stay as assumptions.</div>',
                unsafe_allow_html=True)

    # ---- 2. two-way grid ----
    st.divider()
    if not is_cost:
        st.markdown("### 2 · Two decisions together: promise × number of stores")
        Ts = sorted({6, 8, 10, 12, 14, 16, 18, 20, PR["promise"]})
        nm = int(min(400, max(40, 2 * len(chosen))))
        sw = cached_sweep(PRc, tuple(Ts), nm)
        piv = sw.pivot(index="promise", columns="stores", values="mean")
        fig = go.Figure(go.Heatmap(x=piv.columns, y=piv.index, z=piv.values, colorscale="Blues", zmin=0, zmax=100,
                                   colorbar=dict(title="% reached", thickness=12),
                                   hovertemplate="T %{y} min · %{x} stores<br>%{z:.1f}% reached<extra></extra>"))
        fig.add_trace(go.Scatter(x=[len(chosen)], y=[PR["promise"]], mode="markers", name="your plan",
                                 marker=dict(symbol="star", size=16, color=AMBER, line=dict(color="white", width=1.5))))
        fig.update_xaxes(title="number of stores")
        fig.update_yaxes(title="delivery promise (min)", type="category")
        st.plotly_chart(style(fig, 400), width="stretch")
        st.caption("Read across a row for the effect of more stores, up a column for the effect of a longer promise. "
                   "Cells with similar shading are interchangeable ways of reaching the same coverage.")
    else:
        st.markdown("### 2 · Break-even: how many stores does the model open?")
        fm, lv = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0), (25, 50, 75, 100, 150, 200)
        gr = cached_grid(PR, fm, lv)
        piv = gr.pivot(index="rent_mult", columns="c_lost", values="stores")
        fig = go.Figure(go.Heatmap(x=[f"₹{c}" for c in piv.columns], y=[f"{m:.2f}× rent" for m in piv.index], z=piv.values,
                                   colorscale="Blues", text=piv.values, texttemplate="%{text}", colorbar=dict(title="stores", thickness=12),
                                   hovertemplate="%{y}, lost-order cost %{x}<br>%{z} stores<extra></extra>"))
        fig.update_xaxes(title="cost of a lost order")
        fig.update_yaxes(title="weekly rent vs your input")
        st.plotly_chart(style(fig, 360, False), width="stretch")
        st.caption("Zero means no store pays for itself. Small ripples between neighbouring cells come from the greedy search, not from the economics.")

    # ---- 3. demand noise ----
    st.divider()
    st.markdown("### 3 · How much demand noise can the plan absorb?")
    which = st.radio("Vary the spread of", ["sigma_local", "sigma_ward", "sigma_city"], horizontal=True,
                     format_func=lambda s: {"sigma_local": "each hexagon (independent)", "sigma_ward": "each ward (shared)",
                                            "sigma_city": "the whole city (shared)"}[s])
    vals = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    vs = cached_vol(PR, which, vals, 40 if is_cost else 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vs.sigma, y=vs.p95, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=vs.sigma, y=vs.p5, fill="tonexty", fillcolor="rgba(29,78,216,.16)", line=dict(width=0),
                             name="5th to 95th percentile"))
    fig.add_trace(go.Scatter(x=vs.sigma, y=vs["mean"], mode="lines+markers", name="average",
                             line=dict(color=BLUE, width=3)))
    fig.update_xaxes(title="spread of the random multiplier (σ)")
    fig.update_yaxes(title=unit)
    st.plotly_chart(style(fig, 340), width="stretch")
    d_mean = vs["mean"].iloc[-1] - vs["mean"].iloc[0]
    width = vs.p95.iloc[-1] - vs.p5.iloc[-1]
    st.markdown(f'<div class="note"><b>Reading it:</b> raising σ from 0 to 0.6 moves the average by {d_mean:+.2f} and opens a '
                f'{width:.2f}-wide 5th-to-95th percentile band. '
                + ("A flat average with a widening band means the plan is right on average but less predictable."
                   if abs(d_mean) < 0.5 * max(width, 1e-9) else
                   "An average that drifts means noise itself costs money: capacity binds in busy scenarios and cannot be recovered in quiet ones.")
                + " Other noise sources stay at the sidebar values.</div>", unsafe_allow_html=True)

# ----------------------------------------------------------- competition --
with t_comp:
    st.divider()
    st.markdown("## Access gaps and territory")
    st.caption("Real data: 406 Blinkit, Zepto and Instamart stores (late-2025 snapshot). Independent of the simulation.")
    acc = city.access.copy()
    c1, c2 = st.columns([2, 3])
    with c1:
        thr = st.slider("Gap threshold: farther than … km from nearest store", 0.5, 3.0, 1.5, 0.25)
        gaps = {p_: 100 * acc.population[acc[f"nearest_{p_.lower()}_km"] > thr].sum() / acc.population.sum() for p_ in PLAT}
        fig = go.Figure(go.Bar(x=list(gaps), y=list(gaps.values()), marker_color=list(PLAT.values()),
                               text=[f"{v:.1f}%" for v in gaps.values()], textposition="outside"))
        fig.update_yaxes(title=f"% of residents > {thr} km away", range=[0, max(gaps.values()) * 1.25])
        st.plotly_chart(style(fig, 300, False), width='stretch')
        sat = acc.groupby("platforms_within_2km").population.sum()
        sat = 100 * sat / sat.sum()
        fig = go.Figure(go.Bar(x=[f"{int(k)} platform{'s' if k != 1 else ''}" for k in sat.index], y=sat.values,
                               marker_color=BLUE, text=[f"{v:.1f}%" for v in sat.values], textposition="outside"))
        fig.update_yaxes(title="% of residents (2 km)", range=[0, sat.max() * 1.25])
        st.plotly_chart(style(fig, 300, False), width='stretch')
    with c2:
        sel = st.multiselect("Platforms on the map", list(PLAT), default=list(PLAT), key="terr")
        if sel:
            dist_m = acc[[f"nearest_{p_.lower()}_km" for p_ in sel]].values
            win = np.array(sel)[dist_m.argmin(1)]
            win = np.where(dist_m.min(1) > thr, "Gap", win)
            fig = go.Figure()
            for p_, colr in list(PLAT.items()) + [("Gap", "#CBD5E1")]:
                msk = win == p_
                if msk.any():
                    fig.add_trace(go.Choroplethmap(geojson=gj, locations=acc.hex_id[msk], z=np.ones(msk.sum()),
                                                   featureidkey="properties.hex_id", name=p_, showscale=False,
                                                   colorscale=[[0, colr], [1, colr]], marker_opacity=.75,
                                                   marker_line_width=.3, marker_line_color="white",
                                                   showlegend=True, hovertext=acc.ward_name[msk]))
            fig.update_layout(map=dict(style="carto-positron", center=dict(lat=12.97, lon=77.59), zoom=9.5),
                              height=620, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig, width='stretch')

# ------------------------------------------------------------ robustness --
with t_rob:
    st.divider()
    st.markdown("## How stable are these sites?")
    a, b = st.columns(2)
    with a:
        st.markdown("### How often is each site chosen if stores are re-picked per scenario?")
        if res.pick is None or res.pick.empty:
            st.info("Set 'Scenarios to re-optimise' above 0 in the sidebar.")
        else:
            pk_ = res.pick.head(25).copy()
            pk_["ours"] = pk_.site_id.isin(chosen.site_id)
            fig = go.Figure(go.Bar(x=pk_.site_id, y=100 * pk_.pick_share,
                                   marker_color=np.where(pk_.ours, BLUE, "#CBD5E1")))
            fig.update_yaxes(title="% of scenarios", range=[0, 105])
            fig.update_xaxes(tickangle=-60)
            st.plotly_chart(style(fig, 340, False), width='stretch')
            core = (res.pick.pick_share >= .9).sum()
            st.markdown(f'<div class="note"><b>{core}</b> sites are picked in ≥90% of scenarios (the robust core). '
                        f'Blues bars are in the network chosen on expected demand.</div>', unsafe_allow_html=True)
    with b:
        st.markdown("### Fixed network vs re-choosing every time")
        if res.reopt is not None:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=res.reopt_base, name="fixed network (greedy)", marker_color=BLUE, opacity=.75))
            fig.add_trace(go.Histogram(x=res.reopt, name="re-chosen per scenario", marker_color=AMBER, opacity=.65))
            fig.update_layout(barmode="overlay")
            fig.update_xaxes(title="% reached")
            st.plotly_chart(style(fig, 340), width='stretch')
        else:
            u = res.util
            idx = np.argsort(-np.median(u, axis=0))[:20]
            fig = go.Figure()
            for k in idx:
                fig.add_trace(go.Box(y=u[:, k], name=chosen.site_id[k], marker_color=BLUE, showlegend=False))
            fig.add_hline(y=1, line_dash="dash", line_color=RED, annotation_text="capacity")
            fig.update_yaxes(title="peak-hour utilisation")
            st.plotly_chart(style(fig, 340, False), width='stretch')
    st.markdown("### Open stores")
    st.dataframe(chosen.round(3), width='stretch', hide_index=True, height=260)

# ----------------------------------------------------------------- costs --
if is_cost:
    with t_cost:
        a, b = st.columns([2, 3])
        parts = {"Fixed (rent)": "cost_fixed", "Processing": "cost_proc", "Rider delivery": "cost_transport",
                 "Lateness": "cost_late", "Lost orders": "cost_lost"}
        cc = ["#334155", "#0EA5E9", BLUE, AMBER, RED]
        with a:
            st.markdown("### Where the money goes (average week)")
            vals = [scen[c].mean() / 1e7 for c in parts.values()]
            fig = go.Figure(go.Bar(x=list(parts), y=vals, marker_color=cc,
                                   text=[f"₹{x:.2f} cr" for x in vals], textposition="outside"))
            fig.update_yaxes(title="₹ crore per week")
            st.plotly_chart(style(fig, 360, False), width='stretch')
        with b:
            st.markdown("### Weekly cost across scenarios")
            fig = go.Figure(go.Histogram(x=scen.cost_total / 1e7, nbinsx=40, marker_color=BLUE))
            fig.add_vline(x=scen.cost_total.mean() / 1e7, line_dash="dash", line_color=SLATE, annotation_text="mean")
            fig.add_vline(x=np.percentile(scen.cost_total, 95) / 1e7, line_dash="dot", line_color=RED, annotation_text="P95")
            fig.update_xaxes(title="total weekly cost (₹ crore)")
            st.plotly_chart(style(fig, 360, False), width='stretch')
        a, b = st.columns(2)
        with a:
            st.markdown("### On time vs late vs lost")
            ont, srv = scen.ontime_pct.mean(), scen.served_pct.mean()
            fig = go.Figure(go.Bar(y=["Share of orders"], x=[ont], orientation="h", name="on time", marker_color=BLUE))
            fig.add_trace(go.Bar(y=["Share of orders"], x=[srv - ont], orientation="h", name="late", marker_color=AMBER))
            fig.add_trace(go.Bar(y=["Share of orders"], x=[100 - srv], orientation="h", name="lost", marker_color=RED))
            fig.update_layout(barmode="stack")
            fig.update_xaxes(range=[0, 100], title="% of weekly orders")
            st.plotly_chart(style(fig, 200), width='stretch')
        with b:
            st.markdown("### Store formats chosen")
            fc = chosen.format.value_counts()
            fig = go.Figure(go.Pie(labels=fc.index, values=fc.values, hole=.6,
                                   marker=dict(colors=[BLUE, AMBER])))
            st.plotly_chart(style(fig, 240), width='stretch')

# ----------------------------------------------------------------- method --
st.write("")
with st.expander("Method, data and downloads"):
    a, b = st.columns([3, 2])
    with a:
        st.markdown(f"""
### How the simulation works
1. **Random demand.** For each scenario, every hexagon's population is multiplied by `local × ward × city` random
   factors, each with mean 1 (**{PR['dist']}**, σ = {PR['sigma_local']:.2f} / {PR['sigma_ward']:.2f} / {PR['sigma_city']:.2f}).
   Population → households (÷ {PR['hh_size']}) → weekly orders (× φ × λ) → peak-hour orders (Step 5).
2. **Stage 1: choose stores once** on expected demand (Step 21). {'Maximal covering problem solved to optimality (MILP) or greedily.' if not is_cost else 'Greedy cost search: add the store with the best saving net of rent and capacity, until none pays for itself.'}
3. **Stage 2: score every scenario.** {'A hexagon is reached if packing + road time + last stretch ≤ T (Step 4). Peak-hour load is checked against rider capacity (Step 7).' if not is_cost else 'For each scenario, solve the capacity LP over the open stores (Steps 9 to 11) and add up the five costs (Steps 13 to 17).'}
4. **Robustness.** Re-choose the stores inside each scenario to see which sites are always picked.

### Data
| | |
|---|---|
| Hexagons | {city.H} H3 cells, WorldPop population ({int(city.pop.sum()):,} people) |
| Candidate sites | {city.J:,} OSM sites ({int(city.sites.shop_only.sum())} are existing shops) |
| Road minutes | OSM routing, {int(np.isfinite(city.R).sum()):,} site-to-hexagon pairs |
| Competitors | {len(city.competitors)} stores, late-2025 snapshot |

**Caveats.** φ, λ, riders per store and all rupee costs are placeholders until the survey and rider round 2.
85 BBMP hexagons still lack travel times. Read shapes, not levels.""")
    with b:
        st.markdown("### Download")
        st.download_button("Scenario results (CSV)", scen.to_csv(index=False).encode(), "scenarios.csv", width='stretch')
        st.download_button("Open stores (CSV)", chosen.to_csv(index=False).encode(), "stores.csv", width='stretch')
        st.download_button("Hexagon results (CSV)", hexdf.to_csv(index=False).encode(), "hexagons.csv", width='stretch')
        if st.checkbox("Show parameters of this run"):
            st.json(PR)
