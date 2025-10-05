
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# --- Model Parameters ---
N = 5  # number of nodes
T_max = 24 * 30  # total time in hours (30 days)
dt = 0.1
time = np.arange(0, T_max, dt)

# Initial deployment quantities per node (CSH)
INIT_RBC = 60
INIT_FFP = 40
INIT_PLT = 20
INIT_CRYO = 20
INIT_WBB = 60
INIT_CAS = 0

#packet-style model
#giving 4:4:1 rbc,plt,ffp, or 1 WB
#probably need to find a source to cite for this other than vibes.
# --- Transfusion composition & scaling ---
RBC_PER_PACKET = 4      # your current packet uses 4 RBC
FFP_PER_PACKET = 4
PLT_PER_PACKET = 1
WB_UNITS_PER_PACKET  = 4      # ~4 units WB ≈ one 4:4:1 "packet" (closer to reality than 1)
RBC_PER_PATIENT = 8     # target average RBC units per transfused patient
PACKETS_PER_PATIENT = 2   # each casualty needs 2 packets (~8 WB units)


THRESHOLD_FRACTION = 0.10
INIT_STOCK = {
    "RBC": INIT_RBC,
    "FFP": INIT_FFP,
    "PLT": INIT_PLT,
    "CRYO": INIT_CRYO,
    "WBB": INIT_WBB,
    "CAS": INIT_CAS
}

low_supply_events = {k: [[] for _ in range(N)] for k in INIT_STOCK}
wbb_overwhelmed_events = [[] for _ in range(N)]

# Deployment and resupply ratios
PROPORTIONS = {"RBC": INIT_RBC, "FFP": INIT_FFP, "PLT": INIT_PLT, "CRYO": INIT_CRYO}
TOTAL_DEPLOY = sum(PROPORTIONS.values())
RESUPPLY_RATIO = {k: v / TOTAL_DEPLOY for k, v in PROPORTIONS.items()}

# Blood constraints
B_max = 120  # I figure we only have so many freezers
WBB_RATE = 10.0  # units/hr per node
tau = 56 * 24  # donor cooldown (hours)
setup_delay = 4.0
N_total = 100 # number prescreened donors
MAX_STORAGE = 200  # Total units per node of ALL blood Products

# --- Opportunistic donor assumptions (full effort) ---
PCT_KIA = 0.25                 # 25% of casualties are KIA
PCT_KIA_SUITABLE = 0.25        # 25% of KIA physiologically suitable for donation
PCT_WIA_STABLE = 0.10          # 10% of WIA are stable enough to donate
UNITS_PER_DD = 5.0             # Units collected per deceased donor (upper realistic bound)
UNITS_PER_WIA = 1.0            # Units per stable wounded donor

# Adjacency matrix
A = np.eye(N, k=1) + np.eye(N, k=-1)
t_setup = np.random.uniform(0, setup_delay, N)

# Lat/lon (deg) for 5 CSH nodes
lat_lon = np.array([
    [49.9935, 36.2304],  # Kharkiv
    [48.5862, 38.0000],  # Bakhmut
    [47.8388, 35.1396],  # Zaporizhzhia
    [48.4647, 35.0462],  # Dnipro
    [46.6354, 32.6169]  # Kherson
])


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius (km)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


speed_kmph = 278
buffer_time = 0.167  # ~10 minutes

N = len(lat_lon)
travel_time_matrix = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        if i != j:
            dist = haversine(*lat_lon[i], *lat_lon[j])
            travel_time_matrix[i, j] = dist / speed_kmph + buffer_time

#print("Travel Time Matrix (hrs):")
#print(np.round(travel_time_matrix, 2))


def get_flight_delay(i, j):
    base_time = travel_time_matrix[i, j]
    if not np.isfinite(base_time):
        return np.inf
    jitter = np.random.uniform(0.25, 1.0)
    return base_time + jitter


blackout_windows = []
t = 0
while t < T_max:
    if np.random.rand() < 0.15:
        start = t
        duration = np.random.uniform(24, 48)
        blackout_windows.append((start, start + duration))
        t += duration
    else:
        t += 6


def in_blackout(t, blackout_windows):
    return any(start <= t <= end for start, end in blackout_windows)


resupply_schedule = [[] for _ in range(N)]
for i in range(N):
    t = 0
    while t < T_max:
        interval = np.random.uniform(12, 36)
        delay = np.random.uniform(2, 4)
        arrival = t + delay
        if in_blackout(arrival, blackout_windows):
            for window in blackout_windows:
                if window[0] <= arrival <= window[1]:
                    arrival = window[1] + 0.1
                    break
        resupply_schedule[i].append(arrival)
        t += interval

redistribution_events = []
redistribution_check_interval = 6.0
redistribution_delay_range = (2, 4)
CRITICAL_THRESHOLD = {
    "RBC": 10,
    "FFP": 5,
    "PLT": 1,
    "WBB": 5,
    "CRYO": 3
}

SAFE_THRESHOLD = {
    "RBC": 20,
    "FFP": 10,
    "PLT": 2,
    "WBB": 10,
    "CRYO": 6
}

def generate_casualties(N, t, rate_per_hour=7.5):
    """
    Stochastic WIA arrivals per node per step (Poisson).
    Baseline ≈ 7.5/hr/node; Surge ≈ 15/hr/node (days 5–10).

    Changed to 2 and 4
    """
    baseline_rate_per_hour = rate_per_hour
    surge_rate_per_hour    = rate_per_hour*2
    rate = surge_rate_per_hour if (24*5 < t < 24*10) else baseline_rate_per_hour
    lam = rate * dt  # dt = 0.1 h → λ per step
    return {"CAS": np.random.poisson(lam, size=N)}

def casualty_blood_demand(casualties):
    # patients who truly need transfusion (your same logic)
    patients = (casualties / 2.325).astype(int)# this is about 42% in weird math terms
    packets = patients * PACKETS_PER_PATIENT
    return {
        "PACKETS": packets,         # drive the loop with this
        "PATIENTS": patients,       # keep if you want to log patients
    }

EPS = dt * 0.51  # fuzzy compare for arrivals

def withdraw_from_queue(queue, needed):
    withdrawn = 0.0
    new_queue = []
    for age, qty in queue:
        if withdrawn >= needed:
            new_queue.append([age, qty])
            continue
        take = min(qty, needed - withdrawn)
        withdrawn += take
        if qty > take:
            new_queue.append([age, qty - take])
    return new_queue, withdrawn

def due_this_step(now, prev, event_time):
    return (prev < event_time) and (event_time <= now)


last_redistribution_check = -12.0
unmet_demand_log = np.zeros((N, len(time)))
EXPIRY_WBB = 24
EXPIRY_PLT = 120

wbb_queues = [[] for _ in range(N)]
plt_queues = [[] for _ in range(N)]
# ✅ FIX: Declared nu_queues as a global list of lists
nu_queues = [[] for _ in range(N)]
for i in range(N):
    if INIT_WBB > 0:
        wbb_queues[i].append([0.0, INIT_WBB])
    if INIT_PLT > 0:
        plt_queues[i].append([0.0, INIT_PLT])

# Controls whether step() logs per-timestep telemetry into the global arrays.
collect_live_metrics = True

# ✅ FIX: Moved this section up
B_init = {
    "RBC": np.full(N, INIT_RBC, dtype=float),
    "FFP": np.full(N, INIT_FFP, dtype=float),
    "PLT": np.full(N, INIT_PLT, dtype=float),
    "CRYO": np.full(N, INIT_CRYO, dtype=float),
    "WBB": np.full(N, INIT_WBB, dtype=float),
    "CAS": np.zeros(N, dtype=float),
}
NR_init = np.full(N, N_total, dtype=float)
NU_init = np.zeros(N, dtype=float)


def step(t, B, NR, NU, cumulative_wbb_generated, casualty_rate=7.5, wbb_rate=None):
    global last_redistribution_check, nu_queues
    t_prev = t - dt
    donor_rate = WBB_RATE if wbb_rate is None else wbb_rate

    # --- periodic redistribution planning ---
    if t - last_redistribution_check >= redistribution_check_interval:
        needy_nodes = {k: [] for k in ["RBC", "FFP", "PLT", "CRYO", "WBB"]}
        for i in range(N):
            cur_wbb = sum(q for _, q in wbb_queues[i])
            cur_plt = sum(q for _, q in plt_queues[i])
            if B["RBC"][i]  < CRITICAL_THRESHOLD["RBC"]:  needy_nodes["RBC"].append(i)
            if B["FFP"][i]  < CRITICAL_THRESHOLD["FFP"]:  needy_nodes["FFP"].append(i)
            if cur_plt      < CRITICAL_THRESHOLD["PLT"]:  needy_nodes["PLT"].append(i)
            if B["CRYO"][i] < CRITICAL_THRESHOLD["CRYO"]: needy_nodes["CRYO"].append(i)
            if cur_wbb      < CRITICAL_THRESHOLD["WBB"]:  needy_nodes["WBB"].append(i)

        for k in ["RBC", "FFP", "PLT", "CRYO", "WBB"]:
            for needy in needy_nodes[k]:
                best = -1; best_t = np.inf
                for donor in range(N):
                    if donor == needy: continue
                    if not np.isfinite(travel_time_matrix[donor, needy]): continue
                    if k == "WBB":
                        donor_stock = sum(q for _, q in wbb_queues[donor])
                        ok = donor_stock > SAFE_THRESHOLD["WBB"]
                    elif k == "PLT":
                        donor_stock = sum(q for _, q in plt_queues[donor])
                        ok = donor_stock > SAFE_THRESHOLD["PLT"]
                    else:
                        donor_stock = B[k][donor]
                        ok = donor_stock > SAFE_THRESHOLD[k]
                    if ok and travel_time_matrix[donor, needy] < best_t:
                        best = donor; best_t = travel_time_matrix[donor, needy]
                if best != -1:
                    raw_need = CRITICAL_THRESHOLD[k] - (
                        sum(q for _, q in wbb_queues[needy]) if k == "WBB" else
                        (sum(q for _, q in plt_queues[needy]) if k == "PLT" else B[k][needy])
                    )
                    donor_above_safe = (
                        (sum(q for _, q in wbb_queues[best]) if k == "WBB" else
                         (sum(q for _, q in plt_queues[best]) if k == "PLT" else B[k][best]))
                        - SAFE_THRESHOLD[k]
                    )
                    qty_to_send = max(0.0, min(raw_need, donor_above_safe))
                    if qty_to_send <= 0: continue
                    delay = get_flight_delay(best, needy)
                    redistribution_events.append((best, needy, k, qty_to_send, t + delay))
        last_redistribution_check = t

    # --- casualties and demand ---
    casualties = generate_casualties(N, t, rate_per_hour=casualty_rate)
    patients = (casualties["CAS"] / 2.325).astype(int)
    demand_packets = patients * PACKETS_PER_PATIENT

    # --- Opportunistic FWB collection (Full Effort, logged separately) ---
    global FWB_from_WIA, FWB_from_DD
    for i in range(N):
        cas_i = casualties["CAS"][i]
        if cas_i <= 0:
            continue

        kia_i = cas_i * PCT_KIA
        wia_i = cas_i - kia_i
        cas_need_blood_i = int(round(cas_i / 2.325))
        wia_stable_i = max(0, wia_i - cas_need_blood_i)

        dd_donors  = PCT_KIA_SUITABLE * kia_i
        wia_donors = PCT_WIA_STABLE   * wia_stable_i

        dd_units  = dd_donors  * UNITS_PER_DD
        wia_units = wia_donors * UNITS_PER_WIA

        if dd_units > 0:
            wbb_queues[i].append([0.0, dd_units])
            B["WBB"][i] += dd_units
            FWB_from_DD[i] += dd_units

        if wia_units > 0:
            wbb_queues[i].append([0.0, wia_units])
            B["WBB"][i] += wia_units
            FWB_from_WIA[i] += wia_units



    # --- age & purge expiring products; sync B from queues ---
    for i in range(N):
        wbb_queues[i] = [[age+dt, qty] for age, qty in wbb_queues[i] if age+dt <= EXPIRY_WBB]
        plt_queues[i] = [[age+dt, qty] for age, qty in plt_queues[i] if age+dt <= EXPIRY_PLT]
        B["WBB"][i] = sum(q for _, q in wbb_queues[i])
        B["PLT"][i] = sum(q for _, q in plt_queues[i])
        B["CAS"][i] += casualties["CAS"][i]

    # --- apply arrivals ---
    for event in list(redistribution_events):
        frm, to, k, qty, t_arr = event
        if due_this_step(t, t_prev, t_arr):
            if k == "PLT":
                plt_queues[to].append([0.0, qty])
                plt_queues[frm], _ = withdraw_from_queue(plt_queues[frm], qty)
            elif k == "WBB":
                wbb_queues[to].append([0.0, qty])
                wbb_queues[frm], _ = withdraw_from_queue(wbb_queues[frm], qty)
            else:
                B[k][to] += qty
                B[k][frm] = max(0.0, B[k][frm] - qty)
            B["WBB"][frm] = sum(q for _, q in wbb_queues[frm])
            B["PLT"][frm] = sum(q for _, q in plt_queues[frm])
            redistribution_events.remove(event)

    # --- fulfill demand (packet-by-packet) ---
    for i in range(N):
        allow_wbb_donation = demand_packets[i] > 0 and (B["RBC"][i] + B["WBB"][i]) >= RBC_PER_PACKET
        unmet_any = False

        for _ in range(int(demand_packets[i])):  # loop over packets
            rbc_ok = B["RBC"][i] >= RBC_PER_PACKET
            ffp_ok = B["FFP"][i] >= FFP_PER_PACKET
            plt_ok = B["PLT"][i] >= PLT_PER_PACKET
            wbb_ok = B["WBB"][i] >= WB_UNITS_PER_PACKET

            # Prefer WBB (consume 4 WB units = 1 packet)
            if wbb_ok:
                wbb_queues[i], used = withdraw_from_queue(wbb_queues[i], WB_UNITS_PER_PACKET)
                if used >= WB_UNITS_PER_PACKET:
                    B["WBB"][i] -= used
                    continue

            # Full component packet
            if rbc_ok and ffp_ok and plt_ok:
                plt_queues[i], used = withdraw_from_queue(plt_queues[i], PLT_PER_PACKET)
                if used >= PLT_PER_PACKET:
                    B["RBC"][i] -= RBC_PER_PACKET
                    B["FFP"][i] -= FFP_PER_PACKET
                    B["PLT"][i] -= used
                    continue

            # Degraded combos
            if rbc_ok and ffp_ok:
                B["RBC"][i] -= RBC_PER_PACKET
                B["FFP"][i] -= FFP_PER_PACKET
                continue

            if rbc_ok and plt_ok:
                plt_queues[i], used = withdraw_from_queue(plt_queues[i], PLT_PER_PACKET)
                if used >= PLT_PER_PACKET:
                    B["RBC"][i] -= RBC_PER_PACKET
                    B["PLT"][i] -= used
                    continue

            # Minimal fallback
            if B["RBC"][i] >= (RBC_PER_PACKET // 2):
                B["RBC"][i] -= (RBC_PER_PACKET // 2)
                continue

            if rbc_ok and B["CRYO"][i] >= 1:
                B["RBC"][i] -= RBC_PER_PACKET
                B["CRYO"][i] -= 1
                continue

            unmet_any = True

        if unmet_any:
            unmet_demand_log[i, int(t / dt)] += 1

        # --- collect WBB donations if gate allows ---
        if allow_wbb_donation and t > t_setup[i] and NR[i] > 0 and B["WBB"][i] < B_max:
            max_draw = min(donor_rate * dt, NR[i], B_max - B["WBB"][i])
            if max_draw > 0:
                wbb_queues[i].append([0.0, max_draw])
                NR[i] -= max_draw
                NU[i] += max_draw
                cumulative_wbb_generated[i] += max_draw
                nu_queues[i].append([0.0, max_draw])  # cooldown starts

    # --- timed resupply arrivals ---
    for i in range(N):
        for arrival_time in resupply_schedule[i]:
            if due_this_step(t, t_prev, arrival_time):
                delivery = np.random.normal(loc=120, scale=10)
                delivery = max(0, delivery)
                baseline = {"RBC": INIT_RBC, "FFP": INIT_FFP, "PLT": INIT_PLT, "CRYO": INIT_CRYO}
                live_plt = sum(q for _, q in plt_queues[i])
                depletion = {
                    k: max(0.0, 1.0 - ((B[k][i] + (live_plt if k == "PLT" else 0.0)) / baseline[k]))
                    for k in ["RBC", "FFP", "PLT", "CRYO"]
                }
                total = sum(depletion.values()) + 1e-6
                for k in ["RBC", "FFP", "PLT", "CRYO"]:
                    share = depletion[k] / total
                    units = delivery * share
                    current_stock = (B["RBC"][i] + B["FFP"][i] + B["CRYO"][i]
                                     + sum(q for _, q in wbb_queues[i])
                                     + sum(q for _, q in plt_queues[i]))
                    available_space = max(0.0, MAX_STORAGE - current_stock)
                    units_to_add = min(units, available_space)
                    if units_to_add <= 0: continue
                    if k == "PLT":
                        plt_queues[i].append([0.0, units_to_add])
                    else:
                        B[k][i] += units_to_add
                break

    # --- donor recovery from cooldown ---
    for i in range(N):
        recovered = 0.0
        new_q = []
        for age, qty in nu_queues[i]:
            age_new = age + dt
            if age_new >= tau:
                recovered += qty
            else:
                new_q.append([age_new, qty])
        if recovered > 0:
            NR[i] += recovered
            NU[i] = max(0.0, NU[i] - recovered)
        nu_queues[i] = new_q

    # --- logs, low-supply, overwhelmed ---
    t_idx = int(t / dt)
    for i in range(N):
        cur_wbb = sum(q for _, q in wbb_queues[i])
        cur_plt = sum(q for _, q in plt_queues[i])

        if collect_live_metrics:
            live_WBB[i, t_idx] = cur_wbb
            live_PLT[i, t_idx] = cur_plt
            live_CAS[i, t_idx] = B["CAS"][i]

        for k in INIT_STOCK:
            if k == "CAS":
                continue
            if k == "WBB":
                cur = cur_wbb
            elif k == "PLT":
                cur = cur_plt
            else:
                cur = B[k][i]
            if cur < THRESHOLD_FRACTION * INIT_STOCK[k]:
                low_supply_events[k][i].append(t)

        if (t > t_setup[i] and NR[i] > 0 and cur_wbb < 1.0):
            wbb_overwhelmed_events[i].append(t)

    # --- clamp negatives (defensive) ---
    for k in B:
        B[k] = np.maximum(B[k], 0.0)

    return casualties, cumulative_wbb_generated


# ✅ FIX: Moved this section up to be defined before the global loop
B_init = {
    "RBC": np.full(N, INIT_RBC, dtype=float),
    "FFP": np.full(N, INIT_FFP, dtype=float),
    "PLT": np.full(N, INIT_PLT, dtype=float),
    "CRYO": np.full(N, INIT_CRYO, dtype=float),
    "WBB": np.full(N, INIT_WBB, dtype=float),
    "CAS": np.zeros(N, dtype=float),
}
NR_init = np.full(N, N_total, dtype=float)
NU_init = np.zeros(N, dtype=float)

# Initial state
B_state = {k: B_init[k].copy() for k in B_init}
NR_state = NR_init.copy()
NU_state = NU_init.copy()

# ✅ FIX: Initialize nu_queues as a list of lists, similar to wbb_queues and plt_queues
nu_queues = [[] for _ in range(N)]

live_WBB = np.zeros((N, len(time)))
live_PLT = np.zeros((N, len(time)))
live_CAS = np.zeros((N, len(time)))

results = {k: np.zeros((N, len(time))) for k in ["RBC", "FFP", "PLT", "CRYO", "WBB", "CAS"]}
results["cumulative_wbb_generated"] = np.zeros((N, len(time)))

cumulative_wbb_generated = np.zeros(N)
FWB_from_DD = np.zeros(N)
FWB_from_WIA = np.zeros(N)
for idx, t in enumerate(time):
    casualties, cumulative_wbb_generated = step(t, B_state, NR_state, NU_state, cumulative_wbb_generated)
    for k in ["RBC", "FFP", "PLT", "CRYO", "CAS"]:
        results[k][:, idx] = B_state[k]
    results["WBB"][:, idx] = [sum(qty for age, qty in wbb_queues[i]) for i in range(N)]
    results["cumulative_wbb_generated"][:, idx] = cumulative_wbb_generated.copy()


# Plotting
# ...

def run_sim_with_interval(interval_hours, casualty_rate=7.5, wbb_rate=None, collect_timeseries=True):
    global B_state, NR_state, NU_state, wbb_queues, plt_queues, redistribution_events, last_redistribution_check
    global unmet_demand_log, live_WBB, live_PLT, live_CAS, resupply_schedule, blackout_windows, nu_queues, collect_live_metrics
    global FWB_from_WIA, FWB_from_DD
    FWB_from_WIA = np.zeros(N)
    FWB_from_DD  = np.zeros(N)

    collect_live_metrics = collect_timeseries

    # Reset everything
    B_state = {k: B_init[k].copy() for k in B_init}
    NR_state = NR_init.copy()
    NU_state = NU_init.copy()
    wbb_queues = [[] for _ in range(N)]
    plt_queues = [[] for _ in range(N)]
    nu_queues = [[] for _ in range(N)]
    cumulative_wbb_generated = np.zeros(N)
    # re-init live telemetry for this run
    if collect_timeseries:
        live_WBB = np.zeros((N, len(time)))
        live_PLT = np.zeros((N, len(time)))
        live_CAS = np.zeros((N, len(time)))
    else:
        live_WBB = live_PLT = live_CAS = None


    for i in range(N):
        if INIT_WBB > 0:
            wbb_queues[i].append([0.0, INIT_WBB])
        if INIT_PLT > 0:
            plt_queues[i].append([0.0, INIT_PLT])

    redistribution_events = []
    last_redistribution_check = -12.0

    unmet_demand_log.fill(0)

    # Initialize per-product time-series results for this run
    results_ts = None
    if collect_timeseries:
        results_ts = {k: np.zeros((N, len(time))) for k in ["RBC", "FFP", "PLT", "CRYO", "WBB", "CAS"]}
        results_ts["cumulative_wbb_generated"] = np.zeros((N, len(time)))

    blackout_windows = []
    t = 0
    while t < T_max:
        if np.random.rand() < 0.15:
            start = t
            duration = np.random.uniform(24, 48)
            blackout_windows.append((start, start + duration))
            t += duration
        else:
            t += 6

    resupply_schedule = [[] for _ in range(N)]
    for i in range(N):
        t = 0
        while t < T_max:
            delay = np.random.uniform(2, 4)
            arrival = t + delay
            if not in_blackout(arrival, blackout_windows):
                resupply_schedule[i].append(arrival)
            t += interval_hours

    for idx, t in enumerate(time):
        casualties, cumulative_wbb_generated = step(
            t,
            B_state,
            NR_state,
            NU_state,
            cumulative_wbb_generated,
            casualty_rate=casualty_rate,
            wbb_rate=wbb_rate,
        )

        if collect_timeseries:
            for k in ["RBC", "FFP", "PLT", "CRYO", "CAS"]:
                results_ts[k][:, idx] = B_state[k]
            results_ts["WBB"][:, idx] = [sum(qty for age, qty in wbb_queues[i]) for i in range(N)]
            results_ts["cumulative_wbb_generated"][:, idx] = cumulative_wbb_generated.copy()

    total_unmet = np.sum(unmet_demand_log)
    first_failure_time = None
    if total_unmet > 0:
        failure_indices = np.where(unmet_demand_log.sum(axis=0) > 0)[0]
        if len(failure_indices) > 0:
            first_failure_time = failure_indices[0] * dt

    total_FWB_from_DD  = np.sum(FWB_from_DD)
    total_FWB_from_WIA = np.sum(FWB_from_WIA)

    summary = {
        "interval": interval_hours,
        "total_unmet": total_unmet,
        "first_failure_time": first_failure_time,
        "final_RBC": np.mean(B_state["RBC"]),
        "final_WBB": np.mean(B_state["WBB"]),
        "casualties_total": np.sum(B_state["CAS"]),
        "unmet_by_node": [int(np.sum(unmet_demand_log[i])) for i in range(N)],
        "unmet_over_time": unmet_demand_log.copy() if collect_timeseries else None,
        "FWB_from_DD": total_FWB_from_DD,
        "FWB_from_WIA": total_FWB_from_WIA,
    }

    if collect_timeseries and results_ts is not None:
        summary.update({
            "RBC": results_ts["RBC"],
            "FFP": results_ts["FFP"],
            "PLT": results_ts["PLT"],
            "CRYO": results_ts["CRYO"],
            "WBB": results_ts["WBB"],
            "CAS": results_ts["CAS"],
            "cumulative_wbb_generated": results_ts["cumulative_wbb_generated"],
        })

    return summary


def plot_full_blood_panel(sim_result, interval, wbb_rate_avg=10):
    fig, axs = plt.subplots(5, 1, figsize=(12, 18), sharex=True)

    fob_labels = [f"FOB {i}" for i in range(N)]
    results = sim_result

    # Convert hours to days for plotting
    time_days = time / 24

    # Surge window in days instead of hours
    surge_start_days = 120 / 24
    surge_end_days = 240 / 24

    for i in range(N):
        axs[0].plot(time_days, results["RBC"][i], label=fob_labels[i])
    axs[0].set_ylabel("RBC Units")
    axs[0].axvspan(surge_start_days, surge_end_days, color="red", alpha=0.15, label="Surge")
    axs[0].legend()

    for i in range(N):
        axs[1].plot(time_days, results["FFP"][i], label=fob_labels[i])
    axs[1].set_ylabel("FFP Units")
    axs[1].axvspan(surge_start_days, surge_end_days, color="red", alpha=0.15, label="Surge")

    for i in range(N):
        axs[2].plot(time_days, results["PLT"][i], label=fob_labels[i])
    axs[2].set_ylabel("PLT Units")
    axs[2].axvspan(surge_start_days, surge_end_days, color="red", alpha=0.15, label="Surge")

    # for i in range(N):
    #     axs[3].plot(time, results["CRYO"][i], label=fob_labels[i])
    # axs[3].set_ylabel("CRYO Units")
    # axs[3].axvspan(120, 240, color="red", alpha=0.15, label="Surge")

    ax1 = axs[3]
    ax1.set_ylabel("FWB Units")
    for i in range(N):
        ax1.plot(time_days, results["WBB"][i], label=f'FWB - FOB {i}', linestyle='-')
    ax1.axvspan(surge_start_days, surge_end_days, color="red", alpha=0.15, label="Surge")
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Cumulative WBB Units Generated")
    for i in range(N):
        ax2.plot(time_days, results["cumulative_wbb_generated"][i], label=f'Cumulative via WBB - FOB {i}', linestyle='--')
    ax2.legend(loc='lower right')

    for i in range(N):
        axs[4].plot(time_days, results["CAS"][i], label=fob_labels[i])
    axs[4].set_ylabel("CAS Units")
    axs[4].axvspan(surge_start_days, surge_end_days, color="red", alpha=0.15, label="Surge")
    axs[4].legend()

    axs[-1].set_xlabel("Time (Days)")
    plt.suptitle(f" Blood Product Trends — Resupply Interval = {interval} hrs, Donor rate = {wbb_rate_avg} per hr", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def run_stress_test(intervals=[24,48,72,96,120], num_runs=50):
    """Monte Carlo over resupply intervals. Returns per-interval samples."""
    samples_unmet = {iv: [] for iv in intervals}
    samples_first_fail = {iv: [] for iv in intervals}

    print("\n🧾 MONTE CARLO STRESS TEST SUMMARY (Resupply Intervals):")
    print(f"{'Interval (hrs)':>15} | {'Avg. Unmet Total':>18} | {'Avg. 1st Failure (hr)':>23}")
    print("-" * 65)

    for iv in intervals:
        args = [iv] * num_runs

        worker = partial(run_sim_with_interval, collect_timeseries=False)
        with Pool(cpu_count()) as pool:
            outs = pool.map(worker, args)

        for out in outs:
            samples_unmet[iv].append(out['total_unmet'])



            # keep only finite failure times for the boxplot
            if out['first_failure_time'] is not None and np.isfinite(out['first_failure_time']):
                samples_first_fail[iv].append(out['first_failure_time'])

        avg_unmet = float(np.mean(samples_unmet[iv])) if samples_unmet[iv] else 0.0
        avg_ff = float(np.mean(samples_first_fail[iv])) if samples_first_fail[iv] else None
        print(f"{iv:>15} | {avg_unmet:>18.2f} | {avg_ff or 'None':>23}")

    return {"unmet": samples_unmet, "first_fail": samples_first_fail}



def run_donor_stress_test(rates=[10,8,6,4], num_runs=50):
    """Monte Carlo over WBB generation rates. Returns per-rate samples."""
    global WBB_RATE
    original = WBB_RATE

    samples_unmet = {r: [] for r in rates}
    samples_first_fail = {r: [] for r in rates}

    print("\n🧾 MONTE CARLO STRESS TEST SUMMARY (WBB Generation Rates):")
    print(f"{'WBB Rate (u/h)':>15} | {'Avg. Unmet Total':>18} | {'Avg. 1st Failure (hr)':>23}")
    print("-" * 65)

    for r in rates:
        WBB_RATE = r
        args = [60]*num_runs
        worker = partial(run_sim_with_interval, collect_timeseries=False)
        with Pool(cpu_count()) as pool:
            outs = pool.map(worker, args)

        for out in outs:
            samples_unmet[r].append(out['total_unmet'])
            if out['first_failure_time'] is not None and np.isfinite(out['first_failure_time']):
                samples_first_fail[r].append(out['first_failure_time'])

        avg_unmet = float(np.mean(samples_unmet[r])) if samples_unmet[r] else 0.0
        avg_ff = float(np.mean(samples_first_fail[r])) if samples_first_fail[r] else None
        print(f"{r:>15} | {avg_unmet:>18.2f} | {avg_ff or 'None':>23}")

        detailed_run = run_sim_with_interval(60, wbb_rate=r, collect_timeseries=True)
        plot_full_blood_panel(detailed_run, 60, r)

    WBB_RATE = original
    return {"unmet": samples_unmet, "first_fail": samples_first_fail}

def run_casualty_stress_test(rates=[2,3,4,5,6,7], num_runs=50):
    """Monte Carlo over casualty arrival rates. Returns per-rate samples."""
    samples_unmet = {r: [] for r in rates}
    samples_first_fail = {r: [] for r in rates}

    print("\n🧾 MONTE CARLO STRESS TEST SUMMARY (Casualty Rates):")
    print(f"{'Casualties/hr':>15} | {'Avg. Unmet Total':>18} | {'Avg. 1st Failure (hr)':>23}")
    print("-" * 65)

    for r in rates:
        args = [(24, r)] * num_runs
        worker = partial(run_sim_with_interval, collect_timeseries=False)

        with Pool(cpu_count()) as pool:
            outs = pool.starmap(worker, args)
        for out in outs:
            samples_unmet[r].append(out['total_unmet'])
            if out['first_failure_time'] is not None and np.isfinite(out['first_failure_time']):
                samples_first_fail[r].append(out['first_failure_time'])

        avg_unmet = float(np.mean(samples_unmet[r])) if samples_unmet[r] else 0.0
        avg_ff = float(np.mean(samples_first_fail[r])) if samples_first_fail[r] else None
        print(f"{r:>15} | {avg_unmet:>18.2f} | {avg_ff or 'None':>23}")

    return {"unmet": samples_unmet, "first_fail": samples_first_fail}


def boxplot_dict(metric_dict, title, ylabel, x_label):
    """metric_dict: {category: [samples]}"""
    cats = list(metric_dict.keys())
    data = [metric_dict[c] if len(metric_dict[c]) > 0 else [np.nan] for c in cats]

    fig, ax = plt.subplots(figsize=(12, 8))
    bp = ax.boxplot(data)  # don’t pass labels here to dodge the deprecation

    # set tick labels explicitly (works across versions)
    ax.set_xticks(range(1, len(cats) + 1))
    ax.set_xticklabels([str(c) for c in cats])

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def _run_cell(args):
    """Helper to run one (WBB, casualty) combo with Monte Carlo repeats."""
    resupply, wbb, cas, num_runs, metric = args
    #global WBB_RATE
    #WBB_RATE = wbb  # override global donor rate
    outs = [run_sim_with_interval(resupply, casualty_rate=cas, wbb_rate=wbb, collect_timeseries=False)
            for _ in range(num_runs)]
    return np.mean([o[metric] for o in outs])

def heatmap_casualty_wbb_parallel(resupply_interval=48, casualty_rates=[2,3,4,5,6,7],
                                  wbb_rates=[10,8,6,4], num_runs=1, metric="total_unmet"):
    """
    Parallelized heatmap of outcome vs casualty rate and WBB generation rate.
    """
    jobs = [(resupply_interval, wbb, cas, num_runs, metric)
            for wbb in wbb_rates for cas in casualty_rates]

    # run in parallel
    with Pool(cpu_count()) as pool:
        results_flat = pool.map(_run_cell, jobs)

    # reshape into [len(wbb_rates), len(casualty_rates)]
    results = np.array(results_flat).reshape(len(wbb_rates), len(casualty_rates))

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(results, origin="lower", aspect="auto",
                   extent=[min(casualty_rates), max(casualty_rates),
                           min(wbb_rates), max(wbb_rates)],
                   cmap="viridis")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.replace("_", " ").title())

    ax.set_xlabel("Casualties per Hour")
    ax.set_ylabel("WBB Generation Rate (u/hr)")
    ax.set_title(f"{metric.replace('_',' ').title()} — Resupply = {resupply_interval} hrs")

    plt.show()
    return results

def heatmap_wbb_vs_resupply(casualty_rate, wbb_rates, resupply_intervals, num_runs=10, metric="total_unmet"):
    results = np.zeros((len(wbb_rates), len(resupply_intervals)))

    for i, wbb in enumerate(wbb_rates):
        for j, resupply in enumerate(resupply_intervals):
            outs = [run_sim_with_interval(resupply, casualty_rate=casualty_rate, wbb_rate=wbb,
                                          collect_timeseries=False)
                    for _ in range(num_runs)]
            results[i, j] = np.mean([o[metric] for o in outs])

    return results

def plot_heatmap(results, wbb_rates, resupply_intervals, casualty_rate, metric="total_unmet"):
    plt.figure(figsize=(8,6))
    plt.imshow(results, origin="lower", aspect="auto",
               extent=[min(resupply_intervals), max(resupply_intervals),
                       min(wbb_rates), max(wbb_rates)],
               cmap="viridis")
    plt.colorbar(label=metric.replace("_"," ").title())
    plt.xlabel("Resupply Interval (hrs)")
    plt.ylabel("WBB Generation Rate (u/hr)")
    plt.title(f"{metric.replace('_',' ').title()} — Casualty Rate = {casualty_rate}/hr")
    plt.show()

def collect_data(resupply_intervals, wbb_rates, casualty_rates,kia_suitable_levels=[0.05, 0.10, 0.15, 0.20, 0.25],
                 wia_stable_levels=[0.05, 0.10, 0.15, 0.20, 0.25], num_runs=10):
    rows = []
    for kia_suit in kia_suitable_levels:
        for wia_stable in wia_stable_levels:
            PCT_KIA_SUITABLE = kia_suit
            PCT_WIA_STABLE   = wia_stable

            for cas in casualty_rates:
                for res in resupply_intervals:
                    for wbb in wbb_rates:
                        for _ in range(num_runs):
                            out = run_sim_with_interval(interval_hours=res,
                                                        casualty_rate=cas,
                                                        wbb_rate=wbb,
                                                        collect_timeseries=False)
                            rows.append({
                                "casualties_per_day": cas * 24,
                                "casualties_per_hr": cas,
                                "resupply_interval": res,
                                "wbb_rate": wbb,
                                "pct_kia_suitable": kia_suit,
                                "pct_wia_stable": wia_stable,
                                "unmet_total": out["total_unmet"],
                                "first_failure_time": out["first_failure_time"] if out["first_failure_time"] else T_max,
                                "FWB_from_DD": out["FWB_from_DD"],
                                "FWB_from_WIA": out["FWB_from_WIA"]
                            })
    return pd.DataFrame(rows)

def run_anova(df, metric="unmet_total"):
    model = ols(f'{metric} ~ C(casualties_per_day) * C(resupply_interval) * C(wbb_rate) * C(pct_kia_suitable) * C(pct_wia_stable)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

def plot_interaction(df, outcome="unmet_total"):
    """
    Makes an interaction plot of casualties/day × resupply interval on outcome.
    """
    plt.figure(figsize=(10,6))
    sns.pointplot(
        data=df,
        x="resupply_interval",
        y=outcome,
        hue="casualties_per_day",
        ci="sd",  # show variability
        dodge=True,
        markers="o",
        linestyles="-"
    )
    plt.title(f"Interaction Plot: Resupply Interval × Casualties/Day on {outcome}")
    plt.ylabel(outcome.replace("_"," ").title())
    plt.xlabel("Resupply Interval (hrs)")
    plt.legend(title="Casualties per Day")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def plot_anova_effects(anova_table, title="Variance Explained by Factor"):
    """
    Bar chart of % variance explained per ANOVA factor, color-coded:
      🔵 Casualty-driven = blue
      🟠 Resupply-driven = orange
      🟢 WBB-driven = green
      Mixed factors = base color + diagonal stripe hatch
    """
    # drop residual row and compute % variance explained
    effects = anova_table.drop("Residual")
    total_ss = effects["sum_sq"].sum() + anova_table.loc["Residual","sum_sq"]
    effects["perc_var"] = 100 * effects["sum_sq"] / total_ss
    effects = effects.sort_values("perc_var", ascending=False)

    # prettier labels
    clean_labels = {
        "C(casualties_per_day)": "Casualties per Day",
        "C(resupply_interval)": "Resupply Interval",
        "C(wbb_rate)": "WBB Generation Rate",
        "C(casualties_per_day):C(resupply_interval)": "Casualties × Resupply",
        "C(casualties_per_day):C(wbb_rate)": "Casualties × WBB",
        "C(resupply_interval):C(wbb_rate)": "Resupply × WBB",
        "C(casualties_per_day):C(resupply_interval):C(wbb_rate)": "3-Way Interaction"
    }

    # color + hatch encoding
    color_map = {
        "C(casualties_per_day)": ("royalblue", None),
        "C(resupply_interval)": ("darkorange", None),
        "C(wbb_rate)": ("seagreen", None),
        "C(casualties_per_day):C(resupply_interval)": ("royalblue", "//"),        # blue + orange stripes
        "C(casualties_per_day):C(wbb_rate)": ("royalblue", "xx"),                 # blue + green stripes
        "C(resupply_interval):C(wbb_rate)": ("darkorange", "xx"),                 # orange + green stripes
        "C(casualties_per_day):C(resupply_interval):C(wbb_rate)": ("royalblue", "///xxx")  # all three
    }

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10,6))
    for i, (term, row) in enumerate(effects.iterrows()):
        color, hatch = color_map.get(term, ("gray", None))
        ax.bar(i, row["perc_var"], color=color, hatch=hatch,
               edgecolor="black", linewidth=0.8)

    ax.set_xticks(range(len(effects)))
    ax.set_xticklabels([clean_labels.get(t, t) for t in effects.index],
                       rotation=30, ha="right")
    ax.set_ylabel("% of Variance Explained")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # legend (manual so it matches the hatch system)
    legend_elements = [
        Patch(facecolor="royalblue", edgecolor="black", label="Casualties"),
        Patch(facecolor="darkorange", edgecolor="black", label="Resupply"),
        Patch(facecolor="seagreen", edgecolor="black", label="WBB"),
        Patch(facecolor="royalblue", hatch="//", edgecolor="black", label="Casualties × Resupply"),
        Patch(facecolor="royalblue", hatch="xx", edgecolor="black", label="Casualties × WBB"),
        Patch(facecolor="darkorange", hatch="xx", edgecolor="black", label="Resupply × WBB"),
        Patch(facecolor="royalblue", hatch="///xxx", edgecolor="black", label="3-Way Interaction")
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1),
              loc="upper left", frameon=True, title="Factor Type")

    plt.tight_layout(rect=[0,0,0.85,1])
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_heatmap_grid(df, metric="unmet_total", cmap="mako"):
    """
    Creates a 3×2 grid of heatmaps:
    X-axis = Resupply Interval (hrs)
    Y-axis = Casualties per Day
    Each subplot = one WBB generation rate.
    """
    wbb_rates = sorted(df['wbb_rate'].unique())
    resupply_levels = sorted(df['resupply_interval'].unique())
    casualties_levels = sorted(df['casualties_per_day'].unique())

    fig, axs = plt.subplots(3, 2, figsize=(12, 14), sharex=True, sharey=True)
    axs = axs.flatten()

    # global color range (so color scale consistent across subplots)
    vmin = df[metric].min()
    vmax = df[metric].max()

    for ax, wbb in zip(axs, wbb_rates):
        sub = df[df["wbb_rate"] == wbb]
        pivot = sub.pivot_table(index="casualties_per_day",
                                columns="resupply_interval",
                                values=metric,
                                aggfunc="mean")
        sns.heatmap(pivot, cmap=cmap, vmin=vmin, vmax=vmax,
                    annot=True, fmt=".0f", annot_kws={"size":7},
                    cbar=False, ax=ax, linewidths=0.3, linecolor='gray')

        ax.set_title(f"WBB Rate = {wbb} units/hr", fontsize=11)
        ax.set_xlabel("Resupply Interval (hrs)")
        ax.set_ylabel("Casualties per Day")

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label=metric.replace("_", " ").title())

    fig.suptitle(
        f"Interaction Between Resupply Interval × Casualties/Day by WBB Rate\nMetric: {metric.replace('_',' ').title()}",
        fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()


# Main execution block
if __name__ == '__main__':
    # choose how many runs you want for the distributions
    NUM_RUNS = 25

    print("Travel Time Matrix (hrs):")
    print(np.round(travel_time_matrix, 2))

    #resupply_intervals = [24, 48, 72, 96, 120]
    #wbb_rates = [10, 8, 6, 4]

    #casualty_rates = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7]


    # Monte Carlo: collect samples
    #resupply_samples = run_stress_test(resupply_intervals, num_runs=NUM_RUNS)
    #donor_samples = run_donor_stress_test(wbb_rates, num_runs=NUM_RUNS)

    #casualty_samples = run_casualty_stress_test(casualty_rates, num_runs=25)

    # Boxplots — Unmet totals
    # boxplot_dict(
    #     resupply_samples["unmet"],
    #     title="Resupply Stress Test — Total Unmet Transfusion Patients",
    #     ylabel="Unmet Transfusion Patients (total over horizon)",
    #     x_label="Resupply Interval (hrs)"
    # )
    #
    # boxplot_dict(
    #     donor_samples["unmet"],
    #     title="Donor Rate Stress Test — Total Unmet Transfusion Patients",
    #     ylabel="Unmet Transfusion Patients (total over horizon)",
    #     x_label="WBB Generation Rate (units/hour)"
    # )
    #
    # # Boxplots — First failure times (only for runs that failed)
    # if any(len(v) > 0 for v in resupply_samples["first_fail"].values()):
    #     boxplot_dict(
    #         resupply_samples["first_fail"],
    #         title="Resupply Stress Test — Time to First Failure",
    #         ylabel="Hours",
    #         x_label="Resupply Interval (hrs)"
    #     )
    #
    # if any(len(v) > 0 for v in donor_samples["first_fail"].values()):
    #     boxplot_dict(
    #         donor_samples["first_fail"],
    #         title="Donor Rate Stress Test — Time to First Failure",
    #         ylabel="Hours",
    #         x_label="WBB Generation Rate (units/hour)"
    #     )
    #
    # boxplot_dict(
    #     casualty_samples["unmet"],
    #     title="Casualty Rate Stress Test — Total Unmet Transfusion Patients",
    #     ylabel="Unmet Transfusion Patients",
    #     x_label="Casualties per Hour"
    # )
    #
    # # Boxplot — First failure times (convert to days)
    # # Boxplot — First failure times (convert to days, treat no failures as horizon = 30 days)
    # sim_horizon_days = 30
    # casualty_first_fail_days = {
    #     k * 24: ([val / 24 for val in v] if len(v) > 0 else [sim_horizon_days])
    #     for k, v in casualty_samples["first_fail"].items()
    # }
    #
    # boxplot_dict(
    #     casualty_first_fail_days,
    #     title="Casualty Rate Stress Test — Time to First Failure",
    #     ylabel="Time to First Failure (days)",
    #     x_label="Casualties per Day"
    # )

    # Define ranges
    # resupply_intervals = [24, 48, 72, 96, 120]  # hours
    # casualty_rates = [1, 2, 3, 4, 5, 6, 7]  # casualties/hr
    # wbb_rates = [2,4,6,8,10,12]  # WBB units/hr
    #
    # for cas in casualty_rates:
    #     results = heatmap_wbb_vs_resupply(cas, wbb_rates, resupply_intervals, num_runs=5, metric="total_unmet")
    #     plot_heatmap(results, wbb_rates, resupply_intervals, cas, metric="total_unmet")

    resupply_intervals = [24, 48,96,150,200]
    wbb_rates = [2,4,6,8,10,12]
    casualty_rates = [1.67,2.08,2.50,2.92,3.33,3.75,4.17]  # per hr per node
    num_runs = 50  # start small to test

    df = collect_data(resupply_intervals, wbb_rates, casualty_rates, num_runs=num_runs)

    df.to_csv("ODE_sim_results.csv", index=False)
    print(f"[Saved] {len(df)} rows to ODE_sim_results.csv")

    # Run ANOVA on unmet_total
    print(run_anova(df, metric="unmet_total"))

    # Run ANOVA on time to first failure
    print(run_anova(df, metric="first_failure_time"))

    plot_interaction(df, "unmet_total")

    plot_interaction(df, "first_failure_time")


    # Example usage after your run_anova():
    anova_unmet = run_anova(df, metric="unmet_total")
    plot_anova_effects(anova_unmet, "Unmet Total — Variance Explained by Factor")

    anova_fail = run_anova(df, metric="first_failure_time")
    plot_anova_effects(anova_fail, "First Failure Time — Variance Explained by Factor")

    plot_heatmap_grid(df, metric="unmet_total")
    plot_heatmap_grid(df, metric="first_failure_time")


    # Run resupply stress test with Monte Carlo
    #resupply_results = run_stress_test(resupply_intervals, num_runs=50)
    # Run WBB donation rate stress test with Monte Carlo
    #wbb_results = run_donor_stress_test(wbb_rates, num_runs=50)

    #For example:
    # plot_full_blood_panel(run_sim_with_interval(24), 24)
    # plot_full_blood_panel(run_sim_with_interval(48), 48)
    # plot_full_blood_panel(run_sim_with_interval(72), 72)
    # plot_full_blood_panel(run_sim_with_interval(96), 96)
    # plot_full_blood_panel(run_sim_with_interval(120), 120)



