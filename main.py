"""gta6-trophy-rarity-forecast-01 entry: earners/tracked decomposition with a
saturating progression curve, in Python 3.13 (standard library only).

The score is mean pinball loss on the LOGIT scale, so the forecast is built in
logit space and every rate is clamped to 0.01..99.99 before taking a logit.

Model
-----
Earn rate = earners / players_tracked.  The two move differently, and that is
the whole game:

* players_tracked jumps when Exophase refreshes (the practice cases are
  completely flat at 638 players, then the first refresh lands at the target
  and tracked jumps 6x while every rate but the 100% one drops).  Tracked is
  modelled as a log-time trend, fit to the history when it moves, else a prior
  slope (mN_prior = 5.66, the log-time slope of the observed launch-night
  refresh) times the log-time distance to the target, capped at 50x growth.
* earners grow as players progress.  Each trophy follows a saturating
  (logistic in LINEAR time since launch) progression curve u(t) =
  C / (1 + exp(-k (t - t50))), with ceiling C, half-time t50 and steepness k
  set from the trophy's tier and description: story chapters rise fast to a
  high ceiling (Ch.1: C 0.65, t50 92 h), grind/collectibles rise slowly to a
  low ceiling (C 0.05, t50 500 h), the platinum barely moves (C 0.004).
  Existing players' earners grow by the curve's ratio u(t_target)/u(t_now),
  capped at the cohort's own size; newcomers (the tracked growth) earn at the
  curve's value at a late-horizon-weighted effective playtime.  When the
  history itself shows earners moving, a log-time fit of ln(earners) is
  blended in (at most 70% weight) so real trends override the prior.
* the 100% trophy (Complete Gone Fishing) is already earned by everyone, so
  its curve is C 1.0, t50 0.25 h, k 8/h -- a near-step: newcomers earn it too,
  and it stays ~100%.  (The old t50 1 h / k 2/h implied ~0.2% of newcomers
  had not done the first mission by hour 4, contradicting the observed
  638/638 and the 100% rate surviving the 6x refresh; it cost 2.6 logit of
  pinball on p00 alone.)

Spread
------
Quantiles are logit(median) + z_tau * s, z = (-1.645, -0.674, 0, 0.674, 1.645),
s = sqrt(s0^2 + (sN*sqrt(days))^2 + (sE*sqrt(days)*rarity)^2), so the spread
widens with the horizon and with rarity.  Rounded to 4 decimals (half up,
as the starter does), enforced non-decreasing.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P
keeps this file's directory OFF sys.path, so the sys.path line below puts it
back first (there is no sibling module; this file is self-contained).

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at a
    time, with only that case's file there): JSON {"schema":
    "makefun/gta6-trophies-case/v1", "id", "at", "horizon_minutes", "target",
    "levels", "game", "trophies", "history", "notes"}, where `trophies` lists
    the case's trophies {id, name, description, tier, points, secret} and
    `history` every read before `at`, oldest first, each {slot, at,
    players_tracked, trophies_earned, rates}, `rates` in the order of
    `trophies` (null where that read lacked the trophy);
  - write <id>.out to the output directory: {"trophies": [{"id": 1, "q":
    [q05, q25, q50, q75, q95]}, ...]}, one row for each of the case's
    trophies, each id once, no other key, each q five finite numbers from 0
    to 100 (percent), non-decreasing; at most 65,536 bytes.  Anything else is
    an invalid case, scored as if every quantile equalled the trophy's rate
    in the last read, and an entry with an invalid case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- Model parameters (calibrated on the practice cases; see the docstring) ---
P = {
    "mN_prior": 5.66,     # log-time slope of tracked when history is flat
    "mN_cap": 3.0,        # cap on the fitted tracked slope
    "mE_cap": 3.0,        # cap on the fitted earners slope
    "N_growth_cap": 50.0, # cap on tracked growth over the horizon (x)
    "ratio_cap": 50.0,    # cap on the progression-curve earners ratio
    "sN": 0.6,            # spread: tracked uncertainty per sqrt(day)
    "sE": 0.5,            # spread: earners uncertainty per sqrt(day)
    "s0": 0.15,           # spread: base, in logit units
    "s_max": 2.5,         # spread: cap, in logit units
}

# Grind/collectible descriptions: slow, low ceiling.
_GRIND_PATTERNS = (
    "250 stunts", "50 challenges", "all garage properties", "100 locations",
    "100 vehicles", "3 collectible card sets", "every spot",
    "gold medal in all", "win 20 different", "complete 8 mission strands",
)


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(x: float) -> float:
    if x >= 0:
        return 100 / (1 + math.exp(-x))
    e = math.exp(x)
    return 100 * e / (1 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it.  A whole number comes back as
    an int, so json.dumps writes 100 where JavaScript writes 100."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def trophy_curve(trophy: dict):
    """(ceiling C, half-time t50 in hours, steepness k per hour) of the
    trophy's saturating progression curve u(t) = C / (1 + exp(-k (t - t50))).

    Story chapters rise fast to a high ceiling; grind/collectibles rise
    slowly to a low one; the platinum barely moves; the first mission is
    earned by essentially everyone, immediately.
    """
    desc = trophy["description"].lower()
    tier = trophy["tier"]
    if tier == "platinum":
        return (0.004, 1000.0, 0.008)
    if "100% completion" in desc:
        return (0.02, 800.0, 0.010)
    if any(p in desc for p in _GRIND_PATTERNS):
        return (0.05, 500.0, 0.015)
    if "complete gone fishing" in desc:
        # The first mission: 638/638 tracked players already had it on launch
        # night, and the rate stayed 100% after the 04:30Z refresh to 3,868
        # tracked (RULES.txt).  So it is earned by essentially everyone
        # within minutes of being tracked: a near-step curve (t50 0.25 h,
        # k 8/h) so newcomers earn it with probability ~1, instead of the
        # old (1.0, 1.0, 2.0), which implied ~0.2% of newcomers had not done
        # it by hour 4 and cost 2.6 logit of pinball on p00 alone.
        return (1.0, 0.25, 8.0)
    m = re.search(r"chapter (\d)", desc)
    if m:
        return {
            1: (0.65, 92.0, 0.074),
            2: (0.55, 120.0, 0.060),
            3: (0.45, 150.0, 0.050),
            4: (0.38, 180.0, 0.045),
        }.get(int(m.group(1)), (0.45, 150.0, 0.050))
    if "final mission" in desc and tier == "gold":
        return (0.28, 220.0, 0.040)
    if desc.startswith("complete ") and (
        "mission" in desc or "run" in desc or "heist" in desc
        or "play" in desc or "takedown" in desc or "operation" in desc
    ):
        return (0.30, 180.0, 0.040)
    return (0.20, 150.0, 0.045)  # other activities


def u_curve(C: float, t50: float, k: float, t_hours: float) -> float:
    """The progression curve at t hours since launch: the fraction of
    players who have earned the trophy by then."""
    t = max(t_hours, 0.0)
    z = -k * (t - t50)
    z = max(-60.0, min(60.0, z))
    return C / (1.0 + math.exp(z))


def _fit_loglog(xs, ys):
    """Ordinary least squares of ys on xs; returns (intercept, slope)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0
    m = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - m * mx, m


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    hist = case["history"]
    launch = parse_iso(case["game"]["release_date"] + "T00:00:00Z")
    at = parse_iso(case["at"])
    tgt = parse_iso(case["target"])
    t_c = (at - launch).total_seconds() / 3600.0
    t_t = (tgt - launch).total_seconds() / 3600.0
    dlogt = math.log1p(max(t_t, 0.0)) - math.log1p(max(t_c, 0.0))
    horizon_hours = case["horizon_minutes"] / 60.0
    days = case["horizon_minutes"] / 1440.0

    if not hist:
        # No reads at all: fall back to the progression curve alone.
        rows = []
        for trophy in trophies:
            C, t50, k = trophy_curve(trophy)
            r_med = clamp(u_curve(C, t50, k, max(t_t, 0.0)) * 100.0)
            lg = logit(r_med)
            s = min(P["s_max"], math.sqrt(P["s0"] ** 2 + (P["sN"] * math.sqrt(days)) ** 2))
            qs = [round4(inv_logit(lg + z * s)) for z in Z]
            for kk in range(1, 5):
                if qs[kk] < qs[kk - 1]:
                    qs[kk] = qs[kk - 1]
            rows.append({"id": trophy["id"], "q": qs})
        return {"trophies": rows}

    last = hist[-1]
    N_last = last["players_tracked"]

    # --- Tracked players at the target ---
    xs = []
    ys = []
    for h in hist:
        t = (parse_iso(h["at"]) - launch).total_seconds() / 3600.0
        xs.append(math.log1p(max(t, 0.0)))
        ys.append(math.log(max(h["players_tracked"], 1)))
    N_var = (max(ys) - min(ys)) > 1e-9
    if N_var and len(xs) >= 3:
        _, mN_fit = _fit_loglog(xs, ys)
        mN_fit = max(0.0, min(mN_fit, P["mN_cap"]))
        w = min(0.7, len(xs) / 24.0)
        mN_eff = w * mN_fit + (1 - w) * P["mN_prior"]
    else:
        mN_eff = P["mN_prior"]
    N_t = max(N_last, math.exp(math.log(max(N_last, 1)) + mN_eff * dlogt))
    N_t = min(N_t, N_last * P["N_growth_cap"])
    dN = max(0.0, N_t - N_last)
    # Effective playtime for newcomers (late-horizon weighted).
    t_eff = t_c + 0.75 * horizon_hours

    rows = []
    for i, trophy in enumerate(trophies):
        C, t50, k = trophy_curve(trophy)
        # Last observed rate, falling back to earlier reads if null.
        r_last = last["rates"][i]
        if r_last is None:
            r_last = 0.01
            for h in reversed(hist):
                if h["rates"][i] is not None:
                    r_last = h["rates"][i]
                    break
        E_last = max(r_last * N_last / 100.0, 1e-6)

        # Existing cohort: earners grow by the progression curve's ratio.
        u_c = u_curve(C, t50, k, max(t_c, 0.05))
        u_t = u_curve(C, t50, k, max(t_t, 0.05))
        ratio = min(u_t / max(u_c, 1e-9), P["ratio_cap"])
        E_exist = min(E_last * ratio, float(N_last))
        # Newcomers: earn at the curve's value at their effective playtime.
        E_new = dN * u_curve(C, t50, k, max(t_eff, 0.05))
        E_t = min(E_exist + E_new, N_t)
        E_t = max(E_t, 1e-6)

        # If the history shows earners moving, blend in a log-time fit.
        ex = []
        ey = []
        for h in hist:
            r = h["rates"][i]
            if r is None:
                continue
            tt = (parse_iso(h["at"]) - launch).total_seconds() / 3600.0
            E = max(r * h["players_tracked"] / 100.0, 1e-6)
            ex.append(math.log1p(max(tt, 0.0)))
            ey.append(math.log(E))
        E_var = (max(ey) - min(ey)) > 1e-9 if len(ey) >= 2 else False
        if E_var and len(ex) >= 3:
            _, mE_fit = _fit_loglog(ex, ey)
            mE_fit = max(0.0, min(mE_fit, P["mE_cap"]))
            w = min(0.7, len(ex) / 24.0)
            E_fit = min(E_last * math.exp(mE_fit * dlogt), N_t)
            E_t = min((1 - w) * E_t + w * E_fit, N_t)
            E_t = max(E_t, 1e-6)

        r_med = clamp(E_t / N_t * 100.0)
        # Spread: wider with horizon, and wider for rarer trophies.
        s_track = P["sN"] * math.sqrt(days)
        rarity = 1.0 / math.sqrt(max(r_med, 0.05))
        s_earn = P["sE"] * math.sqrt(days) * min(rarity, 4.0)
        s = math.sqrt(P["s0"] ** 2 + s_track ** 2 + s_earn ** 2)
        s = min(s, P["s_max"])
        lg = logit(r_med)
        qs = [round4(inv_logit(lg + z * s)) for z in Z]
        for kk in range(1, 5):
            if qs[kk] < qs[kk - 1]:
                qs[kk] = qs[kk - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
