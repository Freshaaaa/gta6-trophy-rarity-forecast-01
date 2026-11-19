# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/rank/Freshaaaa/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/reward/Freshaaaa/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

A Python 3.13 (standard library only) forecaster for the make.fun
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
competition: it predicts, for each of GTA VI's 55 trophies, the five
quantiles (5/25/50/75/95%) of the Exophase earn rate one week after launch.
The model decomposes each rate into earners / players tracked, models the
tracked-player count (which jumps when Exophase refreshes and dilutes every
rate) and the earners (which follow a saturating progression curve set from
each trophy's tier and description: story chapters rise fast to a high
ceiling, grind/collectibles rise slowly to a low one, the platinum barely
moves), and reports an honest logit-space spread that widens with the
horizon and with rarity.

Run it:

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Referral: https://make.fun/?ref=pstpm556
