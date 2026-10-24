# Converting RoyaleAPI battles exactly

A converted battle is only usable when the engine rebuilds it as it was played: every tower ends at the HP RoyaleAPI
recorded. These rules took the 2,000-battle sample of the 700k dump from 54% (convert.py before them, October) to **95.6%
exact** in the 6-18 October window (1,076 of 1,126 on the native engine, 73 of them with a found seed), and 1-5 October
to 162 of 165 non-dice battles. On the first 20 parts of the 770k (2026-10-25): **15,222 of 15,800 converted battles exact (96.3%)**
(below). Each rule was measured on real battles and made its battles exact without breaking any
exact one. The full record (every measurement, the engine side) is clapha-engine `docs/training/CONVERSION_RULES.md`.

Needs clapha-engine at ba78ae1 or later (`CrxEnv.selection_of(elixir_less=)`, `queue_card_at(selection=)`,
`ready_ability_sources`, `queue_ability_action_at(check_ready=False)`).

## The rules (all in convert.py)

| # | Rule | Was |
|---|---|---|
| 1 | RoyaleAPI's tick (`data-t`) is the command's **issue** tick; the game lands a play or press **21 ticks later**. Labels: the last decision before the issue tick | landed at tick + 1, 20 ticks early (a live friendly: 71 of 71 plays; the 300-replay sample 210 -> 249 of 266) |
| 2 | A press names its unit; RoyaleAPI does not. Where two different cards were ready, `play_exact` tries each and keeps the one that reproduces the battle | newest ready unit |
| 3 | A press is checked at its issue, not 21 ticks later (`check_ready=False`) | refused when the button was not ready at execution |
| 4 | A card's selection (form, cost) is made at the tap (`selection_of` at the issue tick) | taken at execution |
| 4b | That selection sees the **client's** bar: engine elixir less the player's own plays still in flight (the engine pays at landing) | Spirit Empress mounted where she was on foot |
| 4c | **Mirror** copies the card that has landed when the Mirror lands: no selection at the tap | copied the card before |
| 5 | The King's level is not published: tower troop's level, else the side's commonest card level, whichever reproduces | one level for both Kings |
| 6 | Run until the game's own rules end the battle (7,200 ticks only as a safety stop) | the recorded length (cut tiebreakers short) |
| 7 | Each battle on the game data live at its time (below; `tools/assign_by_date.py`) | one engine per month |
| 9 | Cards kept out of the starting hand (Elixir Collector, Mirror) wait at a queue place set by the **seed alone**: `calibrate_deal` searches seeds | moved the card between deck slots and refused (51 of 1,791) |
| 10 | Score only the six tower objects (native ids 5,000,000-5,000,005) | a troop on a fallen tower's spot was read as the tower |
| 11 | A hot update reaches players **during** its first day: such a battle runs on both data versions (`alt`), the one that reproduces counts | data by date only |

Dice: Skeleton King, Hunter, Elixir Golem (and Graveyard on older data) draw from the battle's seed, which RoyaleAPI does not
publish. `--seeds N` tries up to N seeds per battle that misses (stops after 5 seeds give the same result).

## Game data by battle time

| From (UTC) | Data (`--engine` name) | Engine code |
|---|---|---|
| 10-19 ~09:40 | d22: 16.402.22 (real) | 160402020 |
| 10-06 ~06:30 | d19: 16.402.19 (real) | 160402020 |
| 28 Sep - 1 Oct | oct1: rebuilt -- Hero Ice Wizard rework + three `V16_*_BUG` flags (in no official post) + the Electro Wizard hero (5 Oct; its two files must be listed in `data_manifest.toml`) | 160402012 |
| 09-17 | sep17: rebuilt, the 16 Sep post (4 changes) | 160402012 |
| 09-07 ~05:30 | d2b: 16.402.2 from app 160402012 (8 Sep post) | 160402012 |
| 09-01 | d2a: 16.402.2 from app 160402002 | 160402002 |

The asset server serves only the current data (older versions 403), so sep17, oct1 (and jul6, aug26 for the summer) were
rebuilt from real later files and accepted only when their battles turned exact. **Every engine and data version used,
and a kit that replicates the result** (our converter at the versions used, the sample's 1,791 kept battles, replicate.py):
the release `conversion-2026-10-24` of ClashAlpha/clapha-engine. For future
battles: save each data version when it goes live (its fingerprint changes) and none has to be rebuilt.

## Converting the 770k (ClashAlpha/RoyaleAPI-Replays)

Everything is in this repo; the engine is clapha-engine (crx/), and every engine and data version is in its release
`conversion-2026-10-24` (engines-*.tar.zst, data-*.tar.zst: unpack to ENGINES/ and DATA/).

1. `python tools/hf_ingest.py OUT --parts 0-773 --token-file ~/.hf_token` -- downloads each part from the private Hugging
   Face dataset (checked against its SHA256SUMS), keeps Ranked, Ladder and Grand Challenge from July 2026 on, and writes
   `OUT/payloads/` (converter input) and `OUT/ingest/` (every battle, kept or left out and why). Needs pyarrow.
2. `python tools/assign_by_date.py OUT/assignment.jsonl OUT/payloads` -- each battle's data version by its time, `alt` on
   an update's first day.
3. `python convert.py --assignment OUT/assignment.jsonl --replays OUT/payloads --clapha-engine ~/clapha-engine --out ~/il-data/v1
   --engine d22jul ENGINES/150535029 DATA/d22jul --engine jul6 ENGINES/150535029 DATA/jul6
   --engine d86 ENGINES/150535029 DATA/d86 --engine aug26 ENGINES/150535029 DATA/aug26 --engine d2a ENGINES/160402002 DATA/d2a
   --engine d2b ENGINES/160402012 DATA/d2b --engine sep17 ENGINES/160402012 DATA/sep17 --engine oct1 ENGINES/160402012 DATA/oct1
   --engine d19 ENGINES/160402020 DATA/d19 --engine d22 ENGINES/160402020 DATA/d22`
   -- resumable; update-day battles it did not reproduce run again on their `alt` data; `manifest.json` is what train.py reads.
4. `python tools/conversion_report.py ~/il-data/v1 OUT --ledger ledger.jsonl` -- every battle's status (exact, exact with a
   seed, dice: needs the true seed, not reproduced, refused, left out) with the reason, and the exact rate by day (a drop
   flags an update or mechanic not yet known).

Checked 2026-10-24 on part 1 in the Linux container: ingest identical to clapha-engine's tested pipeline on 2,705 of 2,705
battles; 16 converted, 15 exact + 1 dice (4 seeds in the test). `--seeds 1024` is the default: the goal is the most exact battles,
not speed (64 finds 91% of what 256 finds at a quarter of the cost, if time ever matters). A battle exact only with a searched seed
is reported apart (`exact with a seed`): its end state is the recorded one, but the seed is not the battle's true one, so its dice
may fall differently mid-battle.

## Result on the first 20 parts (2026-10-25)

Parts 0-19 (20,000 battles) through clapha-engine's pipeline (`crx/pipeline/run.py`: the same ingest -- identical to
`tools/hf_ingest.py` on 2,705 of 2,705 battles -- and the same rules as convert.py), native Windows engine, 12 workers,
seed cap 64:

| status | battles |
|---|---|
| exact | 14,263 |
| exact with a seed | 959 |
| dice: needs the true seed | 378 |
| not reproduced | 200 |
| pending (July - early September engines not on that PC) | 403 |
| left out by design (before July 2026, friendlies, tournaments, clan-mate, Classic Challenge, River Race, Showdown) | 3,797 |

**15,222 of 15,800 converted exact (96.3%)**; by data version d19 96.6%, d22 97.2%, oct1 94.6%, sep17 94.8%; every day from
7 Oct 94-98%. Report, a ledger row per battle with the **recipe** of every exact one (data version, King choice, press
choices, seed: what replays it with no search) and the raw results: clapha-engine
`docs/training/evidence/2026-10-25-pc-20-parts/` (private; you have access). convert.py's own results match the pipeline's on
the battles compared so far (the PR #14 check and part 1); a larger side-by-side check is open.

Running it: a run that is stopped resumes where it was (every battle written is skipped). Start it where closing a terminal
cannot end it (`nohup` / `tmux`; on Windows a closed console window ends every process in it). Near the end of each data
version the slowest battles (dice, up to `--seeds` runs each) are left; the workers share one queue, so all stay busy.

## Finding why a battle misses

1. Group the misses by date (a cluster is a data window), by card against the exact battles, by mode.
2. Per battle: towers, each side's elixir spent and leaked against RoyaleAPI's, failed plays and presses, plays held for
   elixir (the first held play: the rebuild already differs before it).
3. Every hit on each tower and what dealt it: one hit too many = a unit died a few ticks late.
4. Move single plays a few ticks as a lead finder, compare the two runs object by object, then find the mechanism. A
   match is a lead, never proof: one battle matched with either of two Miners delayed; its cause was rule 4b.
5. Build candidate data from the real later file (never an invented value) and date it by the battles that need it.
6. Run every fix on all battles with that card: it must fix misses and break none.

Open: the dice seed (the server picks it; clapha-engine docs/training/RNG.md), a play's landing tick after a bad network (RoyaleAPI
shows only the tap), 8 misses in 6-18 October without a confirmed cause.
