# allsky-meteors

Finds and scores meteors in the saved night frames of an
[indi-allsky](https://github.com/aaronwmorris/indi-allsky) camera, aiming to
surface the good-looking ones (for a web page, alerts) rather than count
every meteor.

For each night frame, with the frames before and after it:

1. **New light:** the frame minus the brighter of its neighbours (scaled to
   match), so stars, glow and anything steady cancel.
2. **Streaks:** thresholded fragments merged along common lines.
3. **Vetoes:**
   - **moving:** the streak continues into a neighbouring frame, or lines up
     with a streak in another frame within 40 s (satellites, aircraft);
   - **aircraft:** strobe beads, ripple, red/green lights, dashes, or a
     parallel companion streak;
   - **cloud:** the sky changing around the streak, or glow beside it;
   - **satellite:** a TLE match.
4. **Score:** what's left is scored 0-100 on taper, thinness, SNR and a
   shower radiant, with penalties for strobe ripple, side glow, breaks, width,
   Moon glare and the mask edge. The classes are `meteor` (>= 65), `possible`
   (>= 45) and `junk`. Twilight and evenly bright streaks are capped at
   `possible`.

Results go to `meteors.sqlite`, with a crop and a marked frame per candidate
under `$ALLSKY_METEORS_HOME/YYYY-MM-DD/`.

## Commands

```
python -m allsky_meteors.scorer run                    new frames since the last run (default)
python -m allsky_meteors.scorer night 2026-11-22 [--force]
python -m allsky_meteors.scorer frame IMAGE_ID         one frame, verbose
python -m allsky_meteors.scorer report [NIGHT]
python -m allsky_meteors.scorer inject NIGHT [--n 240] [--synthetic] [--verbose]   recall test
```

## Running

**On a SyncAPI receiver (container, the usual way).** The image
`ghcr.io/ennui2342/allsky-meteors:<version>` runs one `run` pass and exits;
schedule it every 10 minutes, one at a time. It reads:

| | |
|---|---|
| `INDI_DB` | `mysql://USER@HOST:3306/DATABASE`, the receiver's database (read-only access is enough) |
| `INDI_DB_PASSWORD` | its password |
| `ALLSKY_METEORS_CONFIG` | a JSON file of camera settings, the keys as in indi-allsky's config: `LOCATION_LATITUDE`, `LOCATION_LONGITUDE`, `LOCATION_ELEVATION`, `IMAGE_FOLDER`, `DETECT_MASK` (path to the mask PNG), `IMAGE_FLIP_V`, `IMAGE_FLIP_H`, `LENS_AZIMUTH`, `VIRTUALSKY` (`IMAGE_CIRCLE_DIAMETER`, `OFFSET_X`, `OFFSET_Y`) |
| `ALLSKY_METEORS_HOME` | a writable directory for results and the TLE cache |
| `TZ` | the receiver's time zone: SyncAPI stores frame times in the receiver's local time |

TLEs come from indi-allsky's `tle_data` table if it has any. Otherwise they
come from Celestrak's active group, fetched at most once a day (and retried
after 3 h on failure).

**On the camera's Pi.** Use indi-allsky's own virtualenv, which has
everything needed; config and TLEs come from its sqlite database:

```
scp -r allsky_meteors data install.sh PI:allsky-meteors-src/
PI$ cd allsky-meteors-src && ~/indi-allsky/virtualenv/indi-allsky/bin/python -m allsky_meteors.scorer night 2026-11-22
PI$ ./install.sh install     # or as a 10-minute user timer
```

## Evaluation tools

These need numpy, opencv, scipy and, for GMN, `ai-edge-litert`.

- `tools/nsucp_eval.py`: the single-frame rules (and optionally GMN's CNN) on
  the NightSkyUCP dataset; it also extracts real meteor brightness profiles.
- `tools/gmn_score.py`: the Global Meteor Network's RMS meteor/artefact CNN on
  crops written by `scorer mlcrops`.

## Data

- `data/nsucp_profiles.npy`: real meteor brightness profiles, derived from
  NightSkyUCP (CC BY-NC-SA 4.0; see `data/README.md`).
- `data/labels.csv`: hand labels of candidates.

## Licence

Code: MIT (`LICENSE`). `data/nsucp_profiles.npy`: CC BY-NC-SA 4.0, as the
dataset it comes from (`data/README.md`).

Written with AI assistance (Claude Code).
