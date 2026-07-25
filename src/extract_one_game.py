"""
extract_one_game.py — Recover a single game whose tracking archive is missing
or corrupt on the default (sealneaward) mirror by pulling it from an alternate
source, then APPEND its shots to an existing shot-features CSV.

Motivating case: 01.23.2016.CHI.at.CLE.7z has an empty/malformed moments stream
on the sealneaward mirror (fails with a 'Length mismatch' in _format_tracking_data),
but the linouk23 repo hosts an intact copy of the same game.

What it does:
  1. Downloads the game's tracking .7z from --tracking-url (defaults to the
     linouk23 path for this game). Play-by-play events still come from the
     sealneaward mirror, keyed on the game_id parsed from the tracking json.
  2. Extracts shots with the exact same pipeline used for the full batch, so
     the columns match your existing CSV (48-column valid2 schema).
  3. Reindexes the new rows to the existing CSV's header and appends them,
     then records the archive name in <output>.progress so the batch runner's
     resume logic stays consistent (it won't try to re-pull this game).

Usage (run from the project root, after the main batch):
  python src/extract_one_game.py \
      --game 01.23.2016.CHI.at.CLE.7z \
      --output data/shot_features_631.csv

  # optional: point at a specific URL / different repo
  python src/extract_one_game.py --game 01.23.2016.CHI.at.CLE.7z \
      --output data/shot_features_631.csv \
      --tracking-url https://raw.githubusercontent.com/linouk23/NBA-Player-Movements/master/data/2016.NBA.Raw.SportVU.Game.Logs/01.23.2016.CHI.at.CLE.7z
"""

import argparse
import os

import pandas as pd

from game import Game
from shot_features import load_archetypes, extract_shot_features

# linouk23 hosts the raw 2015-16 logs under this directory.
LINOUK23_DIR = ("https://raw.githubusercontent.com/linouk23/NBA-Player-Movements/"
                "master/data/2016.NBA.Raw.SportVU.Game.Logs")


def default_tracking_url(game_7z: str) -> str:
    return f"{LINOUK23_DIR}/{game_7z}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--game', required=True,
                    help="archive name, e.g. 01.23.2016.CHI.at.CLE.7z")
    ap.add_argument('--output', default='data/shot_features_full.csv',
                    help="existing shot-features CSV to append to")
    ap.add_argument('--tracking-url', default=None,
                    help="full URL to the .7z (default: linouk23 path for --game)")
    ap.add_argument('--temp-dir', default='temp/recover_one')
    args = ap.parse_args()

    game_7z = args.game
    tracking_url = args.tracking_url or default_tracking_url(game_7z)

    # Parse date / teams from the archive name: MM.DD.YYYY.AWAY.at.HOME.7z
    parts = game_7z.split('.')
    date = f"{parts[0]}.{parts[1]}.{parts[2]}"
    away_team, home_team = parts[3], parts[5]

    print(f"Recovering {game_7z}")
    print(f"  tracking source: {tracking_url}")

    # Same archetype maps as the batch pipeline.
    s_df, d_df, _s_cols, _d_cols, s_base, d_base = load_archetypes()
    shooter_map = s_df.set_index('Player').to_dict('index')
    defender_map = d_df.set_index('Player').to_dict('index')

    os.makedirs(args.temp_dir, exist_ok=True)
    game = Game(date, home_team, away_team, game_7z=game_7z,
                temp_dir=args.temp_dir, verbose=True, tracking_url=tracking_url)

    df = extract_shot_features(game, shooter_map, defender_map, s_base, d_base, verbose=True)
    if df is None or len(df) == 0:
        print("No shots extracted — nothing appended.")
        return

    # Guard against silently re-adding a game that's already in the output.
    progress_file = args.output + '.progress'
    if os.path.exists(progress_file):
        with open(progress_file) as pf:
            if game_7z in {ln.strip() for ln in pf}:
                print(f"[abort] {game_7z} is already recorded in {progress_file}. "
                      "Remove that line first if you really want to re-append.")
                return

    if os.path.exists(args.output):
        # Align new rows to the existing header so columns never misalign.
        existing_cols = pd.read_csv(args.output, nrows=0).columns.tolist()
        missing = set(existing_cols) - set(df.columns)
        extra = set(df.columns) - set(existing_cols)
        if missing or extra:
            print(f"[warn] column mismatch — missing in new: {sorted(missing)} | "
                  f"extra in new: {sorted(extra)}")
        df = df.reindex(columns=existing_cols)
        df.to_csv(args.output, mode='a', index=False, header=False)
    else:
        df.to_csv(args.output, index=False)

    with open(progress_file, 'a') as pf:
        pf.write(game_7z + '\n')

    # Report new totals.
    total_rows = sum(1 for _ in open(args.output)) - 1
    n_games = 0
    if os.path.exists(progress_file):
        with open(progress_file) as pf:
            n_games = len({ln.strip() for ln in pf if ln.strip()})
    print(f"\nAppended {len(df)} shots from {game_7z}.")
    print(f"  Output now: {total_rows} shots across {n_games} games -> {args.output}")


if __name__ == '__main__':
    main()
