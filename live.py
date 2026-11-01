"""Start live play: asks which client (the folders in ~/cr-clients, or $CR_CLIENTS), then the mode and how many games, and
runs play.py with that client's model and settings.

    python live.py            (~/cr-live/live.sh runs it with the live Python environment)

Modes: manual (the model plays every battle you start), queue (it also presses Battle on the ladder and taps through the
results), watch (you play by hand; every frame is recorded for tools/live_census.py).
"""
import os
from pathlib import Path
import sys

CLIENTS = Path(os.environ.get('CR_CLIENTS', Path.home() / 'cr-clients'))


def ask(question, options):
    while True:
        answer = input(f'{question} ({" / ".join(options)}): ').strip().lower()
        match = [o for o in options if answer and o.startswith(answer)]
        if len(match) == 1:
            return match[0]


def main():
    clients = sorted(p.parent.name for p in CLIENTS.glob('*/client.json'))
    if not clients:
        sys.exit(f'no clients in {CLIENTS} (each is a folder with a client.json)')
    client = ask('Client', clients)
    mode = ask('Mode', ['manual', 'queue', 'watch'])
    games = input('Games (a number; empty: until Ctrl+C): ').strip() or '0'
    command = [sys.executable, '-u', str(Path(__file__).with_name('play.py')), '--client', client, '--games', games,
               *([f'--{mode}'] if mode != 'manual' else []), *sys.argv[1:]]
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
