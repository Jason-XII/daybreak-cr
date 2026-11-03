"""Ask the client, mode and number of games, then run play.py with them (~/cr-live/live.sh starts this)."""
import os, sys
client = input('Client (nulls / tencent): ').strip()
mode = input('Mode (manual / queue / watch): ').strip()
games = input('Games (empty: until Ctrl+C): ').strip() or '0'
flags = [] if mode == 'manual' else [f'--{mode}']
play = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'play.py')
os.execv(sys.executable, [sys.executable, '-u', play, '--client', client, '--games', games, *flags, *sys.argv[1:]])
