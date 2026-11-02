"""Play live Clash Royale games with a trained model on a visible Android emulator, on any client (Null's Royale,
the August build, or another). Each client's private parts (its memory reader, addresses, emulator,
build check) live outside every repository, in ~/cr-clients/<name> (Client); python live.py asks which one to start.

    python play.py --client nulls [CHECKPOINT] [--queue] [--games N]
    python play.py --client NAME --watch             # record battles you play by hand (no model, no taps)

Boots the client's emulator if it is not running, checks the game build, installs the reader, starts the game, then
waits for a battle: start one by hand, or pass --queue to press Battle on the ladder. Logs and captures (every reader
frame and the model's inputs) go to ~/cr-live/runs/<client> (outside OneDrive: it stalls writes).
"""
import argparse
import collections
import hashlib
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import tempfile
import time
from types import SimpleNamespace

import torch

from cr import observation as C
from cr.policy import Agent, load
from cr.diagnostics import Capture, provenance
from cr.live_budget import LiveBudget, Pending

SDK = Path(os.environ.get('ANDROID_SDK_ROOT', Path.home() / 'Library/Android/sdk'))
ADB = shutil.which('adb') or str(SDK / 'platform-tools/adb')
CLIENTS = Path(os.environ.get('CR_CLIENTS', Path.home() / 'cr-clients'))
FRIDA_SERVER = '/data/local/tmp/frida-server'
FRAME_MS = 50
FIRST_DECISION = 90  # as in imitation learning; taps earlier are not taken anyway
STALL_S = 2.0  # the tick standing still this long once it has moved: the battle is over (the results screen keeps the
               # battle object; it also stands still for a few seconds before the start)

# Pixel_9_2 emulator screen (1080x2424): the 30 inner arena rows span y 295..1714, so 32 rows span one row more at each end.
X0, X1, Y0, Y1 = 16, 1053, 295, 1714
TILE_W, TILE_H = (X1 - X0) / 18, (Y1 - Y0) / 30
BOTTOM = Y1 + TILE_H
BUTTON = (440, 2035)  # home screen Battle (yellow) and results screen OK (blue) overlap here, below their labels
ROAD_OK = (539, 2345)  # OK at the bottom of the trophy road page after a new arena
PAIRED_OK = (706, 2035)  # OK on a results screen that also offers Play Again (left): BUTTON falls between the two
ABILITY_BUTTON = (973, 1954)  # hero / champion ability, bottom right above the hand


# ---- the client and its emulator ---------------------------------------------------------------------------------------
class Client(SimpleNamespace):
    """One game client's private parts (1 Nov 2026): CLIENTS/<name>/client.json and the reader it names, kept outside every
    repository (the readers are private: never commit them). client.json: package, activity; avd, serial, emulator (its
    extra arguments), tcp (adb connect to the serial), no_proxy (start the emulator without proxy variables); reader (a
    file in the folder: .c is built with the NDK, anything else pushed as it is), device_reader, manager_rva, root_context;
    seat ('roster': the battle roster and our account, asked through Frida at local_account's libg addresses; 'hand':
    the one hand the client reads); tables / tables_build (unit-state tables); hero_card_class (the card class of hero
    units when not 203, e.g. 306); check (libg_sha256 + live_review_patch, or version_code); model,
    temperature, overlap (the defaults of a live game)."""

    @classmethod
    def load(cls, name):
        folder = Path(name).expanduser() if '/' in str(name) else CLIENTS / name
        return cls(folder=folder, **json.loads((folder / 'client.json').read_text()))

    def card(self, card):
        """A unit's card id as training numbers it: hero units are class 203 there, hero_card_class on this client."""
        return 203000000 + card % 1000000 if self.hero_card_class and card // 1000000 == self.hero_card_class else card


def adb(serial, *arguments, check=True):
    result = subprocess.run([ADB, '-s', serial, *map(str, arguments)], capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'adb command failed')
    return result


def disable_reviews(client, library):
    """Route this exact client's broken review request through its 'unavailable' callback (one instruction; Nulls)."""
    check, patch = client.check, client.check['live_review_patch']
    serial = client.serial
    with tempfile.TemporaryDirectory(prefix='cr-review-') as directory:
        local = Path(directory) / 'libg.so'
        adb(serial, 'pull', library, local)
        original = local.read_bytes()
        actual = hashlib.sha256(original).hexdigest()
        if actual == patch['libg_sha256']:
            return
        if actual != check['libg_sha256']:
            raise RuntimeError(f'refusing to patch unknown game library: {actual}')
        offset = int(patch['file_offset'], 0)
        before, after = bytes.fromhex(patch['before']), bytes.fromhex(patch['after'])
        if original[offset:offset + len(before)] != before:
            raise RuntimeError('review instruction does not match the supported client')
        modified = original[:offset] + after + original[offset + len(before):]
        if hashlib.sha256(modified).hexdigest() != patch['libg_sha256']:
            raise RuntimeError('review patch hash does not match')
        local.write_bytes(modified)
        backup = f'/data/local/tmp/{client.name}-libg-review-enabled.so'
        adb(serial, 'shell', 'cp', '-p', library, backup)
        if adb(serial, 'shell', 'sha256sum', backup).stdout.split()[0] != check['libg_sha256']:
            raise RuntimeError('original library backup verification failed')
        adb(serial, 'shell', 'am', 'force-stop', client.package)
        try:
            adb(serial, 'push', local, library)
            adb(serial, 'shell', 'chmod', '755', library)
            adb(serial, 'shell', 'restorecon', library)
            if adb(serial, 'shell', 'sha256sum', library).stdout.split()[0] != patch['libg_sha256']:
                raise RuntimeError('review patch verification failed')
        except Exception:
            adb(serial, 'shell', 'cp', '-p', backup, library)
            raise
    print(f'Reviews disabled; original library saved at {backup}')


def reader_binary(client):
    """The client's reader for the device: its binary, or its C source built with the NDK."""
    source = client.folder / client.reader
    if source.suffix != '.c':
        return source
    cc = sorted(SDK.glob('ndk/*/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android31-clang'))
    if not cc:
        raise RuntimeError(f'building {source} needs the Android NDK under {SDK}/ndk')
    out = Path(tempfile.gettempdir()) / f'cr-reader-{client.name}'
    subprocess.run([str(cc[-1]), '-O2', '-o', str(out), str(source)], check=True)
    return out


def prepare(client, timeout=180):
    """Boot the client's emulator if needed (never touching other emulators), check the game build, install the reader
    (and Nulls' review patch), start the game. Never queues a match."""
    serial = client.serial

    def booted():
        if client.tcp:
            subprocess.run([ADB, 'connect', serial], capture_output=True)
        return adb(serial, 'shell', 'getprop', 'sys.boot_completed', check=False).stdout.strip() == '1'
    if not booted():
        env = {k: v for k, v in os.environ.items() if not (client.no_proxy and 'proxy' in k.lower())}
        log = Path.home() / 'cr-live/emulator.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:  # Nulls: -gpu host, SwiftShader is far more expensive on the fanless Mac
            subprocess.Popen([str(SDK / 'emulator/emulator'), '-avd', client.avd, *client.emulator], env=env,
                             stdout=stream, stderr=stream, start_new_session=True)
        deadline = time.monotonic() + timeout
        while not booted():
            if time.monotonic() > deadline:
                raise TimeoutError(f'{serial} did not boot within {timeout}s')
            time.sleep(2)
    adb(serial, 'root')
    time.sleep(2)
    if client.tcp:
        subprocess.run([ADB, 'connect', serial], capture_output=True)
    adb(serial, 'wait-for-device')
    if adb(serial, 'shell', 'getprop', 'ro.product.cpu.abi').stdout.strip() != 'arm64-v8a':
        raise RuntimeError('live play requires an ARM64 emulator')
    apks = adb(serial, 'shell', 'pm', 'path', client.package).stdout.splitlines()
    base = next((p.removeprefix('package:') for p in apks if p.endswith('/base.apk')), None)
    if base is None:
        raise RuntimeError(f'{client.package} is not installed')
    if 'version_code' in client.check:
        version = adb(serial, 'shell', f'dumpsys package {client.package} | grep versionCode').stdout
        if f"versionCode={client.check['version_code']}" not in version:
            raise RuntimeError(f"{client.package} {client.check['version_code']} is not installed: {version.strip()!r}")
    if 'libg_sha256' in client.check:
        library = PurePosixPath(base).parent / 'lib/arm64/libg.so'
        actual = adb(serial, 'shell', 'sha256sum', library).stdout.split()[0]
        if actual not in (client.check['libg_sha256'], client.check['live_review_patch']['libg_sha256']):
            raise RuntimeError(f'unsupported game library: {actual}')
        if actual != client.check['live_review_patch']['libg_sha256']:
            disable_reviews(client, library)
    for table in ('iptables', 'ip6tables'):  # remove only the old project's offline rules; ladder needs networking
        lines = adb(serial, 'shell', table, '-L', 'OUTPUT', '--line-numbers', '-n', check=False).stdout.splitlines()
        for rule in reversed([int(l.split()[0]) for l in lines if 'crack-cr-offline' in l and l.split()[0].isdigit()]):
            adb(serial, 'shell', table, '-D', 'OUTPUT', rule)
    reader = reader_binary(client)
    expected = hashlib.sha256(reader.read_bytes()).hexdigest()
    installed = adb(serial, 'shell', 'sha256sum', client.device_reader, check=False).stdout.split()
    if not installed or installed[0] != expected:
        adb(serial, 'push', reader, client.device_reader)
        if adb(serial, 'shell', 'sha256sum', client.device_reader).stdout.split()[0] != expected:
            raise RuntimeError('installed live reader failed hash verification')
    adb(serial, 'shell', 'chmod', '755', client.device_reader)
    adb(serial, 'shell', 'am', 'start', '-n', client.activity)


def screen(serial):
    """Which screen is up, by a few pixels: 'home' (Battle, yellow at BUTTON), 'results' (OK, blue at BUTTON), 'road' (the
    trophy road page after a battle: light blue at both ends of ROAD_OK), 'paired' (results with Play Again and OK), or None (matchmaking, loading, battle)."""
    raw = subprocess.run([ADB, '-s', serial, 'exec-out', 'screencap'], capture_output=True).stdout
    w, h = struct.unpack('<II', raw[:8])

    def pixel(x, y):
        i = len(raw) - w * h * 4 + (y * w + x) * 4
        return raw[i:i + 3]
    for y in (BUTTON[1], 2000):  # 2000: still yellow when a 'daily extra reward' line covers BUTTON (one client, 2 Nov)
        r, g, b = pixel(BUTTON[0], y)
        if r > 200 and g > 130 and b < 80:
            return 'home'
    r, g, b = pixel(*BUTTON)
    if b > 180 and r < 120:
        return 'results'
    r, g, b = pixel(*PAIRED_OK)
    if b > 180 and r < 120 and (r, g, b) != pixel(*BUTTON):
        return 'paired'
    if all(r < 120 and 150 <= g <= 210 and b > 230 for r, g, b in (pixel(470, ROAD_OK[1]), pixel(610, ROAD_OK[1]))):
        return 'road'
    return None


def world_screen(x, y, owner):
    """Native world units -> screen pixels. Our side is at the bottom; owner 0 sees x mirrored."""
    vx, vy = (18 - x / 1000, y / 1000) if owner == 0 else (x / 1000, 32 - y / 1000)
    return round(X0 + TILE_W * vx), round(BOTTOM - TILE_H * vy)


class Taps:
    """One persistent adb shell; each play is `tap card && tap tile`, acknowledged."""

    def __init__(self, serial):
        self.shell = subprocess.Popen([ADB, '-s', serial, 'shell'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      text=True, bufsize=1)

    def run(self, command):
        self.shell.stdin.write(f'{command}; echo done:$?\n')
        self.shell.stdin.flush()
        line = self.shell.stdout.readline().strip()
        if line != 'done:0':
            raise RuntimeError(f'adb failed: {command!r} -> {line!r}')

    def tap(self, x, y):
        self.run(f'input tap {x} {y}')

    def play(self, slot, x, y, owner):
        (sx, sy), (tx, ty) = (335 + 200 * slot, 2220), world_screen(x, y, owner)
        self.run(f'input tap {sx} {sy} && input tap {tx} {ty}')
        return tx, ty


def frames(client):
    """The reader's frames, for good: it keeps reading across battles and menus, and starts again when the game restarts
    (a new process; a battle in progress then is lost)."""
    command = (f'{client.device_reader} $(pidof {client.package}) {FRAME_MS} {client.manager_rva} {client.root_context} '
               '--unified 0')
    while True:
        reader = subprocess.Popen([ADB, '-s', client.serial, 'shell', command], stdout=subprocess.PIPE, text=True, bufsize=1)
        for line in reader.stdout:
            if line.startswith('{'):
                yield json.loads(line)
        print(f'the reader stopped (exit {reader.wait()}; is the game running?); starting it again', flush=True)
        time.sleep(5)


def local_account(client):
    """Our account id, asked of the running client through Frida (attached for this one call): getLocalAccountObject(
    getAccountContext(), 0, 0), its id the u32 at id_offset (Jason's clash-simulator hook_raw_capture.js, Nulls)."""
    import frida  # only for seat 'roster'
    serial, at = client.serial, client.local_account
    adb(serial, 'forward', 'tcp:27042', 'tcp:27042')
    if not adb(serial, 'shell', 'pidof frida-server', check=False).stdout.strip():
        subprocess.Popen([ADB, '-s', serial, 'shell', f'chmod 755 {FRIDA_SERVER}; {FRIDA_SERVER} >/dev/null 2>&1 &'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
    device = frida.get_device_manager().add_remote_device('127.0.0.1:27042')
    session = device.attach(int(adb(serial, 'shell', f'pidof {client.package}').stdout.split()[0]))
    try:
        script = session.create_script(f"""
const base = Process.findModuleByName('libg.so').base;
const context = new NativeFunction(base.add({at['context']}), 'pointer', []);
const local = new NativeFunction(base.add({at['local']}), 'pointer', ['pointer', 'int', 'int']);
rpc.exports.id = () => {{ const a = local(context(), 0, 0); return a.isNull() ? null : a.add({at['id_offset']}).readU32(); }};
""")
        script.load()
        return script.exports_sync.id()
    finally:
        session.detach()


def roster(client):
    """The battle's account ids in roster order (= side), read-only from /proc/<pid>/mem (adb root): battle+0xa8 is the
    roster, its count at +0x60, its accounts at +0x30 + 8 i, each account's id the u32 at +4."""
    def sh(command):
        return subprocess.run([ADB, '-s', client.serial, 'shell', command], capture_output=True, text=True,
                              check=True).stdout
    pid = sh(f'pidof {client.package}').split()[0]
    base = int(next(l for l in sh(f'cat /proc/{pid}/maps').splitlines() if 'libg.so' in l).split('-')[0], 16)

    def read(address, size):
        address &= 0x00FFFFFFFFFFFFFF  # Android heap pointers carry a tag in the top byte
        words = sh(f'dd if=/proc/{pid}/mem bs=1 skip={address} count={size} 2>/dev/null | od -An -tx1').split()
        return bytes(int(w, 16) for w in words)

    def pointer(address):
        return struct.unpack('<Q', read(address, 8))[0]
    battle = pointer(pointer(pointer(base + int(client.get_manager_global, 16)) + 0x248) + 0x90)
    entries = pointer(battle + 0xa8)
    count = struct.unpack('<i', read(entries + 0x60, 4))[0]
    return [struct.unpack('<I', read(pointer(entries + 0x30 + 8 * i) + 4, 4))[0] for i in range(count)]


def our_side(client, frame):
    """-> (our side or None, account ids). 'hand': the client reads only its own hand (the
    opponent's slots read -1). 'roster': Nulls reads both hands, so the roster and our account decide."""
    if client.seat == 'hand':
        visible = [p['side'] for p in frame['players'] if any(s >= 0 for s in p['hand_deck_indices'])]
        return (visible[0] if len(visible) == 1 else None), frame.get('accounts')
    ids = []
    for _ in range(20):  # the roster is not linked yet at the battle's first frames
        try:
            ids = roster(client)
            if len(ids) >= 2:  # the client's local account reads the roster too
                account = local_account(client)
                if ids.count(account) == 1:
                    return ids.index(account), ids
        except (struct.error, ValueError, StopIteration, subprocess.CalledProcessError):
            ids = []
        time.sleep(0.2)
    return None, ids


# ---- reader frames -> the raw decision records PlayerView reads -------------------------------------------------------------
# Plays an evolution needs, by card (evolutionRuntime's cycleRequired in the August build's converted replays). Static game
# data: the live reader reads only the progress.
CYCLE_REQUIRED = {
    26000000: 2, 26000001: 2, 26000004: 1, 26000007: 1, 26000008: 1, 26000010: 2, 26000011: 2, 26000012: 2, 26000013: 2,
    26000014: 2, 26000015: 2, 26000017: 1, 26000022: 1, 26000024: 1, 26000026: 2, 26000030: 2, 26000035: 2, 26000036: 2,
    26000037: 2, 26000040: 2, 26000043: 1, 26000044: 2, 26000045: 1, 26000047: 1, 26000049: 2, 26000050: 2, 26000055: 1,
    26000056: 2, 26000058: 2, 26000059: 2, 26000060: 1, 26000063: 1, 26000064: 2, 27000000: 2, 27000002: 2, 27000006: 2,
    27000010: 2, 27000012: 2, 27000013: 2, 28000004: 2, 28000008: 2, 28000017: 2}
EVOLUTION = 1  # deck_form_flags: 1 evolution, 2 hero


def untag(address):
    """A reader address ('0x...') without the Android heap tag in its top byte: entities come tagged, targets not."""
    return int(address, 16) & 0x00FFFFFFFFFFFFFF if address else 0


STATE_NAMES = {0: 'idle', 1: 'moving', 2: 'attacking', 3: 'dashing', 4: 'deploying', 5: 'river_jump', 6: 'underground',
               7: 'ability_underground', 10: 'ability_cast', 11: 'waiting_to_appear', 12: 'pulled',
               13: 'reeling_to_building', 14: 'hooking', 16: 'deflecting'}


class LiveRecords:
    """Reader frames -> the decision records converted replays hold (lean frame, objects(), unit_states() as
    clapha-engine's crx/observe.py builds them), using the build's exported unit-state tables."""

    def __init__(self, client):
        data = json.loads((client.folder / client.tables).read_text())
        if str(data.get('build')) != client.tables_build or not data.get('characters') or not data.get('effects'):
            raise ValueError(f'live play on {client.name} requires populated unit-state tables of build {client.tables_build}')
        self.card = client.card
        self.effects = {int(k): v for k, v in data['effects'].items()}
        self.characters = {int(k): v for k, v in data['characters'].items()}
        self.ability_buffs = set(data['abilityBuffs'])
        self.last = {}  # unit address -> (use count, use tick): time since an ability use
        # data ids the table lacks: their attack timing, spawn timer and ability counts read as zero (check after a battle)
        self.missing = collections.Counter()

    def reset(self):
        self.last = {}

    def __call__(self, frame):
        entities = frame['entities']
        ids = {untag(e['address']): (e['side'], e['category']) for e in entities}
        units = self.unit_states(frame)
        players = sorted(frame['players'], key=lambda p: p['side'])
        return {'tick': frame['game_tick'],
                'lean': {'state': {'players': [_player(p) for p in players]}, 'players': [_lean(p) for p in players]},
                'objects': [_object(e, ids, self.card) for e in entities],
                'units': [units.get(e['address'], {'nativeObjectId': e['category'], 'kind': 'other'}) for e in entities]}

    def unit_states(self, frame):
        tick = int(frame.get('game_tick') or 0)
        by_address = {}
        controllers = {}     # unit id -> its hero / champion controller (a Clone's copy has none)
        for player in frame.get('players') or ():
            for c in player.get('abilities') or ():
                for unit in c.get('units') or ():
                    if unit >= 0:
                        controllers[int(unit)] = c
        for e in frame.get('entities') or ():
            us = e.get('us')
            if us is None:
                continue
            oid = int(e.get('category') or 0)
            if e.get('kind') == 0 and 'aim' in us:
                row = {'nativeObjectId': oid, 'kind': 'projectile', 'sourceObjectId': us['source'], 'x': e['x'], 'y': e['y'],
                       'z': us['z'], 'dx': us['dx'], 'dy': us['dy'], 'aimX': us['aim'][0], 'aimY': us['aim'][1],
                       'aimObjectId': us['aim_object'] if us['aim_object'] >= 0 else None,
                       'startX': us['start'][0], 'startY': us['start'][1]}
            elif e.get('kind') == 0:
                row = {'nativeObjectId': oid, 'kind': 'area', 'remainingMs': us['area_ms'], 'durationMs': us['area_life']}
            elif int(e.get('hp', -1)) < 0:
                # no hitpoints: a death bomb's fuse (clapha-engine kind 'timed')
                row = {'nativeObjectId': oid, 'kind': 'timed', 'fuseMs': int(e.get('deploy_remaining') or 0)}
            else:
                row = self._unit(e, us, oid, int(e.get('data_id') or 0) & 0xFFFFFFFF, controllers, tick)
            by_address[e['address']] = row
        return by_address

    def _unit(self, e, us, oid, data_id, controllers, tick):
        state = int(e.get('behavior_state_raw') or 0)
        character = self.characters.get(data_id)
        if character is None:
            self.missing[data_id] += 1
            character = {}
        buffs = []
        for effect_id, left, elapsed, side, speed, hit, source in us.get('buffs') or ():
            name = self.effects.get(effect_id & 0xFFFFFFFF, {}).get('name', str(effect_id))
            buffs.append({'name': name, 'globalId': effect_id & 0xFFFFFFFF, 'remainingMs': left, 'elapsedMs': elapsed,
                          'sourceOwner': side, 'sourceObjectId': source if source >= 0 else None,
                          'hitSpeedPercent': hit, 'speedPercent': speed,
                          'stopsMovement': speed <= -100, 'stopsAttacks': hit <= -100})
        row = {'nativeObjectId': oid, 'kind': 'unit', 'name': character.get('name', ''), 'state': state,
               'stateName': STATE_NAMES.get(state), 'hidden': bool(us['hidden']), 'immune': bool(us['immune']),
               'invisible': bool(us['invisible']), 'airborne': bool(us['airborne']), 'buffs': buffs, 'attack': None,
               'charge': int(e['charge']) if e.get('has_move') else None, 'pushed': bool(us['pushed']),
               'pushRemainingMs': int(us['push_ms']), 'shield': int(us['shield'][0]), 'shieldMax': int(us['shield'][1]),
               'z': int(us.get('z', 0)), 'jumpRemainingMs': int(us.get('jump_ms', 0)),
               'invisibleAgainMs': int(us.get('invisible_again_ms', 0)),
               'surfaceAt': tuple(us['surface']) if state in (6, 7) and us.get('surface', [-1])[0] >= 0 else None}
        if 5000000 <= oid < 5000006 and 'King' in row['name']:
            row['kingAwake'] = int(e.get('kind') or 0) != 12      # kind 12 asleep, 13 awake (clapha-engine observe.py)
        elif us.get('daggers', -1) >= 0:
            row['daggers'] = int(us['daggers'])                   # Dagger Duchess's daggers left
        if character.get('spawns') and us.get('spawn_ms', -1) >= 0:
            row['spawns'] = character['spawns']
            row['spawnTimerMs'] = us['spawn_ms']
        if e.get('has_attack') and character:
            hit_speed = int(character.get('hitSpeedMs') or 0)
            timeline = int(e.get('atk_timeline') or 0)
            row['attack'] = {'stage': int(e.get('atk_stage') or 0), 'timelineMs': timeline, 'loadMs': int(e.get('atk_load') or 0),
                             'hitCount': us['hit_count'], 'hitSpeedMs': hit_speed, 'loadTimeMs': character.get('loadTimeMs'),
                             'cycleMs': timeline % hit_speed if state == 2 and hit_speed > 0 else None,
                             'dashCooldownMs': us['dash_ms'], 'dashHop': us.get('hop', 0),
                             'specialCooldownMs': us.get('special_ms', 0),
                             'returnShot': {'elapsedMs': us['shot'][1], 'totalMs': us['shot'][2]} if us.get('shot', [0])[0] else None}
        controller = controllers.get(oid)
        if controller is not None:
            most = character.get('abilityMaxCharges')
            charges = int(controller.get('charges', -1))
            uses = most - charges if most and most > 0 and charges >= 0 else None
            ability = {'buttonState': controller.get('button'), 'remainingCooldownMs': controller.get('cooldown_ms'),
                       'configuredCooldownMs': controller.get('configured_ms'), 'remainingChargesRaw': charges,
                       'maxCharges': most, 'uses': uses,
                       'used': uses > 0 if uses is not None else bool(controller.get('used')),
                       'effectDelayMs': max(0, int(controller.get('effect_delay_ms') or 0))}
            own = [b for b in buffs if b['name'] in self.ability_buffs and b['sourceObjectId'] == oid]
            active = bool(own) or state in (7, 16) or (ability['used'] and state == 4) \
                or (ability['used'] and state == 3 and row['name'] == 'GoldenKnight')
            ability['phase'] = 'casting' if state == 10 else 'active' if active \
                else 'casting' if ability['effectDelayMs'] > 0 else None
            ability['phaseElapsedMs'] = own[0]['elapsedMs'] if own and ability['phase'] == 'active' else None
            C.ability_clock(self.last, tick, [{'nativeObjectId': e['address'], 'ability': ability}])
            row['ability'] = ability
        return row


def _object(e, ids, card=lambda c: c):
    """One objects() row: the attack's target, load and the deploy timer only with an attack component; `card` maps the
    client's card ids to training's (Client.card)."""
    attack = bool(e.get('has_attack'))
    target = ids.get(untag(e.get('target')))  # no target: 0x0, no entity
    return {'nativeObjectId': e['category'], 'owner': e['side'], 'cardId': card(e['card_id']),
            'dataGlobalId': (e.get('data_id') or 0) & 0xFFFFFFFF, 'x': e['x'], 'y': e['y'],
            'hp': e['hp'] if e['hp'] >= 0 else None, 'maxHp': e['max_hp'] if e['max_hp'] >= 0 else None,
            'targetEntityKey': [target[0], -2, target[1]] if attack and target else None,
            'phaseRuntime': {'attackValidated': attack, 'deployRemainingMs': e['deploy_remaining'] if attack else None,
                             'loadRemainingMs': e['atk_load'] if attack else None}}


def _player(p):
    """players_state()'s row: hand and cycle by deck slot, elixir, deck."""
    deck = p['deck_card_ids']
    return {'owner': p['side'], 'elixirRaw': p['elixir_raw'], 'elixir': p['elixir_raw'] / 10000.0,
            'hand': [{'handIndex': i, 'deckSlot': s, 'cardId': deck[s]} for i, s in enumerate(p['hand_deck_indices']) if s >= 0],
            'cycle': [{'cycleIndex': i, 'deckSlot': s, 'cardId': deck[s]} for i, s in enumerate(p['cycle_deck_indices']) if s >= 0],
            'deck': [{'deckSlot': i, 'cardId': c} for i, c in enumerate(deck)]}


def _lean(p):
    """players_lean()'s row: the hero / champion controllers and each deck slot's evolution progress."""
    abilities = [{'controllerSlot': a['controller_slot'], 'available': a['character_id'] != 0, 'buttonState': a['button'],
                  'remainingChargesRaw': a['charges'], 'remainingCooldownMs': a['cooldown_ms'],
                  'configuredCooldownMs': a['configured_ms'], 'championObjectIds': a['units'], 'actionDataName': ''}
                 for a in p['abilities']]
    evolutions = []
    for slot, (card, form, progress) in enumerate(zip(p['deck_card_ids'], p['deck_form_flags'], p['evo_progress'])):
        need = CYCLE_REQUIRED.get(card) if form == EVOLUTION else None
        evolutions.append({'deckSlot': slot, 'cardId': card, 'evolvable': need is not None, 'progress': progress,
                           'cycleRequired': need, 'cycleRemaining': max(0, need - progress) if need else None,
                           'ready': progress >= need if need else None})
    return {'owner': p['side'], 'abilityRuntime': abilities, 'evolutionRuntime': evolutions}


# Hero / champion ability controllers: character id -> card and elixir cost (build 160402020 / d22's
# export, 44 characters; data ids are stable across builds, so the August heroes are in it). Live play names each button
# by its card, places it on the screen and budgets its press with it.
_COSTS = json.loads((C.ASSETS / 'ability_costs.json').read_text())
HERO_CARDS = {int(k): row['card_id'] for k, row in _COSTS['characters'].items()}
ABILITY_COSTS = {int(k): float(row['elixir_cost']) for k, row in _COSTS['characters'].items()}
ABILITY_LEFT = (107, 1951)  # the left one of two ability buttons (1 Nov screenshot, Golden Knight + Archer Queen deck)


def ability_buttons(player):
    """{controller slot: screen position} of our ability buttons. With two ability cards in the deck the first in deck
    order has the left button and the other the right one; with one, the right one (the user, 1 Nov). The game's
    controller numbers do not follow the deck (1 Nov: Archer Queen, second in the deck, was controller 1), so each
    controller is placed by its card (its character id -> HERO_CARDS)."""
    deck = player['deck_card_ids']
    used = [a for a in player.get('abilities') or () if a.get('character_id', -1)]  # character 0: an empty slot
    if len(used) == 1:  # the one button is on the right, whatever its card (also one HERO_CARDS lacks)
        return {used[0]['controller_slot']: ABILITY_BUTTON}
    cards = {a['controller_slot']: HERO_CARDS.get(a['character_id']) for a in used}
    order = sorted((deck.index(c), slot) for slot, c in cards.items() if c in deck)
    spots = [ABILITY_LEFT, ABILITY_BUTTON] if len(order) == 2 else [ABILITY_BUTTON]
    return {slot: spot for (_, slot), spot in zip(order, spots)}


class LivePlays:
    """Confirmed hand changes and currently visible queued cards, in PlayerView's command format.

    Our own queue entry is intent, not proof of execution: only a hand change confirms our play; matching it to a
    previously observed command recovers its coordinates. The opponent's commands reach our queue from the server, which
    has accepted them: they land at their execute tick (1 Nov: 180 of 183 on 31 Oct's Nulls captures were confirmed by
    the hand, the 3 others did not match by card id), which needs no view of their hand (where it is hidden,
    the 'cards they played' input was empty). Where their hand is readable (Nulls), a change in it that no queued
    command explains still counts: Training Camp's trainer plays never pass through the queue (1 Nov: 0 queued, 11 seen in
    the hand; a ladder game: 39 queued, 38 in the hand). Each command's 'seen' is the tick it first showed in the queue.
    Create a new instance per battle: sequence numbers and native addresses may be reused."""

    def __init__(self, owner=None):
        self.owner = owner
        self.hands, self.landed, self.seen, self.flying = {}, [], {}, []

    def update(self, frame):
        if not frame.get('coherent', True) or frame['failure'] != 'none':
            return list(self.landed)
        tick = frame['game_tick']
        self.flying = []
        for c in frame.get('commands', ()) if frame.get('queue_ok') else ():
            key = (c['owner'], c['sequence'], c['issue_tick'])
            command = self.seen.setdefault(key, dict(c, first_seen_tick=tick, seen=tick, confirmed=False))
            if c['execute_tick'] > tick and not command['confirmed']:
                self.flying.append((c['execute_tick'], command))
        if self.owner is not None:
            for command in self.seen.values():
                if command['owner'] != self.owner and not command['confirmed'] and tick >= command['execute_tick']:
                    command['confirmed'] = True
                    self.landed.append((command['execute_tick'], dict(command)))
        for p in frame['players']:
            theirs = self.owner is not None and p['side'] != self.owner
            deck = p['deck_card_ids']
            hand = {i: deck[s] for i, s in enumerate(p['hand_deck_indices']) if s >= 0}
            before = self.hands.get(p['side'])
            if before is not None:
                for i, card in before.items():
                    if hand.get(i) == card:
                        continue
                    if theirs and any(c['owner'] == p['side'] and c['card_id'] == card and c['confirmed']
                                      and 0 <= tick - c['execute_tick'] <= 10 for c in self.seen.values()):
                        continue  # landed from the queue already
                    candidates = [c for c in self.seen.values() if c['owner'] == p['side'] and c['card_id'] == card
                                  and not c['confirmed'] and 0 <= tick - c['execute_tick'] <= 10]
                    command = min(candidates, key=lambda c: c['execute_tick']) if candidates else \
                        {'owner': p['side'], 'card_id': card}
                    command['confirmed'] = True
                    self.landed.append((command.get('execute_tick', tick), dict(command)))
            self.hands[p['side']] = hand
        self.flying = [(due, c) for due, c in self.flying if not c['confirmed']]
        return list(self.landed)

    def commands(self, pending=None, owner=None, *, reservations=(), tick=None):
        """Known history plus visible commands. A just-sent own tap fills the queue's acknowledgement delay only."""
        flying = list(self.flying)
        if pending and not any(c['owner'] == owner and c['card_id'] == pending['card'] for _, c in flying):
            flying.append((pending['due'], {'owner': owner, 'card_id': pending['card'],
                                            'x': pending['x'], 'y': pending['y'], 'tap': pending['tick']}))
        for due, c in reservations:
            if c.get('kind') == 'ability':  # our press in flight (PlayerView's pending flag)
                flying.append((due, c))
                continue
            match = next((q for _, q in flying if q['owner'] == c['owner'] and q.get('card_id') == c['card_id']), None)
            if match is not None:
                match['tap'] = c['tap']
            else:
                flying.append((max(due, tick + 1) if tick is not None else due, c))
        return self.landed + flying


# ---- a battle -----------------------------------------------------------------------------------------------------------
def board(record):
    """Live log frame rows: owner, card, x, y, hp, max hp, type (0 troop, 1 tower, 2 projectile, 3 area)."""
    def kind(o):
        if o['hp'] is None:
            return 2 if 4000000 <= o['nativeObjectId'] < 5000000 else 3
        return 1 if o['cardId'] == -1 else 0
    return [[o['owner'], o['cardId'], o['x'], o['y'], -1 if o['hp'] is None else o['hp'],
             -1 if o['maxHp'] is None else o['maxHp'], kind(o)] for o in record['objects']]


def ability_costs_now(player, overrides):
    """Each button's press cost: an override, else its character's (ABILITY_COSTS); None while unknown."""
    def character(k):
        return next((int(a.get('character_id', 0)) & 0xFFFFFFFF for a in player.get('abilities') or ()
                     if a['controller_slot'] == k + 1), 0)
    return tuple(o if o is not None else ABILITY_COSTS.get(character(k)) for k, o in enumerate(overrides))


def play_battle(stream, first, model, vocab, records, owner, taps, log, *, overlap=False, ability_costs=(None, None),
                capture=None):
    """One battle from its first frame to its end -> the last record. ability_costs: per-button overrides (else each
    hero's cost from ABILITY_COSTS); capture: a diagnostics Capture for every frame and the model's exact inputs."""
    other = 1 - owner
    records.reset()
    record = records(first)
    forms = next(p['deck_form_flags'] for p in first['players'] if p['side'] == owner)
    view = C.PlayerView(owner, vocab, C.cached_legal(), forms)
    agent, plays = Agent(model, owner), LivePlays(owner)
    budget = LiveBudget(owner, overlap=overlap)
    model.overlap = overlap
    unknown_costs = set()
    log({'event': 'start', 'owner': owner, 'tick': first['game_tick']})
    decided, reported = None, 0
    battle = first['chain']['battle']
    frame, last_tick, moved = first, first['game_tick'], time.monotonic()
    queue_seen = moved
    while True:
        if capture is not None:
            capture.frame(frame)
        queue_ok = frame['game_tick'] < FIRST_DECISION or frame.get('queue_ok')
        if queue_ok:
            queue_seen = time.monotonic()
        elif time.monotonic() - queue_seen > 3:  # one frame without it happens (2026-10-16, tick 686 mid-game): skip those
            raise RuntimeError('command queue unavailable: is the client\'s queue-enabled reader installed?')
        if frame['failure'] == 'none' and frame.get('coherent', True) and queue_ok:
            landed = plays.update(frame)
            for tick, c in landed[reported:]:
                log({'event': 'landed' if c['owner'] == owner else 'opponent', 'tick': tick, 'card': c['card_id'],
                     'world': [c['x'], c['y']] if 'x' in c else None,
                     **{k: c[k] for k in ('issue_tick', 'execute_tick', 'first_seen_tick', 'sequence') if k in c}})
            reported = len(landed)
            tick = frame['game_tick']
            for p in budget.update(frame):
                log({'event': 'unconfirmed', 'tick': tick, 'kind': p.kind, 'card': p.card, 'tap': p.tap,
                     **({'controller': p.slot} if p.kind == 'ability' else {})})
            for p in budget.confirmed:
                log({'event': 'ability_confirmed', 'tick': tick, 'tap': p.tap, 'controller': p.slot})
            if tick >= FIRST_DECISION and (decided is None or tick >= decided + 5):
                decided = tick
                record = records(frame)
                me = next(p for p in record['lean']['state']['players'] if p['owner'] == owner)
                inputs = view.step(record, plays.commands(reservations=budget.commands(), tick=tick))
                elixir = me['elixirRaw'] / 10000
                hand = {h['handIndex']: h['cardId'] for h in me['hand']}
                costs = [round(float(h[0]) * 10, 6) for h in inputs['hand_float']]  # includes Mirror's surcharge
                due = [budget.card_due(elixir, tick, i, hand[i], costs[i]) if i in hand else None for i in range(4)]
                ours = next(p for p in frame['players'] if p['side'] == owner)
                buttons = ability_buttons(ours)  # {controller: screen position}, from the deck order (1 Nov)
                ready = [next((a for a in ours.get('abilities', ()) if a['controller_slot'] == k + 1 and k + 1 in buttons
                               and a['button'] in (2, 4) and not a.get('cooldown_ms') and a.get('charges', -1) != 0
                               and any(u >= 0 for u in a.get('units') or ())), None) for k in (0, 1)]
                press_costs = ability_costs_now(ours, ability_costs)
                budget.ability_costs = tuple(0.0 if c is None else float(c) for c in press_costs)
                for k in (0, 1):
                    if ready[k] and press_costs[k] is None and ready[k].get('character_id') not in unknown_costs:
                        unknown_costs.add(ready[k].get('character_id'))
                        log({'event': 'ability_cost_unknown', 'tick': tick, 'controller': k,
                             'character_id': ready[k].get('character_id'), 'reason': 'press blocked: no cost known'})
                ability_due = [budget.ability_due(elixir, tick, k) if ready[k] and press_costs[k] is not None else None
                               for k in (0, 1)]
                if capture is not None:
                    capture.audit(frame, records, vocab, inputs)
                    agent.input_capture = lambda x, state, tick=tick: capture.tensors(tick, x, state)
                started = time.perf_counter()
                choice = agent.act(inputs, live_playable=[d is not None for d in due],
                                   live_abilities=[d is not None for d in ability_due])
                log({'event': 'decision', 'tick': tick, 'elixir': me['elixirRaw'], 'choice': choice[0],
                     'inference_ms': round((time.perf_counter() - started) * 1000, 2), 'ability_costs': press_costs,
                     'value': round(agent.value, 3), 'p_what': [round(q, 4) for q in agent.p_what], 'board': board(record),
                     'abilities': [[a['controllerSlot'], a['available'], a['buttonState'], a['remainingChargesRaw'],
                                    a['remainingCooldownMs']] for a in next(p for p in record['lean']['players']
                                                                          if p['owner'] == owner)['abilityRuntime']],
                     'opponent_commands': [c for _, c in plays.flying if c['owner'] == other]})
                if choice[0] == 'ability' and ability_due[choice[1]] is not None:
                    k = choice[1]
                    taps.tap(*buttons[k + 1])
                    budget.pending.append(Pending('ability', budget.ability_costs[k], tick, ability_due[k], k,
                                                  charges=ready[k].get('charges', -1), used=ready[k].get('used'),
                                                  cooldown_ms=ready[k].get('cooldown_ms', 0)))
                    log({'event': 'tap', 'tick': tick, 'card': None, 'ability': True, 'controller': k,
                         'screen': buttons[k + 1], 'cost': budget.ability_costs[k], 'due': ability_due[k]})
                elif choice[0] == 'card':
                    _, slot, x, y = choice
                    card = next((h['cardId'] for h in me['hand'] if h['handIndex'] == slot), None)
                    # Recheck before tapping; both gates use the same complete
                    # ledger, including pending abilities and regeneration.
                    payment = budget.card_due(elixir, tick, slot, card, costs[slot]) if card is not None else None
                    if payment is not None:
                        started = time.perf_counter()
                        pixels = taps.play(slot, x, y, owner)
                        budget.pending.append(Pending('card', costs[slot], tick, payment, slot, card, x, y))
                        log({'event': 'tap', 'tick': tick, 'card': card, 'slot': slot, 'world': [x, y], 'screen': pixels,
                             'cost': costs[slot], 'due': payment,
                             'input_ms': round((time.perf_counter() - started) * 1000)})
        frame = next(stream)
        if frame['game_tick'] != last_tick:
            last_tick, moved = frame['game_tick'], time.monotonic()
        replaced = frame['failure'] == 'none' and frame['chain']['battle'] != battle
        stalled = last_tick > first['game_tick'] and time.monotonic() - moved > STALL_S
        if not frame['battle_active'] or replaced or stalled:
            return record


def next_battle(stream, client, finished, taps=None):
    """The first frame of a new battle, once our side is known on a 'hand' client (the results screen keeps a finished
    battle active with its tick standing still). With taps (--queue): OK on results / trophy road, then Battle."""
    checked = time.monotonic()
    frame = next(stream)
    while not (frame['battle_active'] and frame['failure'] == 'none' and frame['chain']['battle'] != finished
               and frame['game_tick'] < FIRST_DECISION and (client.seat != 'hand' or our_side(client, frame)[0] is not None)):
        if taps is not None and time.monotonic() - checked > 5:
            checked = time.monotonic()
            now = screen(client.serial)
            if now:
                taps.tap(*{'road': ROAD_OK, 'paired': PAIRED_OK}.get(now, BUTTON))
        frame = next(stream)
    return frame


def watch_battle(stream, first, records, capture):
    """--watch: record one battle you play by hand (no model, no taps) -> its last record."""
    battle, frame = first['chain']['battle'], first
    last_tick, moved = first['game_tick'], time.monotonic()
    while True:
        capture.frame(frame)
        last = frame
        frame = next(stream)
        if frame['game_tick'] != last_tick:
            last_tick, moved = frame['game_tick'], time.monotonic()
        replaced = frame['failure'] == 'none' and frame['chain']['battle'] != battle
        stalled = last_tick > first['game_tick'] and time.monotonic() - moved > STALL_S
        if not frame['battle_active'] or replaced or stalled:
            return records(last)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path, nargs='?', help="a model checkpoint; default: the client's model")
    parser.add_argument('--client', required=True, help=f'a client folder name in {CLIENTS} (e.g. nulls) or a path')
    parser.add_argument('--games', type=int, default=1, help='battles to play (0: until stopped)')
    parser.add_argument('--losses', type=int, default=0, help='instead of --games: play until this many losses')
    parser.add_argument('--queue', action='store_true', help='press OK and Battle to start each ladder game')
    parser.add_argument('--watch', action='store_true', help='record battles you play by hand: no model, no taps')
    parser.add_argument('--temperature', type=float, help="card and tile temperature; default: the client's")
    parser.add_argument('--overlap', type=int, choices=(0, 1), help="allow two outstanding card taps; default: the client's")
    parser.add_argument('--ability-costs', type=float, nargs=2, metavar=('BUTTON_1', 'BUTTON_2'),
                        help="override each button's press cost (default: its hero's, from assets/ability_costs.json)")
    parser.add_argument('--out', type=Path, help='logs and captures; default: ~/cr-live/runs/<client>')
    args = parser.parse_args()
    client = Client.load(args.client)
    checkpoint = Path(args.checkpoint or client.model).expanduser()
    temperature = client.temperature if args.temperature is None else args.temperature
    overlap = bool(client.overlap if args.overlap is None else args.overlap)
    out = args.out or Path.home() / 'cr-live/runs' / client.name
    torch.set_num_threads(2)
    model = vocab = None
    if not args.watch:
        model, vocab, _ = load(checkpoint)
        model.temperature = temperature
    records = LiveRecords(client)
    prepare(client)
    taps, stream = Taps(client.serial), frames(client)
    out.mkdir(parents=True, exist_ok=True)
    finished = None  # the last battle played: the results screen keeps it readable
    lost = 0
    for game in itertools.count() if args.losses or not args.games else range(args.games):
        print(f'{client.name}: waiting for a battle ({"start it and play by hand" if args.watch else "queue" if args.queue else "start it"})...',
              flush=True)
        frame = next_battle(stream, client, finished, taps if args.queue and not args.watch else None)
        finished = frame['chain']['battle']
        owner, ids = our_side(client, frame)
        if owner is None:
            raise SystemExit(f'cannot tell our side in this battle (accounts {ids})')
        path = out / time.strftime(('watch-' if args.watch else '') + '%Y%m%d_%H%M%S.jsonl')
        with path.open('w') as f:
            def log(row):
                f.write(json.dumps(row) + '\n')
                f.flush()
            log({'event': 'checkpoint', 'client': client.name, 'path': None if args.watch else str(checkpoint),
                 'temperature': temperature, 'overlap': overlap, 'ability_cost_overrides': args.ability_costs})
            log({'event': 'roster', 'accounts': ids, 'owner': owner})
            # every reader frame and the model's exact inputs, for comparing live inputs with training (tools/live_census.py)
            capture = Capture(path.with_suffix('.inputs.jsonl.gz'), max_queue=4096)
            here = Path(__file__).resolve().parent
            capture.emit({'event': 'diagnostic_schema', 'version': 1, 'client': client.name,
                          'files': provenance([*([] if args.watch else [checkpoint]), client.folder / client.tables,
                                               client.folder / client.reader, here / 'play.py', here / 'cr/policy.py',
                                               here / 'cr/observation.py', here / 'cr/diagnostics.py']),
                          'obj_float_names': C.OBJ_FLOAT_NAMES, 'hand_float_names': C.HAND_FLOAT_NAMES,
                          'scalar_names': C.SCALAR_NAMES})
            try:
                if args.watch:
                    records.reset()
                    log({'event': 'start', 'owner': owner, 'tick': frame['game_tick']})
                    last = watch_battle(stream, frame, records, capture)
                else:
                    last = play_battle(stream, frame, model, vocab, records, owner, taps, log, overlap=overlap,
                                       ability_costs=tuple(args.ability_costs or (None, None)), capture=capture)
            finally:
                log(capture.close())
            alive = {'objects': [o for o in last['objects'] if o['hp'] is None or o['hp'] > 0]}  # the last frame still
            score = (C.crowns(alive, owner), C.crowns(alive, 1 - owner))                     # lists a tower at 0 hp
            log({'event': 'end', 'tick': last['tick'], 'crowns': score})
            lost += score[0] < score[1]  # ponytail: equal crowns count as no loss (a tiebreak loss would be missed)
            print(f'game {game + 1}: crowns {score[0]}-{score[1]} -> {path}' + (f' ({lost} lost)' if args.losses else ''),
                  flush=True)
            if args.losses and lost >= args.losses:
                break


if __name__ == '__main__':
    main()
