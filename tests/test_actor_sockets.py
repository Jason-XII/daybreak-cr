"""Actor IPC must not depend on filesystem socket paths on Linux compute nodes."""
from multiprocessing.connection import Client, Listener
import os
import sys
import threading

import pytest

from cr import actors


def test_linux_decision_addresses_are_abstract_and_distinct(monkeypatch):
    monkeypatch.setattr(actors.sys, 'platform', 'linux')
    address = actors.decision_address(123, 0)
    assert address == b'\0cr-ppo-123-0'
    assert address != actors.decision_address(123, 1)
    assert address != actors.decision_address(124, 0)


def test_other_hosts_keep_filesystem_sockets(monkeypatch):
    monkeypatch.setattr(actors.sys, 'platform', 'darwin')
    assert actors.decision_address(123, 0) == '/tmp/cr-ppo-123-0.sock'


@pytest.mark.skipif(sys.platform != 'linux', reason='abstract Unix sockets are Linux-specific')
def test_abstract_socket_roundtrip_survives_replacing_same_named_file(tmp_path):
    address = actors.decision_address(f'test-{os.getpid()}', 0)
    key = b'actor-socket-regression'
    errors = []
    # A file with the same printable name has no relationship to an abstract socket.
    path = tmp_path / address[1:].decode()
    path.write_text('old filesystem')
    with Listener(address, 'AF_UNIX', authkey=key) as listener:
        def serve():
            try:
                with listener.accept() as connection:
                    assert connection.recv() == 'game'
                    connection.send('decision')
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        path.unlink()
        path.write_text('replacement filesystem')
        with Client(address, 'AF_UNIX', authkey=key) as connection:
            connection.send('game')
            assert connection.poll(5), 'decision server did not reply'
            assert connection.recv() == 'decision'
        thread.join(timeout=5)
        assert not thread.is_alive() and not errors
    # Closing the listener releases the kernel name for a replacement actor job.
    with Listener(address, 'AF_UNIX', authkey=key):
        pass
