"""WebGUI のポート番号の決め方（app/webgui_port.py）と、amamizu gui の表示。サーバを起動するのは、番号を取られたときの終了と、
接続を閉じたあとの SIGINT での終了を確かめる試験の 2 つだけ（子プロセス）。"""
import errno
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from amamizu import cli
from amamizu.app import webgui_port as wp


def _bound(port: int = 0) -> socket.socket:
    """127.0.0.1 の port を bind したままのソケット（使用中の状態）。0 なら OS が選んだ空き番号。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((wp.HOST, port))
    return sock


@pytest.fixture
def held():
    """使用中にしたソケットを、試験の終わりに閉じる。"""
    socks: list[socket.socket] = []
    yield socks
    for s in socks:
        s.close()


def _pair(held) -> int:
    """port と port + 1 の両方が空いている番号を探し、port を使用中にして返す（port + 1 は空きのまま）。"""
    for _ in range(50):
        first = _bound()
        port = first.getsockname()[1]
        if port < wp.MAX_PORT and wp.port_state(port + 1) is wp.PortState.FREE:
            held.append(first)
            return port
        first.close()
    pytest.fail("連続した 2 つの空き番号が見つからない")


def test_omitted_uses_first_when_free(monkeypatch):
    with _bound() as probe:
        free = probe.getsockname()[1]
    monkeypatch.setattr(wp, "DEFAULT_PORT", free)
    assert wp.resolve_port(None).port == free


def test_omitted_moves_to_next_when_first_busy(monkeypatch, held):
    port = _pair(held)
    monkeypatch.setattr(wp, "DEFAULT_PORT", port)
    assert wp.resolve_port(None).port == port + 1


def test_omitted_all_busy_raises(monkeypatch, held):
    port = _pair(held)
    held.append(_bound(port + 1))
    monkeypatch.setattr(wp, "DEFAULT_PORT", port)
    monkeypatch.setattr(wp, "PORT_TRIES", 2)
    with pytest.raises(ValueError, match=f"{port} から {port + 1} までがすべて使用中、または使用できません"):
        wp.resolve_port(None)


def test_specified_busy_raises_with_number(held):
    sock = _bound()
    held.append(sock)
    port = sock.getsockname()[1]
    with pytest.raises(ValueError, match=f"ポート番号 {port} は使用中です。--port で別の番号を指定するか、--port を省略してください。"):
        wp.resolve_port(port)


def test_specified_free_returns_it():
    with _bound() as probe:
        free = probe.getsockname()[1]
    assert wp.resolve_port(free).port == free


def _server_socket() -> socket.socket:
    """WebGUI のサーバ（asyncio）と同じ設定のソケット。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if not wp.IS_WINDOWS:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return sock


def _assert_server_can_serve(port: int) -> None:
    """WebGUI と同じ設定で bind・listen でき、HOST の port への接続をそのソケットが受ける。"""
    with _server_socket() as server:
        server.bind((wp.HOST, port))
        server.listen()
        with socket.create_connection((wp.HOST, port), timeout=5):
            server.settimeout(5)
            server.accept()[0].close()


def test_port_with_just_closed_connection_is_free():
    """サーバ側から接続を閉じ、listen のソケットも閉じた直後の番号は、port_state が FREE で、WebGUI のサーバと同じ設定の bind・listen・接続の受け付けができる。"""
    with _server_socket() as server:
        server.bind((wp.HOST, 0))
        server.listen()
        port = server.getsockname()[1]
        with socket.create_connection((wp.HOST, port)) as client:
            conn, _ = server.accept()
            conn.close()
            assert client.recv(1) == b""
    assert wp.port_state(port) is wp.PortState.FREE
    _assert_server_can_serve(port)


def _ipv6_available() -> bool:
    try:
        socket.socket(socket.AF_INET6, socket.SOCK_STREAM).close()
    except OSError:
        return False
    return True


# 他のアプリが port を持つ状態（id, アドレスの族, アドレス, listen するか, 設定, 空きかどうかの期待, 期待を実測で確かめた OS の sys.platform）。
# 設定: reuse = SO_REUSEADDR、exclusive = SO_EXCLUSIVEADDRUSE（Windows にだけある）、v6only = IPV6_V6ONLY=1、dual = IPV6_V6ONLY=0
# 期待は Windows 11 と Linux（WSL2 Ubuntu 24.04）で実測した。ALL は OS を限らず実行する（実測は Windows と Linux だけで、macOS は実測していない）、
# WIN_LINUX は Windows と Linux だけで実行する、WIN は Windows だけで実行する
V4, V6 = socket.AF_INET, socket.AF_INET6
ALL, WIN_LINUX, WIN = None, ("win32", "linux"), ("win32",)
HOLDINGS = [
    ("127-listen", V4, "127.0.0.1", True, "", False, ALL),
    ("127-bind", V4, "127.0.0.1", False, "", False, ALL),
    ("127-2-listen", V4, "127.0.0.2", True, "", True, WIN_LINUX),
    ("127-2-bind", V4, "127.0.0.2", False, "", True, WIN_LINUX),
    ("any-listen", V4, "0.0.0.0", True, "", False, ALL),
    ("any-bind", V4, "0.0.0.0", False, "", False, WIN_LINUX),
    ("any-listen-reuse", V4, "0.0.0.0", True, "reuse", False, ALL),
    ("any-listen-exclusive", V4, "0.0.0.0", True, "exclusive", False, WIN),
    ("v6any-v6only-listen", V6, "::", True, "v6only", True, ALL),
    ("v6any-dual-listen", V6, "::", True, "dual", False, ALL),
    ("v6any-dual-bind", V6, "::", False, "dual", False, WIN_LINUX),
    ("v6loop-listen", V6, "::1", True, "", True, ALL),
]


def _hold(held, family: int, address: str, listen: bool, option: str) -> int:
    """他のアプリの代わりに、address の OS が選んだ番号を持つソケットを held に入れ、その番号を返す。"""
    sock = socket.socket(family, socket.SOCK_STREAM)
    held.append(sock)
    if option == "reuse":
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    elif option == "exclusive":
        sock.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_EXCLUSIVEADDRUSE"), 1)
    elif option in ("v6only", "dual"):
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, int(option == "v6only"))
    sock.bind((address, 0))
    if listen:
        sock.listen()
    return sock.getsockname()[1]


def test_free_port_is_free():
    with _bound() as probe:
        port = probe.getsockname()[1]
    assert wp.port_state(port) is wp.PortState.FREE
    _assert_server_can_serve(port)


@pytest.mark.parametrize("name, family, address, listen, option, expected, platforms", HOLDINGS, ids=[h[0] for h in HOLDINGS])
def test_is_free_by_holding(held, name, family, address, listen, option, expected, platforms):
    if platforms is not None and sys.platform not in platforms:
        pytest.skip(f"この状態の期待は {', '.join(platforms)} でだけ確かめている")
    if family == V6 and not _ipv6_available():
        pytest.skip("IPv6 のソケットを作れない環境")
    port = _hold(held, family, address, listen, option)
    assert (wp.port_state(port) is wp.PortState.FREE) is expected
    if expected:
        _assert_server_can_serve(port)


def _wildcard_pair(held) -> int:
    """port を 0.0.0.0 で listen 中にし、port + 1 が空いている番号を返す。"""
    for _ in range(50):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("0.0.0.0", 0))
        sock.listen()
        port = sock.getsockname()[1]
        if port < wp.MAX_PORT and wp.port_state(port + 1) is wp.PortState.FREE:
            held.append(sock)
            return port
        sock.close()
    pytest.fail("連続した 2 つの空き番号が見つからない")


def test_omitted_skips_port_held_on_wildcard(monkeypatch, held):
    port = _wildcard_pair(held)
    monkeypatch.setattr(wp, "DEFAULT_PORT", port)
    assert wp.resolve_port(None).port == port + 1


def test_specified_port_held_on_wildcard_raises(held):
    port = _hold(held, socket.AF_INET, "0.0.0.0", True, "")
    with pytest.raises(ValueError, match=f"ポート番号 {port} は使用中です"):
        wp.resolve_port(port)


def _deny_port(monkeypatch, denied: int) -> None:
    """denied の番号への bind だけを、EACCES の PermissionError で失敗させる（OS の予約・他のアプリの排他的な使用の代わり）。"""
    real = socket.socket.bind

    def bind(self, address):
        if address[1] == denied:
            raise PermissionError(errno.EACCES, "permission denied")
        return real(self, address)
    monkeypatch.setattr(socket.socket, "bind", bind)


def test_specified_denied_raises_with_cause(monkeypatch):
    _deny_port(monkeypatch, 1)
    with pytest.raises(ValueError) as e:
        wp.resolve_port(1)
    assert e.value.args[0] == ("WebGUI のポート番号 1 は使用できません。他のアプリが排他的に使用しているか、OS が使用を許可していない"
                               "（Windows が予約している範囲など）ためです。--port で別の番号を指定するか、--port を省略してください。")


def test_omitted_skips_denied_default_and_reports_state(monkeypatch, held):
    port = _pair(held)
    held.pop().close()                                   # port は空きに戻し、bind の拒否で使えない番号にする
    monkeypatch.setattr(wp, "DEFAULT_PORT", port)
    _deny_port(monkeypatch, port)
    assert wp.resolve_port(None) == (port + 1, wp.PortState.DENIED)


def test_omitted_skips_busy_default_and_reports_state(monkeypatch, held):
    port = _pair(held)
    monkeypatch.setattr(wp, "DEFAULT_PORT", port)
    assert wp.resolve_port(None) == (port + 1, wp.PortState.IN_USE)


@pytest.mark.parametrize("first, second", [(wp.PortState.DENIED, wp.PortState.IN_USE), (wp.PortState.IN_USE, wp.PortState.DENIED)],
                         ids=["denied-then-in-use", "in-use-then-denied"])
def test_omitted_skipped_is_state_of_default_port(monkeypatch, first, second):
    """skipped は DEFAULT_PORT の状態。その次の番号を飛ばした理由（別の状態）では上書きしない。"""
    states = {wp.DEFAULT_PORT: first, wp.DEFAULT_PORT + 1: second}
    monkeypatch.setattr(wp, "port_state", lambda port: states.get(port, wp.PortState.FREE))
    assert wp.resolve_port(None) == (wp.DEFAULT_PORT + 2, first)


def test_omitted_all_denied_raises(monkeypatch):
    monkeypatch.setattr(wp, "DEFAULT_PORT", 40000)
    monkeypatch.setattr(wp, "PORT_TRIES", 2)
    monkeypatch.setattr(wp, "port_state", lambda port: wp.PortState.DENIED)
    with pytest.raises(ValueError, match="40000 から 40001 までがすべて使用中、または使用できません。他のアプリを終了するか"):
        wp.resolve_port(None)


def _fail_bind(monkeypatch, make_error) -> None:
    """socket の bind を、make_error() の例外で失敗させる。"""
    def fail(self, address):
        raise make_error()
    monkeypatch.setattr(socket.socket, "bind", fail)


# bind が送出する例外。(id, 作り方, 状態)。Windows の実際の拒否（Windows が予約している範囲の番号）は PermissionError で errno 13・winerror 10013、
# 使用中は OSError で errno 10048（errno.EADDRINUSE）・winerror 10048（Windows で確かめた）
ERRORS = [
    ("in-use", lambda: OSError(errno.EADDRINUSE, "address already in use"), wp.PortState.IN_USE),
    ("denied", lambda: PermissionError(errno.EACCES, "permission denied"), wp.PortState.DENIED),
]
WINDOWS_ERRORS = [
    ("win-in-use", lambda: OSError(None, "address already in use", None, 10048), wp.PortState.IN_USE),
    ("win-denied", lambda: OSError(None, "access denied", None, 10013), wp.PortState.DENIED),
]


@pytest.mark.parametrize("make_error, state", [e[1:] for e in ERRORS], ids=[e[0] for e in ERRORS])
def test_port_state_from_bind_error(monkeypatch, make_error, state):
    _fail_bind(monkeypatch, make_error)
    assert wp.port_state(1) is state
    assert wp.port_state(1) is not wp.PortState.FREE


@pytest.mark.skipif(not wp.IS_WINDOWS, reason="winerror を持つ OSError は Windows にだけある")
@pytest.mark.parametrize("make_error, state", [e[1:] for e in WINDOWS_ERRORS], ids=[e[0] for e in WINDOWS_ERRORS])
def test_port_state_from_windows_bind_error(monkeypatch, make_error, state):
    _fail_bind(monkeypatch, make_error)
    assert wp.port_state(1) is state


def test_other_os_error_is_not_in_use(monkeypatch):
    """EADDRINUSE・EACCES 以外の bind の失敗は、使用中・拒否と見なさず、そのまま送出する。"""
    _fail_bind(monkeypatch, lambda: OSError(errno.EADDRNOTAVAIL, "address not available"))
    with pytest.raises(OSError) as e:
        wp.port_state(1)
    assert e.value.errno == errno.EADDRNOTAVAIL


@pytest.mark.skipif(not wp.IS_WINDOWS, reason="IPv6 の確認は Windows にだけある")
def test_without_ipv6_decides_by_remaining_checks(monkeypatch, held):
    busy = _hold(held, socket.AF_INET, "0.0.0.0", True, "")
    with _bound() as probe:
        free = probe.getsockname()[1]
    families = []

    class NoIPv6(socket.socket):
        def __init__(self, family=-1, *args, **kwargs):
            families.append(family)
            if family == socket.AF_INET6:
                raise OSError(errno.EAFNOSUPPORT, "address family not supported")
            super().__init__(family, *args, **kwargs)
    monkeypatch.setattr(socket, "socket", NoIPv6)
    assert wp.port_state(busy) is wp.PortState.IN_USE
    assert socket.AF_INET6 not in families             # 使用中は IPv6 の確認の前に決まる
    assert wp.port_state(free) is wp.PortState.FREE
    assert socket.AF_INET6 in families                 # IPv6 のソケットを作れず、残りの確認で空きと決まった


@pytest.mark.skipif(not wp.IS_WINDOWS, reason="IPv6 の確認は Windows にだけある")
def test_ipv6_probe_other_os_error_is_raised(monkeypatch):
    """IPv6 の確認（C）の bind が EAFNOSUPPORT 以外の OSError で失敗したときは、空きとせず、そのまま送出する（IPv4 の bind は成功する）。"""
    real = socket.socket.bind

    def bind(self, address):
        if self.family == socket.AF_INET6:
            raise OSError(errno.EADDRNOTAVAIL, "address not available")
        return real(self, address)
    monkeypatch.setattr(socket.socket, "bind", bind)
    with _bound() as probe:
        free = probe.getsockname()[1]
    with pytest.raises(OSError) as e:
        wp.port_state(free)
    assert e.value.errno == errno.EADDRNOTAVAIL


def _force_non_windows_connect_check(monkeypatch) -> None:
    """Windows 以外の分岐（bind の確認のあと、HOST への接続を試す）を、どの OS でも通す。bind の確認は成功させる。"""
    monkeypatch.setattr(wp, "IS_WINDOWS", False)
    monkeypatch.setattr(wp, "_bind_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(wp, "CONNECT_TIMEOUT", 0.2)    # 接続が拒否されず待ち時間で終わる OS（Windows）の待ちを 0.2 秒にする


def test_non_windows_branch_connectable_port_is_in_use(monkeypatch, held):
    """bind の確認を通っても、HOST の port への接続が成り立てば使用中。"""
    _force_non_windows_connect_check(monkeypatch)
    sock = _bound()
    held.append(sock)
    sock.listen()
    assert wp.port_state(sock.getsockname()[1]) is wp.PortState.IN_USE


def test_non_windows_branch_refused_port_is_free(monkeypatch):
    """bind の確認を通り、HOST の port への接続が成り立たなければ空き（接続が拒否されない Windows では 0.2 秒の待ち）。"""
    _force_non_windows_connect_check(monkeypatch)
    with _bound() as probe:
        free = probe.getsockname()[1]
    assert wp.port_state(free) is wp.PortState.FREE


@pytest.mark.parametrize("port", [0, -1, 70000])
def test_specified_out_of_range_raises(port):
    with pytest.raises(ValueError, match=f"1 から 65535 の間で指定してください（指定した値: {port}）"):
        wp.resolve_port(port)


def _run_gui(monkeypatch, capsys, tmp_path, port_args):
    """cmd_gui を、サーバの代わりに gui.app.main を記録するものに差し替えて呼び、(main に渡った番号, 標準出力の行) を返す。"""
    from amamizu.gui import app
    passed = {}
    monkeypatch.setattr(app, "main", lambda studies, port, show: passed.update(port=port))
    cli.main(["gui", "--studies", str(tmp_path), "--no-browser", *port_args])
    return passed["port"], capsys.readouterr().out.splitlines()


def test_gui_prints_changed_port_and_url(monkeypatch, capsys, tmp_path, held):
    first = _pair(held)
    monkeypatch.setattr(wp, "DEFAULT_PORT", first)
    port, lines = _run_gui(monkeypatch, capsys, tmp_path, [])
    assert port == first + 1
    assert lines[0] == f"ポート {first} は使用中のため、{first + 1} で起動します。"
    assert f"http://127.0.0.1:{first + 1}/" in lines[1]


def test_gui_prints_only_url_when_first_free(monkeypatch, capsys, tmp_path):
    with _bound() as probe:
        free = probe.getsockname()[1]
    monkeypatch.setattr(wp, "DEFAULT_PORT", free)
    port, lines = _run_gui(monkeypatch, capsys, tmp_path, [])
    assert port == free
    assert len(lines) == 1 and f"http://127.0.0.1:{free}/" in lines[0]


def test_gui_specified_busy_port_exits_with_message(monkeypatch, capsys, tmp_path, held):
    sock = _bound()
    held.append(sock)
    port = sock.getsockname()[1]
    called = []
    from amamizu.gui import app
    monkeypatch.setattr(app, "main", lambda *a, **k: called.append(1))
    with pytest.raises(SystemExit) as e:
        cli.run(["gui", "--studies", str(tmp_path), "--port", str(port)])
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert f"ポート番号 {port} は使用中です。--port で別の番号を指定するか、--port を省略してください。" in err and "Traceback" not in err
    assert not called


def test_gui_prints_denied_default_skip(monkeypatch, capsys, tmp_path, held):
    first = _pair(held)
    held.pop().close()
    monkeypatch.setattr(wp, "DEFAULT_PORT", first)
    _deny_port(monkeypatch, first)
    port, lines = _run_gui(monkeypatch, capsys, tmp_path, [])
    assert port == first + 1
    assert lines[0] == (f"ポート {first} は他のアプリが排他的に使用しているか、OS が使用を許可していない（Windows が予約している範囲など）ため、"
                        f"{first + 1} で起動します。")


def _run_gui_exit(monkeypatch, capsys, tmp_path, port_args) -> str:
    """cli.run(["gui", ...]) が終了コード 1 で終わり、gui.app.main を呼ばないことを確かめて、標準エラーを返す。"""
    called = []
    from amamizu.gui import app
    monkeypatch.setattr(app, "main", lambda *a, **k: called.append(1))
    with pytest.raises(SystemExit) as e:
        cli.run(["gui", "--studies", str(tmp_path), *port_args])
    assert e.value.code == 1
    assert not called
    err = capsys.readouterr().err
    assert "Traceback" not in err
    return err


def test_gui_specified_denied_port_exits_with_message(monkeypatch, capsys, tmp_path):
    _deny_port(monkeypatch, 40000)
    err = _run_gui_exit(monkeypatch, capsys, tmp_path, ["--port", "40000"])
    assert "ポート番号 40000 は使用できません。他のアプリが排他的に使用しているか、OS が使用を許可していない" in err


def test_gui_specified_unverifiable_port_exits_with_message(monkeypatch, capsys, tmp_path):
    error = OSError(errno.EADDRNOTAVAIL, "address not available")
    _fail_bind(monkeypatch, lambda: error)
    err = _run_gui_exit(monkeypatch, capsys, tmp_path, ["--port", "40000"])
    assert f"エラー: WebGUI のポート番号 40000 を確かめられません（OS のエラー: {error}）。--port で別の番号を指定してください。" in err
    assert "ファイルを読み書きできません" not in err


def test_gui_omitted_unverifiable_port_exits_with_message(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(wp, "DEFAULT_PORT", 40000)
    error = OSError(errno.EADDRNOTAVAIL, "address not available")
    _fail_bind(monkeypatch, lambda: error)
    err = _run_gui_exit(monkeypatch, capsys, tmp_path, [])
    assert f"エラー: WebGUI のポート番号 40000 を確かめられません（OS のエラー: {error}）。--port で別の番号を指定してください。" in err
    assert "ファイルを読み書きできません" not in err


def test_gui_out_of_range_port_exits_with_message(monkeypatch, capsys, tmp_path):
    called = []
    from amamizu.gui import app
    monkeypatch.setattr(app, "main", lambda *a, **k: called.append(1))
    with pytest.raises(SystemExit) as e:
        cli.run(["gui", "--studies", str(tmp_path), "--port", "70000"])
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "1 から 65535 の間で指定してください（指定した値: 70000）" in err and "Traceback" not in err
    assert not called


def test_port_taken_during_startup_exits_with_message(tmp_path):
    """番号を確かめたあと、起動する前に他のアプリが同じ番号を使い始めると、Traceback を表示せずに理由の文を表示して終了コード 1 で終わる。"""
    studies = Path(__file__).parents[1] / "studies"
    with _bound() as probe:
        port = probe.getsockname()[1]
    script = tmp_path / "run.py"
    script.write_text(
        "import socket\n"
        "from nicegui import app\n"
        "taker = socket.socket()\n"
        f"app.on_startup(lambda: (taker.bind(('127.0.0.1', {port})), taker.listen()))\n"
        "from amamizu.cli import run\n"
        f"run(['gui', '--studies', r'{studies}', '--port', '{port}', '--no-browser'])\n", encoding="utf-8")
    r = subprocess.run([sys.executable, str(script)], capture_output=True, timeout=90,
                       env={**{k: v for k, v in os.environ.items() if not k.startswith(("NICEGUI", "PYTEST"))},
                            "PYTHONIOENCODING": "utf-8"})
    out, err = r.stdout.decode("utf-8"), r.stderr.decode("utf-8")
    assert r.returncode == 1, err
    assert f"エラー: WebGUI のポート番号 {port} は、起動の途中で他のアプリが使用を始めました。" in err
    assert "Traceback" not in err


def _main_with_exit(monkeypatch, port: int, code: int):
    """ui.run が SystemExit(code) を送出するものに差し替えて main を呼ぶ。"""
    from amamizu.gui import app

    def fail(*args, **kwargs):
        raise SystemExit(code)
    monkeypatch.setattr(app.ui, "run", fail)
    app.main(Path("."), port, False)


def test_main_startup_failure_with_free_port_raises_message(monkeypatch):
    """起動の失敗で、番号が空いているときは、起動に失敗したことを知らせる文にする。"""
    from uvicorn.config import STARTUP_FAILURE
    with _bound() as probe:
        free = probe.getsockname()[1]
    with pytest.raises(ValueError, match="WebGUI の起動に失敗しました"):
        _main_with_exit(monkeypatch, free, STARTUP_FAILURE)


def test_main_startup_failure_with_busy_port_raises_conflict_message(monkeypatch, held):
    from uvicorn.config import STARTUP_FAILURE
    sock = _bound()
    held.append(sock)
    port = sock.getsockname()[1]
    with pytest.raises(ValueError, match=f"ポート番号 {port} は、起動の途中で他のアプリが使用を始めました"):
        _main_with_exit(monkeypatch, port, STARTUP_FAILURE)


def test_main_startup_failure_with_denied_port_raises_denied_message(monkeypatch):
    from uvicorn.config import STARTUP_FAILURE
    _deny_port(monkeypatch, 40000)
    with pytest.raises(ValueError, match="ポート番号 40000 は使用できません。他のアプリが排他的に使用しているか"):
        _main_with_exit(monkeypatch, 40000, STARTUP_FAILURE)


def test_main_startup_failure_with_unverifiable_port_raises_check_message(monkeypatch):
    from uvicorn.config import STARTUP_FAILURE
    _fail_bind(monkeypatch, lambda: OSError(errno.EADDRNOTAVAIL, "address not available"))
    with pytest.raises(ValueError, match="ポート番号 40000 を確かめられません（OS のエラー: .*address not available）。--port で別の番号を指定してください"):
        _main_with_exit(monkeypatch, 40000, STARTUP_FAILURE)


def test_main_other_exit_code_is_reraised(monkeypatch, held):
    sock = _bound()
    held.append(sock)
    with pytest.raises(SystemExit) as e:
        _main_with_exit(monkeypatch, sock.getsockname()[1], 1)
    assert e.value.code == 1


@pytest.mark.skipif(os.name != "nt", reason="WebGUI のイベントループを SelectorEventLoop にする設定は Windows だけ")
def test_gui_stops_after_a_connection_shutdown_failed(tmp_path):
    """サーバ側の接続の shutdown を、呼ばれたら 1 回だけ ConnectionResetError（WinError 10054）で失敗させる。
    SelectorEventLoop は shutdown を呼ばないので、実際には失敗は起きない。サーバは応答のあと接続を閉じ、Ctrl+C（SIGINT）で WebGUI が終わる。"""
    studies = Path(__file__).parents[1] / "studies"
    with _bound() as probe:
        port = probe.getsockname()[1]
    script, stop = tmp_path / "run.py", tmp_path / "stop"
    script.write_text(
        "import os, signal, socket, threading, time\n"
        "real, left = socket.socket.shutdown, [1]\n"
        "def shutdown(self, how):\n"
        f"    if left and self.getsockname()[1] == {port}:\n"          # サーバ側の接続の shutdown が呼ばれたら、1 回だけ失敗させる
        "        left.pop()\n"
        "        raise ConnectionResetError(10054, 'reset by peer')\n"
        "    return real(self, how)\n"
        "socket.socket.shutdown = shutdown\n"
        "def stop():\n"                                                 # 親が合図のファイルを置いたら、Ctrl+C と同じ SIGINT を上げる
        f"    while not os.path.exists(r'{stop}'):\n"
        "        time.sleep(0.1)\n"
        "    signal.raise_signal(signal.SIGINT)\n"
        "threading.Thread(target=stop, daemon=True).start()\n"
        "from amamizu.cli import main\n"
        f"main(['gui', '--studies', r'{studies}', '--port', '{port}', '--no-browser'])\n", encoding="utf-8")
    p = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env={**{k: v for k, v in os.environ.items() if not k.startswith(("NICEGUI", "PYTEST"))},
                              "PYTHONIOENCODING": "utf-8"})
    try:
        for _ in range(150):                                            # 起動を待つ
            try:
                client = socket.create_connection((wp.HOST, port), timeout=0.5)
                break
            except OSError:
                assert p.poll() is None, "WebGUI が起動せずに終わった"
                time.sleep(0.2)
        else:
            pytest.fail("WebGUI が起動しない")
        with client:
            client.settimeout(20)
            client.sendall(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            assert client.recv(65536).startswith(b"HTTP/1.1 200")
            try:
                while client.recv(65536):                               # 応答のあと、サーバが接続を閉じる
                    pass
                closed = True
            except TimeoutError:
                closed = False
        stop.touch()
        try:
            out, err = (t.decode("utf-8", "replace") for t in p.communicate(timeout=20))
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = (t.decode("utf-8", "replace") for t in p.communicate())
            pytest.fail(f"Ctrl+C（SIGINT）のあと 20 秒待っても WebGUI が終わらない（接続を閉じた: {closed}）\n{out}\n{err}")
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
    assert closed, "サーバが応答のあと接続を閉じない"
    assert p.returncode == 0, err
    assert "終了しました。" in out and "Traceback" not in err
