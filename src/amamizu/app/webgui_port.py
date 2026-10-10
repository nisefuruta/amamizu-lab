"""WebGUI のポート番号の決め方（amamizu gui が使う）。確かめるアドレスと WebGUI が bind するアドレスは HOST で 1 か所にそろえる。"""
import errno
import socket
import sys
from enum import Enum
from typing import NamedTuple

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MIN_PORT = 1
MAX_PORT = 65535
PORT_TRIES = 20            # --port を省略したとき、DEFAULT_PORT から確かめる番号の数
IS_WINDOWS = sys.platform == "win32"      # Windows とそれ以外で、ソケットの設定とイベントループを分ける（1 か所）
CONNECT_TIMEOUT = 0.5      # Windows 以外で、HOST の port への接続を試す待ち時間（秒）


class PortState(Enum):
    """番号の状態。FREE は WebGUI が使える、IN_USE は他のアプリが使用中、DENIED は bind が EACCES で拒否される（他のアプリの排他的な使用、
    Windows が予約している範囲（Hyper-V・WSL など）、Linux の 1024 未満）。DENIED はアプリを終了しても解消するとは限らない。"""
    FREE = "free"
    IN_USE = "in_use"
    DENIED = "denied"


# bind の失敗の errno。Windows は OSError.errno が EADDRINUSE=10048、アクセス拒否（winerror 10013）は EACCES=13、IPv6 なしは EAFNOSUPPORT=10047。
# これ以外の OSError は、そのまま送出する
_NO_AF = {errno.EAFNOSUPPORT}

DENIED_CAUSE = "他のアプリが排他的に使用しているか、OS が使用を許可していない（Windows が予約している範囲など）"


def _bind_failure(family: int, address: str, port: int, *, reuse: bool = False, dual: bool = False) -> PortState | None:
    """address の port に bind できれば None（listen はしない）、EADDRINUSE なら IN_USE、EACCES なら DENIED。ほかの OSError は、そのまま送出する。"""
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        if reuse:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if dual:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        try:
            sock.bind((address, port))
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                return PortState.IN_USE
            if e.errno == errno.EACCES:
                return PortState.DENIED
            raise
    return None


def _connects(port: int) -> bool:
    """HOST の port への接続が成り立てば True。拒否・待ち時間の超過は False、そのほかの OSError は、そのまま送出する。"""
    try:
        socket.create_connection((HOST, port), timeout=CONNECT_TIMEOUT).close()
    except (ConnectionRefusedError, TimeoutError):
        return False
    return True


def port_state(port: int) -> PortState:
    """HOST の port の状態。bind が拒否されたときの状態（IN_USE か DENIED）を最初の失敗から返す。
    A: HOST の port に bind できる。WebGUI のサーバ（asyncio）と同じ設定で、Windows は設定なし、それ以外は SO_REUSEADDR
       （終了した直後の接続が残っている番号を、サーバは使える。Windows で設定すると、使用中の番号にも bind できる）。
    Windows は、HOST への接続を受ける他のアプリを、bind で確かめる。
    B: 0.0.0.0 の port に設定なしで bind できる。
    C: [::] の port に IPV6_V6ONLY=0（デュアルスタック）で bind できる。IPv6 のソケットを作れない環境は確かめない。
    Windows 以外は A のあと、HOST の port へ接続できたら IN_USE。
       Linux（WSL2 Ubuntu 24.04）で実測すると、listen 中の持ち主はすべて A で失敗するので、この接続の確認は結果を変えない。
       SO_REUSEADDR 付きの 127.0.0.1 への bind が、他のアプリが全アドレスで listen していても成功する系統（BSD 系、macOS）のためにある（macOS は実測していない）。
    bind の失敗が EADDRINUSE・EACCES でなければ、OSError をそのまま送出する。"""
    failure = _bind_failure(socket.AF_INET, HOST, port, reuse=not IS_WINDOWS)
    if failure is not None:
        return failure
    if not IS_WINDOWS:
        return PortState.IN_USE if _connects(port) else PortState.FREE
    failure = _bind_failure(socket.AF_INET, "0.0.0.0", port)
    if failure is not None:
        return failure
    try:
        failure = _bind_failure(socket.AF_INET6, "::", port, dual=True)
    except OSError as e:
        if e.errno in _NO_AF:
            return PortState.FREE
        raise
    return PortState.FREE if failure is None else failure


def checked_state(port: int) -> PortState:
    """port_state。OS のエラーで確かめられないときは、番号と OS のエラーを示す ValueError にする。"""
    try:
        return port_state(port)
    except OSError as e:
        raise ValueError(f"WebGUI のポート番号 {port} を確かめられません（OS のエラー: {e}）。--port で別の番号を指定してください。") from None


def unusable_message(port: int, state: PortState) -> str:
    """--port で指定した番号が使えないときの文。state は IN_USE か DENIED。"""
    if state is PortState.IN_USE:
        return f"WebGUI のポート番号 {port} は使用中です。--port で別の番号を指定するか、--port を省略してください。"
    return f"WebGUI のポート番号 {port} は使用できません。{DENIED_CAUSE}ためです。--port で別の番号を指定するか、--port を省略してください。"


class Resolved(NamedTuple):
    port: int
    skipped: PortState | None      # --port を省略して DEFAULT_PORT を使えなかったときの、DEFAULT_PORT の状態（それ以外は None）


def resolve_port(requested: int | None) -> Resolved:
    """起動するポート番号を返す。指定があればその番号、省略すれば DEFAULT_PORT から順に確かめて最初に空いていた番号。
    cmd_gui が DEFAULT_PORT を飛ばした理由（使用中か拒否か）を、番号を決めたときの確認の結果で表示できるように、その状態も返す。
    使えない・確かめられないときは ValueError。"""
    if requested is not None:
        if not MIN_PORT <= requested <= MAX_PORT:
            raise ValueError(f"WebGUI のポート番号は {MIN_PORT} から {MAX_PORT} の間で指定してください（指定した値: {requested}）。")
        state = checked_state(requested)
        if state is not PortState.FREE:
            raise ValueError(unusable_message(requested, state))
        return Resolved(requested, None)
    last = DEFAULT_PORT + PORT_TRIES - 1
    skipped = None
    for port in range(DEFAULT_PORT, last + 1):
        state = checked_state(port)
        if state is PortState.FREE:
            return Resolved(port, skipped)
        if port == DEFAULT_PORT:
            skipped = state
    raise ValueError(f"WebGUI のポート番号 {DEFAULT_PORT} から {last} までがすべて使用中、または使用できません。"
                     "他のアプリを終了するか、--port で別の番号を指定してください。")
