"""게이트 공통. 판정은 여기서, 주장은 어디에서도 하지 않는다.

⛔ 모든 게이트는 exit 0(통과) / 1(실패) / 3(측정 불가)로 끝난다.
   3 을 0 으로 뭉개지 마라 — "못 쟀다"와 "통과했다"는 다른 사건이다.
"""
from __future__ import annotations
import gzip, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUDGETS = json.loads((ROOT / "gates" / "budgets.json").read_text(encoding="utf-8"))

PASS, FAIL, UNMEASURED = 0, 1, 3


def gzip_bytes(p: pathlib.Path) -> int:
    """전송 바이트 = gzip 후 크기. 디스크 크기가 아니라 사용자가 실제로 받는 양."""
    return len(gzip.compress(p.read_bytes(), 9))


def walk_files(target: pathlib.Path):
    if target.is_dir():
        yield from (f for f in sorted(target.rglob("*")) if f.is_file())
    elif target.is_file():
        yield target


def run(cmd: list[str], cwd: pathlib.Path | None = None, timeout: int = 900):
    return subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True, timeout=timeout)


def report(name: str, code: int, lines: list[str]) -> int:
    mark = {PASS: "✅ PASS", FAIL: "❌ FAIL", UNMEASURED: "⬜ UNMEASURED"}[code]
    print(f"{mark}  {name}")
    for ln in lines:
        print(f"    {ln}")
    return code


def main_guard(fn):
    try:
        sys.exit(fn())
    except FileNotFoundError as e:
        print(f"⬜ UNMEASURED  전제 파일 없음: {e}")
        sys.exit(UNMEASURED)
