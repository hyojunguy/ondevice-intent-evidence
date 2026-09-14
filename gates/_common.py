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


def report(name: str, code: int, lines: list[str], facts: dict | None = None) -> int:
    """게이트 판정을 출력하고, `facts` 가 있으면 **원장으로 남긴다**.

    ⛔ 2026-09-14: 게이트 14개 중 9개가 수치를 출력만 하고 파일에 안 남기고 있었다.
       그래서 논문이 p95 2.404 ms · 페이로드 69,482 B · Δℓ2 1.7321 처럼 **어느 원장에도
       없는 값**을 인용했고, 그 사실을 사람이 읽어서 잡을 방법이 없었다. 게이트가 매 실행
       재측정한다는 것은 "값이 흔들린다"는 뜻이고, 흔들리는 값은 못 박지 않으면 인용할 수
       없다. 이제 판정과 원장 기록이 같은 함수다 — 하나만 하는 것이 불가능해진다.
    """
    mark = {PASS: "✅ PASS", FAIL: "❌ FAIL", UNMEASURED: "⬜ UNMEASURED"}[code]
    print(f"{mark}  {name}")
    for ln in lines:
        print(f"    {ln}")
    if facts is not None:
        import datetime as _dt, json as _json, re as _re
        d = ROOT / "artifacts" / "gate-facts"
        d.mkdir(parents=True, exist_ok=True)
        slug = _re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        (d / f"{slug}.json").write_text(_json.dumps(
            {"_measured_at": _dt.datetime.now().isoformat(timespec="seconds"),
             "_gate": name,
             "_what": "이 게이트가 실제로 잰 값. 논문은 이 파일을 인용한다.",
             "verdict": mark.split()[-1], **facts},
            ensure_ascii=False, indent=2), encoding="utf-8")
    return code


def main_guard(fn):
    try:
        sys.exit(fn())
    except FileNotFoundError as e:
        print(f"⬜ UNMEASURED  전제 파일 없음: {e}")
        sys.exit(UNMEASURED)
