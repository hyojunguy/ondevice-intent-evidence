#!/usr/bin/env python3
"""L3 — 언팩 후 기기에 상주(RAM/mmap)해야 하는 바이트가 상한 이하인가.

⛔ size_gate(L1, gzip=전송량)와 다른 축이다. 여기서 재는 건 기기가 실제로 들고 있거나
   mmap 해야 하는 raw 바이트다. 왜 생겼는지·상한 근거는 gates/budgets.json 의 `ram._comment`.

지금은 `artifacts/embedding.bin` 하나만 잰다 — 크레이트에서 `resident_bytes()` 를 노출하는
유일한 아티팩트라서다(crates/sdk-core/src/format.rs). taxonomy.bin·affinity.bin·catalog.bin·
vocab.txt 는 파싱 후 Vec<String>/구조체로 부풀어(라벨·이름·해시맵) 파일 바이트가 상주 바이트의
대리값이 못 된다 — 이 게이트가 UNMEASURED 로 떨어뜨리는 대신 아예 목록에서 뺀 이유가 그것이다.

이 파일의 `_resident_bytes()`는 crates/sdk-core/src/format.rs 의
`EmbeddingTable::from_bytes` + `resident_bytes()`(= table.len() + scales.len()*4)를
Python 으로 미러링한다. ⛔ 그쪽 포맷이 바뀌면 여기도 같이 고쳐야 한다 — 어긋나면
파일 실제 길이 대조에서 UNMEASURED 로 떨어진다(추측해서 계산하지 않는다).
"""
from __future__ import annotations
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from _common import BUDGETS, ROOT, PASS, FAIL, UNMEASURED, report, main_guard, run

MAGIC = b"PLAIDEMB"
UEM_MAGIC = b"PLAIDUEM"
HEADER_V1 = 8 + 4 + 4 + 4 + 4  # magic + version + vocab + dim + scale
HEADER_V2 = 8 + 4 + 4 + 4 + 4  # magic + version + vocab + dim + bits
# PLAIDUEM v1: magic + version + dim + hist_len + n_l1
UEM_HEADER = 8 + 4 + 4 + 4 + 4

# ── 실측 RSS (crates/sdk-core/examples/rss_probe.rs) ──────────────────────
# 위 _resident_bytes() 는 계산된 **상한**이다. 여기부터는 실제 프로세스가 OS 로부터
# 받은 RSS(peak, /usr/bin/time)를 별도로 재서 **나란히** 보고한다 — 대체하지 않는다.
# ⛔ 이건 macOS 실측이다. 저사양 안드로이드 실기기 RSS 는 여전히 미측정이다.
PROBE_BIN = ROOT / "target" / "release" / "examples" / "rss_probe"
_RSS_STAGES: list[tuple[str, str]] = [
    ("baseline", "baseline (아무 아티팩트도 로드하지 않음)"),
    ("embedding", "+embedding.bin (누적 스택 시작)"),
    ("taxonomy", "+taxonomy.bin (embedding 위에 누적)"),
    ("catalog", "+catalog.bin (taxonomy 위에 누적)"),
    ("full", "+vocab.txt · validate_pair (전 스택)"),
    ("taxonomy_only", "taxonomy.bin 단독 (embedding 없이, 고립 측정)"),
    ("catalog_only", "catalog.bin 단독 (고립 측정)"),
    ("vocab_only", "vocab.txt 단독 (고립 측정)"),
]
_MAXRSS_MACOS = re.compile(r"(\d+)\s+maximum resident set size")
_MAXRSS_LINUX = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def _measure_stage_rss(stage: str) -> int | None:
    """그 stage 까지 로드한 프로세스의 peak RSS(바이트) — 못 재면 None.

    macOS(`time -l`)·Linux(GNU `time -v`) 둘 다 시도한다. 실패는 삼키지 않고
    호출부가 UNMEASURED 로 표면화하게 None 을 돌려준다."""
    for flag, pattern, kb_to_bytes in (("-l", _MAXRSS_MACOS, False), ("-v", _MAXRSS_LINUX, True)):
        try:
            proc = run(["/usr/bin/time", flag, str(PROBE_BIN), stage], timeout=30)
        except (FileNotFoundError, OSError):
            return None
        m = pattern.search(proc.stderr or "")
        if m:
            return int(m.group(1)) * (1024 if kb_to_bytes else 1)
    return None


def _measure_real_rss() -> tuple[list[str], dict[str, int] | None]:
    """실측 RSS 섹션 rows 를 만든다. 프로브 빌드/실행에 실패하면 그 사실을 rows 에
    명시하고 (None) 을 돌려준다 — 조용히 생략하지 않는다."""
    build = run(
        ["cargo", "build", "--release", "--example", "rss_probe", "-p", "oicr-sdk-core"],
        timeout=300,
    )
    if build.returncode != 0 or not PROBE_BIN.exists():
        return (
            [
                "⚠️ 실측 RSS: rss_probe 빌드 실패 — 산술값만 보고한다 (실측 없음).",
                f"   cargo 종료코드 {build.returncode}: {(build.stderr or '').strip()[-300:]}",
            ],
            None,
        )

    rss: dict[str, int] = {}
    for stage, _label in _RSS_STAGES:
        v = _measure_stage_rss(stage)
        if v is None:
            return (
                [
                    "⚠️ 실측 RSS: /usr/bin/time 이 없거나 출력 형식을 못 읽었다 "
                    f"(stage={stage}) — 산술값만 보고한다 (실측 없음)."
                ],
                None,
            )
        rss[stage] = v

    base = rss["baseline"]
    rows = ["── 실측 RSS (프로세스 peak, /usr/bin/time — macOS, 안드로이드 실기기 아님) ──"]
    for stage, label in _RSS_STAGES:
        delta = rss[stage] - base
        rows.append(f"{rss[stage]:>10,}  {label}  (Δbaseline={delta:,}B)")
    return rows, rss


# ── 실측 RSS — 저사양 Android 에뮬레이터 (2026-08-28 추가) ────────────────────
# 위 macOS 실측은 이 게이트가 답 못 하던 질문("저사양 Android 실기기에서는?")을 그대로
# 남긴다. 실기기가 없으므로 에뮬레이터(API 30, RAM 1536MB, 2코어 — `rss_lowspec` AVD)로
# 근사한다. ⛔ 에뮬레이터는 실기기가 아니다 — 결과 라벨에 항상 명시한다.
#
# `crates/sdk-core/examples/rss_probe.rs` 를 그대로 크로스컴파일하지 않는 이유: 그 파일의
# `artifacts_dir()` 이 `env!("CARGO_MANIFEST_DIR")` 를 **컴파일 시점**에 굽는다 — 이 macOS
# 경로(`/home/user/oicr-sdk/crates/sdk-core`)를 에뮬레이터에 그대로 재현하려면
# `/` 를 쓰기 가능하게 remount 해야 하는데, dm-verity 파티션이라 `adb remount` 가 거부한다
# (`Read-only file system`, 실측). 그래서 여기서는 **별도의 작은 프로브**를 만든다 — 아티팩트
# 디렉터리를 argv 로 받고(baked path 없음), peak RSS 는 외부 `/usr/bin/time` 대신 Android/Linux
# 가 프로세스 안에서 직접 노출하는 `/proc/self/status` 의 `VmHWM`(peak resident set) 으로 잰다.
# `crates/**` 는 건드리지 않는다 — sdk-core 의 공개 API(`EmbeddingTable::from_bytes`)만 쓴다.
_ANDROID_PROBE_MAIN = r"""
use oicr_sdk_core::EmbeddingTable;
use std::io::Read as _;

fn vmhwm_kb() -> u64 {
    let mut s = String::new();
    if let Ok(mut f) = std::fs::File::open("/proc/self/status") {
        let _ = f.read_to_string(&mut s);
    }
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            if let Some(tok) = rest.split_whitespace().next() {
                return tok.parse().unwrap_or(0);
            }
        }
    }
    0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let stage = args.get(1).map(String::as_str).unwrap_or("baseline");
    let dir = args.get(2).cloned().unwrap_or_else(|| ".".to_string());
    if stage == "embedding" {
        let data = std::fs::read(format!("{}/embedding.bin", dir)).expect("read embedding.bin");
        let table = EmbeddingTable::from_bytes(&data).expect("parse embedding.bin");
        std::hint::black_box(table.resident_bytes());
    }
    println!("stage={} vmhwm_kb={}", stage, vmhwm_kb());
}
"""

_ANDROID_PROBE_CARGO = """[package]
name = "android_rss_probe"
version = "0.0.0"
edition = "2024"

[[bin]]
name = "android_rss_probe"
path = "src/main.rs"

[dependencies]
oicr-sdk-core = {{ path = "{core}" }}

[profile.release]
opt-level = 2
strip = true
"""

ANDROID_TARGET = "aarch64-linux-android"
ANDROID_ARTIFACT_REMOTE = "/data/local/tmp/oicr-rss-probe"


def _toolchain() -> str:
    f = ROOT / "rust-toolchain.toml"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("channel"):
                return line.split("=", 1)[1].strip().strip('"\'')
    return "stable"


def _find_adb() -> str | None:
    for c in [
        shutil.which("adb"),
        (os.environ.get("ANDROID_SDK_ROOT", "") + "/platform-tools/adb") or None,
        (os.environ.get("ANDROID_HOME", "") + "/platform-tools/adb") or None,
        "/opt/homebrew/share/android-commandlinetools/platform-tools/adb",
    ]:
        if c and Path(c).exists():
            return c
    return None


def _adb_device_ready(adb: str) -> str | None:
    """연결된 device 직렬번호 하나. 없으면 None(스킵 사유는 호출부가 rows 에 남긴다)."""
    try:
        out = subprocess.run([adb, "devices"], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return None
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2 and parts[1] == "device":
            return parts[0]
    return None


def _measure_android_emulator_rss(rows: list[str]) -> dict | None:
    """rows 에 섹션을 추가하고, 성공하면 {stage: vmhwm_kb} 를 돌려준다. 실패는 예외를
    삼키지 않되 게이트 전체를 막지는 않는다 — rows 에 사유를 남기고 None."""
    embedding = ROOT / "artifacts" / "embedding.bin"
    if not embedding.exists():
        rows.append("⬜ 실측 RSS(Android 에뮬레이터): artifacts/embedding.bin 없음 — 스킵")
        return None

    adb = _find_adb()
    if not adb:
        rows.append("⬜ 실측 RSS(Android 에뮬레이터): adb 없음 — 스킵(NDK/SDK 미설치)")
        return None
    serial = _adb_device_ready(adb)
    if not serial:
        rows.append("⬜ 실측 RSS(Android 에뮬레이터): 연결된 device/emulator 없음 — 스킵")
        return None

    toolchain = _toolchain()
    installed = subprocess.run(
        ["rustup", "target", "list", "--installed", "--toolchain", toolchain],
        capture_output=True, text=True,
    ).stdout.split()
    if ANDROID_TARGET not in installed:
        rows.append(f"⬜ 실측 RSS(Android 에뮬레이터): rustup 타깃 {ANDROID_TARGET} 미설치 — 스킵")
        return None

    try:
        with tempfile.TemporaryDirectory(prefix="oicr-rss-android-") as td:
            d = Path(td) / "android_rss_probe"
            (d / "src").mkdir(parents=True)
            core = ROOT / "crates/sdk-core"
            (d / "Cargo.toml").write_text(_ANDROID_PROBE_CARGO.format(core=core), encoding="utf-8")
            (d / "src/main.rs").write_text(_ANDROID_PROBE_MAIN, encoding="utf-8")
            env = dict(os.environ, CARGO_TARGET_DIR=str(Path(td) / "target"),
                       RUSTUP_TOOLCHAIN=toolchain)
            b = subprocess.run(
                ["cargo", "build", "--release", "--quiet", "--target", ANDROID_TARGET],
                cwd=d, env=env, capture_output=True, text=True, timeout=300,
            )
            if b.returncode != 0:
                rows.append("⬜ 실측 RSS(Android 에뮬레이터): 프로브 빌드 실패 — "
                            f"{b.stderr.strip()[-400:]}")
                return None
            binpath = Path(td) / "target" / ANDROID_TARGET / "release" / "android_rss_probe"

            def _adb(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
                return subprocess.run([adb, "-s", serial, *args], capture_output=True,
                                       text=True, timeout=timeout)

            _adb("shell", "mkdir", "-p", ANDROID_ARTIFACT_REMOTE)
            push_bin = _adb("push", str(binpath), f"{ANDROID_ARTIFACT_REMOTE}/probe")
            push_art = _adb("push", str(embedding), f"{ANDROID_ARTIFACT_REMOTE}/embedding.bin")
            if push_bin.returncode != 0 or push_art.returncode != 0:
                rows.append("⬜ 실측 RSS(Android 에뮬레이터): adb push 실패 — "
                            f"{(push_bin.stderr or push_art.stderr).strip()[-300:]}")
                return None
            _adb("shell", "chmod", "755", f"{ANDROID_ARTIFACT_REMOTE}/probe")

            results: dict[str, int] = {}
            for stage in ("baseline", "embedding"):
                r = _adb("shell", f"{ANDROID_ARTIFACT_REMOTE}/probe", stage,
                          ANDROID_ARTIFACT_REMOTE, timeout=30)
                m = re.search(r"vmhwm_kb=(\d+)", r.stdout or "")
                if not m:
                    rows.append(f"⬜ 실측 RSS(Android 에뮬레이터): stage={stage} 파싱 실패 — "
                                f"stdout={r.stdout!r} stderr={r.stderr!r}")
                    return None
                results[stage] = int(m.group(1)) * 1024  # KiB -> bytes

            _adb("shell", "rm", "-rf", ANDROID_ARTIFACT_REMOTE)
    except subprocess.TimeoutExpired as exc:
        rows.append(f"⬜ 실측 RSS(Android 에뮬레이터): 타임아웃 — {exc}")
        return None
    except Exception as exc:  # noqa: BLE001 — 실측 실패는 UNMEASURED 로, 게이트 크래시 아님
        rows.append(f"⬜ 실측 RSS(Android 에뮬레이터): 실패 — {exc}")
        return None

    base = results["baseline"]
    delta = results["embedding"] - base
    rows += [
        "── 실측 RSS — Android 에뮬레이터 (⚠️ 실기기 아님, VmHWM/proc peak) ──",
        "   AVD: rss_lowspec · API 30 · arm64-v8a · RAM 1536MB · vCPU 2 (저사양 근사)",
        f"{base:>10,}  baseline (프로세스 자체 VmHWM)",
        f"{results['embedding']:>10,}  +embedding.bin  (Δbaseline={delta:,}B)",
    ]
    return results


def _uem_resident_bytes(data: bytes) -> tuple[int, dict] | None:
    """`artifacts/user_embedding.bin`(PLAIDUEM v1) 의 상주 바이트.

    crates/sdk-core/src/user_embedding.rs 의 `UserEmbedding::resident_bytes()`
    (= params * 4)를 미러링한다. **on-disk 는 int8 인데 상주는 f32 다** — 로더가
    역양자화해 들고 있기 때문이고, 그래서 파일 크기(L1)와 상주(L3)가 4배 갈린다.
    그 4배를 안 세면 L3 예산이 우리에게 유리한 방향으로 틀린다.
    """
    if len(data) < UEM_HEADER or data[:8] != UEM_MAGIC:
        return None
    version, dim, hist_len, n_l1 = struct.unpack_from("<IIII", data, 8)
    if version != 1 or dim == 0 or hist_len == 0 or n_l1 > hist_len:
        return None
    params = hist_len * dim + dim * dim
    # 행별 f32 스케일 + int8 가중치 — 길이가 어긋나면 포맷 드리프트이니 추측하지 않는다.
    expected = UEM_HEADER + (hist_len * 4 + hist_len * dim) + (dim * 4 + dim * dim)
    if len(data) != expected:
        return None
    return params * 4, {
        "version": version, "vocab": hist_len, "dim": dim, "bits": 32,
        "scales_bytes": (hist_len + dim) * 4,
        "_note": "상주는 f32(역양자화 후) — on-disk int8 의 4배",
    }


def _resident_bytes(data: bytes) -> tuple[int, dict] | None:
    """(resident_bytes, meta) 를 반환한다. 헤더/버전을 못 알아보거나 실제 파일 길이가
    계산과 어긋나면 None — "모르겠다"를 UNMEASURED 로 정직하게 떨어뜨리기 위해서다.

    magic 으로 아티팩트 종류를 가른다 — 종류마다 상주 산식이 다르다."""
    if len(data) >= 8 and data[:8] == UEM_MAGIC:
        return _uem_resident_bytes(data)
    if len(data) < 8 or data[:8] != MAGIC:
        return None
    version, vocab, dim = struct.unpack_from("<III", data, 8)
    if vocab == 0 or dim == 0:
        return None

    if version == 1:
        # v1: 전역 스케일 f32 + i8[vocab*dim] (bits=8 고정, scales 없음).
        table_bytes = vocab * dim
        expected_len = HEADER_V1 + table_bytes
        if len(data) != expected_len:
            return None  # 포맷 드리프트 — 추측하지 않는다
        resident = table_bytes  # + scales.len()*4 (v1은 0)
        return resident, {"version": 1, "vocab": vocab, "dim": dim, "bits": 8, "scales_bytes": 0}

    if version == 2:
        # v2: bits u32 + f32[vocab](행별 스케일) + u8[vocab*row_bytes](니블 팩킹 가능).
        (bits,) = struct.unpack_from("<I", data, 20)
        if bits not in (4, 8):
            return None
        row_bytes = -(-(dim * bits) // 8)  # ceil div — format.rs row_bytes()와 동일
        scales_bytes = vocab * 4
        table_bytes = vocab * row_bytes
        expected_len = HEADER_V2 + scales_bytes + table_bytes
        if len(data) != expected_len:
            return None
        resident = table_bytes + scales_bytes
        return resident, {
            "version": 2, "vocab": vocab, "dim": dim, "bits": bits,
            "scales_bytes": scales_bytes,
        }

    return None  # 미지 version — 크레이트가 새 버전을 낼 때까지 UNMEASURED


def run_gate() -> int:
    cfg = BUDGETS["ram"]
    limit = cfg["limit_bytes"]
    rows: list[str] = []
    unmeasured: list[str] = []
    total = 0

    for rel in cfg["artifacts"]:
        target = ROOT / rel
        if not target.exists():
            unmeasured.append(f"없는 아티팩트: {rel} → scripts/build_artifacts.py 를 먼저 돌려라")
            continue
        parsed = _resident_bytes(target.read_bytes())
        if parsed is None:
            unmeasured.append(f"{rel} — magic/version/길이 불일치 (포맷 드리프트 의심, 추측 안 함)")
            continue
        resident, meta = parsed
        total += resident
        rows.append(
            f"{resident:>10,}  {rel}  "
            f"(v{meta['version']} · vocab={meta['vocab']:,} · dim={meta['dim']} · "
            f"{meta['bits']}bit · scales={meta['scales_bytes']:,}B)"
        )

    if unmeasured:
        return report("L3 ram", UNMEASURED, rows + unmeasured)

    pct = total / limit * 100
    rows += [
        "─" * 10,
        f"{total:>10,}  **부분** 합계 (언팩 후 상주 상한 — 계산값, 실제 RSS 아님)",
        "",
        "⚠️ 이 합계는 완전하지 않다 — 아래는 RAM 에 상주하는데 여기서 안 센다:",
        "     taxonomy.bin · catalog.bin — 벡터 + 문자열/맵으로 파싱돼 파일 바이트 != 상주",
        "     vocab.txt                  — HashMap 으로 파싱돼 항목당 오버헤드가 붙는다",
        "   아래 실측 RSS 섹션의 *_only 델타가 이 세 아티팩트의 실제 크기를 보여준다 —",
        "   \"512d 면 약 1.5MB 추가\"라는 예전 추정보다 훨씬 크다(taxonomy·catalog 는",
        "   [f32; DIM] 로 즉시 역양자화되므로 on-disk 비트폭과 무관하게 DIM 에 선형 비례한다).",
        f"{limit:>10,}  상한",
        f"{'':>10}  사용률 {pct:.1f}% · 여유 {limit - total:,} 바이트",
        "",
    ]
    rss_rows, rss = _measure_real_rss()
    rows += rss_rows
    if rss is not None:
        emb_delta = rss["embedding"] - rss["baseline"]
        rows += [
            "",
            f"⚠️ 계산값({total:,}B) vs 실측 embedding 델타({emb_delta:,}B) 오차: "
            f"{emb_delta - total:,}B ({(emb_delta / total - 1) * 100:+.1f}%) — from_bytes() 가 "
            "raw 파일 바이트를 복사해 파싱하는 동안 원본+파싱본이 잠깐 동시에 상주하기",
            "   때문이다(peak RSS 는 high-water mark 라 그 순간이 지나도 안 내려간다).",
            "⚠️ 위는 macOS 프로세스 RSS 다. 아래는 저사양 Android **에뮬레이터** 근사치다",
            "   (실기기 아님 — 이 게이트가 완전히 답하지 못하는 질문은 그대로 남는다).",
        ]
    rows.append("")
    android_rss = _measure_android_emulator_rss(rows)
    if android_rss is not None:
        delta = android_rss["embedding"] - android_rss["baseline"]
        rows.append(f"⚠️ 계산값({total:,}B) vs Android 에뮬레이터 embedding 델타({delta:,}B) 오차: "
                    f"{delta - total:,}B ({(delta / total - 1) * 100:+.1f}%)")
    return report("L3 ram", PASS if total <= limit else FAIL, rows)


# ⛔ 최상위에서 부르면 이 모듈을 import 할 수 없다 — size_gate.py 와 동일한 이유로
#    main_guard 는 __main__ 가드 안에서만 부른다.
if __name__ == "__main__":
    main_guard(run_gate)
