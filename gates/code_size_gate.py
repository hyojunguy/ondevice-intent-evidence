#!/usr/bin/env python3
"""L2 — 라이브러리 코드가 앱에 더하는 바이트를 잰다.

⛔ 왜 있나 (2026-08-28): `gates/budgets.json` 은 `size.measure: "gzip"` 으로 **다운로드
   데이터**만 세고 라이브러리 코드 항목이 아예 없었다. 그런데 매체사가 보는 숫자는
   APK/IPA 증가분 = 코드 + 데이터다. 즉 우리는 "SDK 5MB 미만"을 팔면서 고객이 재는
   것을 재고 있지 않았다.

   같은 혼선이 리서치에도 있었다 — telemetry-input.md 가 Microsoft Clarity 의
   Android +400KB / iOS +900KB 를 우리 예산의 "하한선"이라 적었는데, 그건 코드
   다운로드 크기고 우리 5 MiB 는 데이터 gzip 이다. 단위가 다른 둘을 더하고 있었다.

측정 방법 — **링커가 실제로 남긴 바이트**를 잰다:
   probe-empty : fn main() {}
   probe-core  : 같은 main 에서 sdk-core 공개 API 를 실제로 호출(black_box 로 DCE 방지)
   두 실행파일을 같은 release 프로필로 빌드·strip 한 뒤 **차분**한다.
   .a(staticlib) 크기를 쓰지 않는 이유: 링커가 안 쓰는 오브젝트를 버리므로 .a 는
   항상 과대평가다. 우리가 알고 싶은 건 "앱이 실제로 커지는 양"이다.

⚠️ 이 값은 **arm64 네이티브 코드**이지 APK/IPA 최종 증가분이 아니다. 실제 배포는 그 위에
   FFI 바인딩(JNI/Swift)과 각 플랫폼 런타임이 얹힌다. 그 몫은 별도로 재야 하고, 재기 전에는
   여기 숫자를 "SDK 코드 크기"라고 인용하지 마라 — 하한선이다.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "crates/sdk-core"

FIXTURE = "\n".join(["[UNK]", "[CLS]", "가", "##나"])

EMPTY_MAIN = "fn main() { std::hint::black_box(0u8); }\n"

# sdk-core 의 실사용 경로를 태운다. 한 줄만 부르면 링커가 나머지를 버려 과소평가된다.
CORE_MAIN = r"""
use oicr_sdk_core as core;
use std::hint::black_box;

fn main() {
    let vocab_txt: &str = black_box(include_str!("mini_vocab.txt"));
    let bytes: &[u8] = black_box(&[]);

    // 파싱 + 토크나이즈 + 임베딩 경로
    let v = core::Vocab::from_str(vocab_txt);
    black_box(v.len());
    black_box(core::tokenize_surface(black_box("강아지 사료 알러지 없는 걸로"), &v).len());
    if let Ok(t) = core::EmbeddingTable::from_bytes(bytes) {
        black_box(core::validate_pair(&v, &t).is_ok());
        black_box(core::embed(black_box("텍스트"), &v, &t));
    }

    // 매칭 / 분류 / 투영 경로
    if let Ok(tx) = core::Taxonomy::from_bytes(bytes) {
        let q = [0f32; core::DIM];
        black_box(tx.scores(&q).len());
        black_box(tx.lexical_overlap(&core::Taxonomy::query_terms(black_box("사료"))).len());
    }
    if let Ok(c) = core::Catalog::from_bytes(bytes) { black_box(c.dim()); }
    if let Ok(h) = core::IntentHead::from_bytes(bytes) {
        black_box(h.classify(&[0f32; core::DIM], 3).len());
    }
    if let Ok(p) = core::Projection::from_bytes(bytes) { black_box(p.apply(&[0f32; core::DIM])); }

    // 이벤트 입력 계층 (배포되므로 L2 에 센다)
    if let Some(mut log) = core::EventLog::new(
        core::ServiceConsent::Granted, core::BehavioralConsent::Granted) {
        log.push(core::Event::new(black_box(0), core::EventKind::Tap, 1,
                                  core::dwell_bucket(black_box(1500)), 7));
        log.push(core::Event::new(black_box(900), core::EventKind::Scroll, 1,
                                  core::scroll_bucket(black_box(60)), 9));
        black_box(log.kind_counts());
        let mut hints = [0f32; 32];
        black_box(log.weighted_hints(black_box(1000), 20_000.0, &mut hints).is_ok());
        black_box(core::recency_bucket(black_box(800)));
        black_box(core::time_decay(black_box(500), 1000.0));
    }

    // 지각 1층 — 콘텐츠 스케치 (2026-08-31)
    // ⛔ 이 블록이 없으면 링커가 콘텐츠·유저항 코드를 통째로 잘라내고, 게이트는
    //    "크기 변화 없음"을 **재지도 않고** 보고한다. 그건 우리에게 유리한 방향의
    //    누락이고, 크기 게이트가 taxonomy 206KB 를 빠뜨렸던 것과 같은 사고다.
    if let Some(mut sk) = core::ContentSketch::new(
        core::BehavioralConsent::Granted, core::ContentConsent::Granted) {
        sk.fold_written(&[0f32; core::DIM]);
        sk.fold_viewed(&[0f32; core::DIM]);
        black_box(sk.folds());
        black_box(core::UserEmbedding::content_from_sketch(&sk).is_some());
    }
    // 지각 2.6층 — 배포된 유저 임베딩 W 의 추론 경로
    if let Ok(ue) = core::UserEmbedding::from_bytes(bytes) {
        black_box(ue.params());
        black_box(ue.resident_bytes());
        let h = ue.hist_from(black_box(&[0f32; 4]), black_box(&[0f32; 4]));
        black_box(ue.hist_vector(&h));
        black_box(ue.content_vector(&[0f32; core::DIM]));
        if let Ok(tx) = core::Taxonomy::from_bytes(bytes) {
            // ⛔ 지각(2·3층) 인자는 `None` 이다 — **제품이 그렇게 부르기 때문**이고
            //    (`TIER1_RANKING_CHANNEL = false`), 프로브는 배포되는 경로만 잰다.
            //    채널을 여는 커밋은 여기도 같이 채워야 크기가 실제로 재어진다
            //    (링커는 안 부르는 코드를 잘라낸다 — 2026-08-31 사고).
            let t = ue.terms(&tx, &h, None, None);
            let mut sc = vec![0f32; tx.len()];
            black_box(t.add_to(&mut sc, core::USER_TERM_ALPHA));
        }
    }

    // 개인화 상태 경로
    let mut mem = core::InterestMemory::new(black_box(8));
    black_box(&mut mem);
    let mut prior = core::DomainPrior::new(black_box(28));
    black_box(&mut prior);
}
"""

CARGO = """[package]
name = "{name}"
version = "0.0.0"
edition = "2024"

[[bin]]
name = "{name}"
path = "src/main.rs"

{deps}

[profile.release]
opt-level = "z"
lto = true
codegen-units = 1
panic = "abort"
strip = true
"""

# ⛔ 대리 측정(PROXY) 전용 — NDK 없는 타깃(android)에서 "링크된 바이트"를 잴 수 없을 때만 쓴다.
# staticlib 는 시스템 링커(ld) 없이도 빌드된다(archiving 은 rustc 번들 llvm-ar). 단 이 문서
# 상단 주석이 이미 경고하듯 .a 는 링커의 dead-code-elim(--gc-sections)을 거치지 않는다 —
# darwin/ios 실측(링크됨)과 대조해 오차가 얼마나 큰지 반드시 같이 보고한다(run_gate 참조).
CARGO_STATICLIB = """[package]
name = "{name}"
version = "0.0.0"
edition = "2024"

[lib]
crate-type = ["staticlib"]
path = "src/lib.rs"

{deps}

[profile.release]
opt-level = "z"
lto = true
codegen-units = 1
panic = "abort"
strip = true
"""


def _toolchain() -> str:
    """레포가 고정한 채널. 프로브가 다른 컴파일러로 빌드되면 측정이 무의미하다."""
    f = ROOT / "rust-toolchain.toml"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("channel"):
                return line.split("=", 1)[1].strip().strip('"\'')
    return "stable"


# ── 타깃별 wall-clock 캡 ───────────────────────────────────────────────────
#
# ⛔ 왜 있나 (2026-08-31): 이 게이트에 캡이 없어서 `run_all.py` 가 **1시간 5분** 매달렸다.
#    걸린 자리는 Android NDK 링커 래퍼(arm64 맥에서 x86_64 바이너리 → Rosetta)이고,
#    그 내내 cargo·rustc·env 세 프로세스가 **CPU 0.0%** 였다. 증상이 "느리다"로만 보여서
#    사람이 원인을 못 짚는다 — 판정은 경과시간이 아니라 `ps -o %cpu=` 였다.
#
# 캡의 계약: **초과는 그 타깃만 UNMEASURED 로 강등**한다. 스위트를 세우지 않는다 —
# 측정 못 한 것과 예산 초과는 다른 사건이고, 전자로 후자를 주장할 수 없다.
#
# 값의 근거: 이 머신 실측으로 darwin 한 타깃(실측 2빌드 + 대리 2빌드)이 **약 11초**다.
# 180 은 그 16배 — 콜드 캐시·느린 머신을 넉넉히 덮으면서, 걸린 타깃 하나가 태우는 시간을
# 3분으로 묶는다. ⛔ 관측 최댓값(=행 1시간 5분)으로 잡지 마라. 그러면 캡이 영원히 안 걸린다.
BUILD_TIMEOUT_S = int(os.environ.get("CODE_SIZE_BUILD_TIMEOUT_S", "180"))


class BuildTimeout(RuntimeError):
    """캡 초과. ⛔ 빌드 **실패**(SystemExit)와 다른 사건이라 타입을 나눈다 —
    실패는 코드가 안 되는 것이고, 이건 우리가 안 기다리기로 한 것이다."""


def _cargo(cmd: list[str], cwd: Path, env: dict[str, str], label: str):
    """cargo 를 **자기 세션**으로 띄우고 캡을 건다.

    ⛔ `subprocess.run(timeout=)` 으로는 부족하다. 그건 직계 자식(cargo)만 죽이는데,
       매달린 것은 손자(rustc → 링커 래퍼)이고 그놈이 stdout 파이프를 쥐고 있어
       kill 뒤의 `communicate()` 가 다시 블록한다 — 캡을 걸고도 매달린다.
       `start_new_session=True` + `killpg` 여야 프로세스 트리가 통째로 죽는다
       (이 레포가 긴 로컬 잡에서 이미 배운 패턴이다).
    """
    import signal

    p = subprocess.Popen(  # noqa: S603
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        out, err = p.communicate(timeout=BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            p.kill()
        p.communicate()
        raise BuildTimeout(f"{label}: {BUILD_TIMEOUT_S}s 캡 초과 — 프로세스 그룹 종료") from None
    return p.returncode, out, err


def build(tmp: Path, name: str, main_src: str, with_core: bool, target: str) -> int:
    d = tmp / name
    (d / "src").mkdir(parents=True)
    deps = f'[dependencies]\noicr-sdk-core = {{ path = "{CORE}" }}\n' if with_core else "[dependencies]\n"
    (d / "Cargo.toml").write_text(CARGO.format(name=name, deps=deps), encoding="utf-8")
    (d / "src/main.rs").write_text(main_src, encoding="utf-8")
    if with_core:
        # 아주 작은 vocab 픽스처 — include_bytes! 가 바이너리에 더하는 몫은 양쪽에서 같지
        # 않으므로 크기를 기록해 빼 준다.
        (d / "src/mini_vocab.txt").write_text(FIXTURE, encoding="utf-8")
    # 프로브는 tmp 에 있어서 레포의 rust-toolchain.toml 오버라이드를 못 받는다.
    # 우리가 실제로 배포하는 툴체인과 다른 컴파일러로 재면 그 숫자는 우리 것이 아니다.
    env = dict(os.environ, CARGO_TARGET_DIR=str(tmp / "target"), RUSTUP_TOOLCHAIN=_toolchain())
    rc, _out, err = _cargo(
        ["cargo", "build", "--release", "--quiet", "--target", target],
        d, env, f"{name} / {target}",
    )
    if rc != 0:
        print(err[-2500:], file=sys.stderr)
        raise SystemExit(f"⛔ 빌드 실패: {name} / {target}")
    return (tmp / "target" / target / "release" / name).stat().st_size


PROXY_METHOD = "unlinked-staticlib-object-text+data-diff"


def _llvm_tool(name: str) -> str | None:
    """rustup 번들 llvm-{name} 경로. 없으면 None(호출자가 UNMEASURED 로 처리)."""
    env = dict(os.environ, RUSTUP_TOOLCHAIN=_toolchain())
    r = subprocess.run(["rustc", "--print", "sysroot"], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    sysroot = Path(r.stdout.strip())
    for hit in sysroot.glob(f"lib/rustlib/*/bin/llvm-{name}*"):
        if hit.is_file() and os.access(hit, os.X_OK):
            return str(hit)
    return None


def _proxy_component_bytes(tmp: Path, name: str, main_src: str, with_core: bool, target: str) -> int:
    """staticlib 를 빌드하고, 그 크레이트 **자신의** codegen-unit 오브젝트만 골라
    text+data 바이트를 센다(std/core/compiler_builtins 는 empty/core 양쪽에 결정론적으로
    동일하게 딸려오므로 크레이트 필터링만으로 사실상 상쇄된다 — 굳이 .a 전체를 더하지
    않는다). ⚠️ 링커 DCE 가 없으므로 실제 링크된 값보다 크게 잡힌다(run_gate 의 실측 대조 참조).
    """
    llvm_ar = _llvm_tool("ar")
    llvm_size = _llvm_tool("size")
    if not llvm_ar or not llvm_size:
        raise RuntimeError("llvm-tools-preview 없음 — `rustup component add llvm-tools-preview`")

    d = tmp / name
    (d / "src").mkdir(parents=True)
    deps = f'[dependencies]\noicr-sdk-core = {{ path = "{CORE}" }}\n' if with_core else "[dependencies]\n"
    (d / "Cargo.toml").write_text(CARGO_STATICLIB.format(name=name, deps=deps), encoding="utf-8")
    lib_src = main_src.replace(
        "fn main()", '#[unsafe(no_mangle)]\npub extern "C" fn probe()'
    )
    (d / "src/lib.rs").write_text(lib_src, encoding="utf-8")
    if with_core:
        (d / "src/mini_vocab.txt").write_text(FIXTURE, encoding="utf-8")

    env = dict(os.environ, CARGO_TARGET_DIR=str(tmp / "target-proxy"), RUSTUP_TOOLCHAIN=_toolchain())
    rc, _out, err = _cargo(
        ["cargo", "build", "--release", "--quiet", "--target", target],
        d, env, f"proxy {name} / {target}",
    )
    if rc != 0:
        raise RuntimeError(f"proxy 빌드 실패: {name} / {target}\n{err[-1500:]}")

    archive = tmp / "target-proxy" / target / "release" / f"lib{name}.a"
    members = subprocess.run([llvm_ar, "t", str(archive)], capture_output=True, text=True).stdout.splitlines()
    own = [m for m in members if m.startswith(f"{name}-") and m.endswith("-cgu.0.rcgu.o")]
    if not own:
        raise RuntimeError(f"proxy: {name} 자신의 오브젝트를 .a 에서 못 찾음 (멤버 {len(members)}개)")

    extract_dir = tmp / f"extract-{name}"
    extract_dir.mkdir(exist_ok=True)
    total = 0
    for member in own:
        subprocess.run([llvm_ar, "x", str(archive), member], cwd=extract_dir, check=True,
                        capture_output=True, text=True)
        out = subprocess.run([llvm_size, str(extract_dir / member)], capture_output=True, text=True)
        lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
        if len(lines) < 2:
            raise RuntimeError(f"proxy: llvm-size 출력 파싱 실패: {out.stdout!r}")
        header, data = lines[0].split(), lines[1].split()
        # Mach-O: "__TEXT __DATA __OBJC others dec hex" / ELF sysv: "text data bss dec hex filename"
        skip = {"dec", "hex", "filename", "others", "bss"}
        for key, val in zip(header, data):
            if key.lower().lstrip("_") not in skip and val.isdigit():
                total += int(val)
    return total


def proxy_diff(tmp: Path, target: str) -> int:
    """empty 대비 core 프로브의 대리(unlinked) 바이트 차분. 실패하면 예외를 던진다 —
    UNMEASURED 를 조용히 0으로 뭉개지 않는다."""
    empty = _proxy_component_bytes(tmp, "proxy_empty", EMPTY_MAIN, False, target)
    withc = _proxy_component_bytes(tmp, "proxy_core", CORE_MAIN, True, target)
    return withc - empty - len(FIXTURE.encode())


def run_gate(targets: list[str] | None = None) -> int:
    from _common import BUDGETS, PASS, FAIL, UNMEASURED, report

    cfg = BUDGETS["code"]
    limit = int(cfg["limit_bytes"])
    targets = targets or list(cfg["targets"])
    rows, measured, worst = [], {}, 0
    # 캡에 걸려 강등된 타깃 — "재보니 컸다"가 아니라 "안 기다렸다"이므로 따로 적는다.
    timed_out: dict[str, str] = {}
    # PROXY 검증용: 실측 타깃마다 "같은 대리 방법"도 같이 돌려서, 링크 없는 방법이
    # 진짜 링크값 대비 얼마나 어긋나는지 이 실행 안에서 직접 잰다(하드코딩 금지 —
    # 오늘 어긋남이 내일도 같으리란 보장이 없다).
    proxy_error_pct: dict[str, float] = {}

    if not shutil.which("cargo"):
        return report("C12 code", UNMEASURED, ["cargo 없음 — 이건 통과가 아니다"])

    for t in targets:
        installed = subprocess.run(["rustup", "target", "list", "--installed",
                                    "--toolchain", _toolchain()],
                                   capture_output=True, text=True).stdout.split()
        if t not in installed:
            rows.append(f"⬜ {t}: 타깃 미설치 — `rustup target add --toolchain {_toolchain()} {t}`")
            continue
        try:
            with tempfile.TemporaryDirectory(prefix="OICR-l2-") as td:
                tmp = Path(td)
                empty = build(tmp, "probe_empty", EMPTY_MAIN, False, t)
                withc = build(tmp, "probe_core", CORE_MAIN, True, t)
        except BuildTimeout as exc:
            # ⛔ 초과는 그 타깃만 UNMEASURED 로 강등한다 — 예산 초과가 아니다.
            timed_out[t] = str(exc)
            rows.append(f"⏱ {t}: {exc} → 이 타깃만 UNMEASURED (스위트는 계속)")
            continue
        d = withc - empty - len(FIXTURE.encode())
        measured[t] = d
        worst = max(worst, d)
        rows.append(f"{d:>10,}  {t}   (빈 {empty:,} -> +core {withc:,})")

        try:
            with tempfile.TemporaryDirectory(prefix="OICR-l2-proxy-") as td2:
                pd = proxy_diff(Path(td2), t)
            proxy_error_pct[t] = (pd - d) / d * 100 if d else float("nan")
            rows.append(f"{'':>10}  ↳ 같은 대리방법({PROXY_METHOD}) {pd:,} B "
                        f"— 실측 대비 {proxy_error_pct[t]:+.1f}% (링커 DCE 없음, 참고용)")
        except Exception as exc:  # noqa: BLE001 — PROXY 실패는 실측 게이트를 막지 않는다
            rows.append(f"{'':>10}  ↳ 대리방법 검증 실패(무시, 실측엔 영향 없음): {exc}")

    if not measured:
        return report("C12 code", UNMEASURED, rows + ["측정된 타깃 0개"])

    unmeasured_proxy: dict[str, dict] = {}
    promoted_real: dict[str, dict] = {}
    for u in cfg.get("_unmeasured", []):
        ut = u.split(" ", 1)[0].split(" —", 1)[0].strip()
        installed = subprocess.run(["rustup", "target", "list", "--installed",
                                    "--toolchain", _toolchain()],
                                   capture_output=True, text=True).stdout.split()
        if ut not in installed:
            rows.append(f"⬜ {u}")
            continue

        # ⛔ 실측 우선 시도 — NDK(또는 이 타깃용 링커)가 이 머신에 있으면 여기서
        # measured 로 승격된다. 없으면(다른 머신) 조용히 기존 proxy 폴백으로 강등한다
        # — 크래시 금지([[mcp-graceful-degradation]] Path A/B 와 동형).
        real_bytes: int | None = None
        try:
            with tempfile.TemporaryDirectory(prefix="OICR-l2-real-") as tdr:
                tmpr = Path(tdr)
                empty_r = build(tmpr, "probe_empty", EMPTY_MAIN, False, ut)
                withc_r = build(tmpr, "probe_core", CORE_MAIN, True, ut)
            real_bytes = withc_r - empty_r - len(FIXTURE.encode())
        except BuildTimeout as exc:
            # ⛔ 이 자리가 2026-08-31 에 1시간 5분 매달린 그 자리다(NDK 링커 래퍼).
            #    조용히 proxy 로 내려가면 다음 사람이 같은 원인을 또 못 짚는다.
            timed_out[ut] = str(exc)
            rows.append(f"⏱ {ut}: {exc}")
            rows.append(f"{'':>10}  ↳ 원인 후보: NDK 링커 래퍼가 arm64 맥에서 Rosetta 를 탄다"
                        f" (`ps -o %cpu=` 가 0.0% 면 행이다). 대리치로 강등한다.")
            real_bytes = None
        except SystemExit:
            real_bytes = None
        except Exception:  # noqa: BLE001 — 링커 부재 등, 실측 실패는 proxy 로 강등
            real_bytes = None

        if real_bytes is not None:
            measured[ut] = real_bytes
            worst = max(worst, real_bytes)
            rows.append(f"{real_bytes:>10,}  {ut}   (실측, NDK 링커 승격 — 예전엔 proxy 였다)")
            try:
                with tempfile.TemporaryDirectory(prefix="OICR-l2-proxy-") as td3:
                    pd = proxy_diff(Path(td3), ut)
                err = (pd - real_bytes) / real_bytes * 100 if real_bytes else float("nan")
                rows.append(f"{'':>10}  ↳ 예전 대리치 {pd:,} B 와 대조 — 실측 대비 {err:+.1f}%"
                            f" (참고용, 판정엔 실측만 쓴다)")
                promoted_real[ut] = {"bytes": real_bytes, "priorProxyBytes": pd,
                                      "priorProxyErrorPct": err,
                                      "note": "NDK 링커로 실측 성공. proxy 는 더 이상 이 타깃의 "
                                              "판정에 쓰지 않는다 — 대조용으로만 남긴다."}
            except Exception as exc:  # noqa: BLE001
                rows.append(f"{'':>10}  ↳ (예전 대리치 대조 실패, 실측치는 유효): {exc}")
                promoted_real[ut] = {"bytes": real_bytes}
            continue

        rows.append(f"⬜ {u}")
        try:
            with tempfile.TemporaryDirectory(prefix="OICR-l2-proxy-") as td3:
                pd = proxy_diff(Path(td3), ut)
        except Exception as exc:  # noqa: BLE001 — 대리치 실패는 UNMEASURED 유지, 크래시 아님
            rows.append(f"{'':>10}  ↳ 대리치 계산 실패: {exc}")
            continue
        if proxy_error_pct:
            lo, hi = min(proxy_error_pct.values()), max(proxy_error_pct.values())
            band = f"이번 실행에서 darwin/ios 실측 대비 대리방법 오차 {lo:+.1f}%~{hi:+.1f}%"
        else:
            band = "오차 대조 불가(darwin/ios 실측 자체가 이번 실행에 없음)"
        rows.append(f"{'':>10}  ↳ 대리치({PROXY_METHOD}) {pd:,} B — {band}")
        rows.append(f"{'':>10}     ⛔ 링커 DCE 없음 · 부호가 갈리는 오차라 PASS/FAIL 미사용, 참고용 상한 감")
        unmeasured_proxy[ut] = {"bytes": pd, "method": PROXY_METHOD,
                                 "validatedErrorPctByMeasuredTarget": proxy_error_pct,
                                 "note": "링크 안 된 crate 오브젝트의 text+data 합 — 링커 GC(--gc-sections) "
                                         "없이 잰 값이라 실제 android 링크 바이트가 아니다. "
                                         "validatedErrorPctByMeasuredTarget 이 같은 방법을 이번 실행에서 "
                                         "darwin/ios 실측과 대조한 오차(부호가 갈릴 수 있음 — 2026-08-28 "
                                         "1회 실측: darwin -7.3% / ios +39.7%). 부호가 갈리므로 PASS/FAIL "
                                         "판정에 쓰지 않고 참고용 상한 감으로만 쓴다."}

    if timed_out:
        rows.append("")
        rows.append(f"⏱ 캡({BUILD_TIMEOUT_S}s) 초과로 강등된 타깃 {len(timed_out)}개: "
                    f"{', '.join(timed_out)} — 재려면 `CODE_SIZE_BUILD_TIMEOUT_S=<초>`")

    rows += [f"{'-' * 10}",
             f"{worst:>10,}  최악 타깃 기준 L2",
             f"{limit:>10,}  L2 상한",
             f"{'':>10}  사용률 {worst / limit * 100:.1f}%"]

    # ⛔ 제품 약속은 L1+L2 다. 한 라인만 통과했다고 5MB 를 말할 수 없다.
    try:
        from size_gate import run_gate as _sz  # noqa: F401
    except Exception:
        pass
    from _common import gzip_bytes, walk_files
    l1 = sum(sum(gzip_bytes(f) for f in walk_files(ROOT / rel))
             for rel in BUDGETS["size"]["artifacts"] if (ROOT / rel).exists())
    total_limit = int(BUDGETS["total"]["limit_bytes"])
    rows += ["", f"{l1:>10,}  L1 데이터(gzip)",
             f"{worst:>10,}  L2 코드(네이티브, 하한선)",
             f"{l1 + worst:>10,}  L1+L2  vs 제품 약속 {total_limit:,}  "
             f"({(l1 + worst) / total_limit * 100:.1f}%)",
             "",
             "⚠️ 포함: 추론 코어 + 이벤트 링버퍼/버킷화(코어 측).",
             "⚠️ 미포함: FFI 바인딩(JNI/Swift) · 플랫폼 훅(터치 리스너·뷰 순회) · 각 런타임.",
             "   그 셋을 재기 전에는 'SDK 5MB 미만'을 외부에 말하지 마라 — 이건 하한선이다."]

    # UI 는 cargo 를 돌릴 수 없으므로 측정치를 캐시로 남긴다.
    # ⛔ 캐시가 없으면 화면은 UNMEASURED 를 보여야 한다 — 0 이 아니다.
    (ROOT / "artifacts" / "code-size.json").write_text(
        json.dumps({"measuredBytes": measured, "worstBytes": worst,
                    # ⛔ 타깃별 배포 페이로드 합계를 **코드가** 계산해 남긴다. 논문이
                    #    L1+L2 를 본문에서 손으로 더하면 그 합계는 어느 원장에도 없고,
                    #    cite_audit 이 "원장에서 못 찾은 수치" 로 잡는다(2026-09-15 실측).
                    #    유도값이라도 인용될 값이면 원장에 있어야 한다.
                    "payloadBytes": {t: l1 + b for t, b in sorted(measured.items())},
                    "l1GzipBytes": l1, "payloadLimitBytes": total_limit,
                    "limitBytes": limit, "unmeasured": cfg.get("_unmeasured", []),
                    "unmeasuredProxy": unmeasured_proxy,
                    "promotedReal": promoted_real,
                    "timedOut": timed_out, "buildTimeoutSeconds": BUILD_TIMEOUT_S,
                    "note": "링커가 남긴 네이티브 코드. FFI·플랫폼 훅·런타임 미포함 하한선. "
                            "unmeasuredProxy 는 링크 안 된 대리치 — measuredBytes/worstBytes 와 "
                            "다른 종류의 숫자이니 절대 같은 필드에 섞지 마라. promotedReal 은 이 "
                            "실행에서 NDK 등으로 실측에 성공해 measuredBytes 로 승격된 타깃(이전엔 "
                            "proxy 뿐이었음) — 다른 머신(NDK 없음)에서 재실행하면 다시 "
                            "unmeasuredProxy 로 강등될 수 있다. 이는 버그가 아니라 그 머신의 실제 "
                            "측정 가능 범위다."},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    ok = worst <= limit and (l1 + worst) <= total_limit
    return report("C12 code", PASS if ok else FAIL, rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", dest="targets")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.json:
        from _common import BUDGETS
        out = {}
        for t in (a.targets or BUDGETS["code"]["targets"]):
            with tempfile.TemporaryDirectory(prefix="OICR-l2-") as td:
                tmp = Path(td)
                out[t] = (build(tmp, "probe_core", CORE_MAIN, True, t)
                          - build(tmp, "probe_empty", EMPTY_MAIN, False, t)
                          - len(FIXTURE.encode()))
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    return run_gate(a.targets)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
