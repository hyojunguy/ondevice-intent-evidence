#!/usr/bin/env python3
"""AI-Hub 71603(속성기반 감정분석) Validation 라벨링데이터 20묶음을 내려받아 푼다.

⛔ 원본도 재가공본도 이 저장소에 커밋되지 않는다 — 받는 위치는 `.gitignore` 의 `aihub_raw/` 다.
   AI-Hub 표준 약관: 원본·단순 재가공 데이터는 출처를 표기해도 공개·재배포할 수 없고,
   해외 플랫폼 저장은 국외 반출에 해당한다. 커밋되는 것은 매핑·하네스·집계값뿐이다.

⛔ `aihubshell` 을 쓰지 않는다. 그 스크립트의 병합 루틴은 `printf '%q'` 로 이스케이프한 문자열을
   `find -name` 에 넣어서 한글 파일명이 매칭되지 않는데, 그 상태로 출력 리다이렉트가 원본을
   비우고 `rm ...part*` 가 조각을 지운다(0바이트 산출). 여기서는 필요한 파일만 직접 받는다.

⛔ 키는 인자로도 로그로도 나오지 않는다 — 환경변수 `AIHUB_APIKEY`(또는 `AIHUB_ENV_FILE`)에서만 읽는다.
"""
from __future__ import annotations
import io, os, pathlib, sys, urllib.request, zipfile

HERE = pathlib.Path(__file__).resolve().parent
DEST = HERE / "aihub_raw"
# Validation / 02.라벨링데이터 / 쇼핑몰 20묶음 (SNS 5개는 제외 — 제품명 필드가 없다)
FILE_KEYS = list(range(502285, 502305))
URL = "https://api.aihub.or.kr/down/0.6/71603.do?fileSn={}"


def apikey() -> str:
    """환경변수 우선, 없으면 `AIHUB_ENV_FILE` 이 가리키는 .env 에서 읽는다.

    ⛔ 경로를 하드코딩하지 않는다. 이 스크립트는 증거 번들로 공개되는데, 개인 디렉터리
       경로는 그 자체로 신원 노출이다(2026-09-15 중화 게이트가 잡았다).
    ⛔ 키는 인자로도 로그로도 나오지 않는다.
    """
    key = os.environ.get("AIHUB_APIKEY")
    if key:
        return key.strip()
    env = pathlib.Path(os.environ.get("AIHUB_ENV_FILE", "")) if os.environ.get("AIHUB_ENV_FILE") else None
    if env and env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("AIHUB_APIKEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("AIHUB_APIKEY 환경변수를 설정하거나 AIHUB_ENV_FILE 로 .env 경로를 알려라")


def main() -> int:
    key = apikey()
    DEST.mkdir(parents=True, exist_ok=True)
    ok = 0
    for fs in FILE_KEYS:
        req = urllib.request.Request(URL.format(fs), headers={"apikey": key})
        try:
            blob = urllib.request.urlopen(req, timeout=180).read()
        except Exception as e:                                  # noqa: BLE001
            print(f"  ✗ {fs}: {type(e).__name__}", flush=True); continue
        # 응답은 tar 이고 그 안에 zip 이 들어 있다. tar 없이 zip 시그니처를 찾아 바로 연다.
        n = 0
        try:
            import re, tarfile
            with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
                # ⛔ zip 이 통째로 오지 않는다 — `....zip.part0`, `.part1` 로 쪼개져 온다.
                #    aihubshell 이 여기서 죽는 자리다. 번호순으로 이어 붙여야 zip 이 된다.
                parts = []
                for m in tf.getmembers():
                    hit = re.search(r"\.zip(?:\.part(\d+))?$", m.name, re.I)
                    if m.isfile() and hit:
                        parts.append((int(hit.group(1) or 0), m))
                if not parts:
                    print(f"  ✗ {fs}: zip 조각이 없다", flush=True); continue
                parts.sort(key=lambda x: x[0])
                raw = b"".join(tf.extractfile(m).read() for _i, m in parts)
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        # ⛔ 엔트리 이름이 CP949 다 — unzip 은 "Illegal byte sequence" 로 죽는다.
                        try:
                            nm = info.filename.encode("cp437").decode("cp949")
                        except Exception:                   # noqa: BLE001
                            nm = info.filename
                        (DEST / pathlib.Path(nm).name).write_bytes(zf.read(info)); n += 1
        except tarfile.ReadError:
            print(f"  ✗ {fs}: tar 아님 (앞 120바이트: {blob[:120]!r})", flush=True); continue
        print(f"  ✓ {fs}: json {n}개", flush=True); ok += 1
    print(f"\n{ok}/{len(FILE_KEYS)} 묶음 · {DEST} 에 파일 {len(list(DEST.glob('*')))}개")
    return 0 if ok == len(FILE_KEYS) else 1


if __name__ == "__main__":
    sys.exit(main())
