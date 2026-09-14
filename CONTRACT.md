# 구현 계약 (모든 크레이트·에이전트가 따른다)

⛔ 이 파일과 `gates/budgets.json` 이 상수의 정본이다. 크레이트가 자기 상수를 새로 정의하지
않는다 — 어긋나면 불변식 I1(같은 벡터 공간)이 조용히 깨진다.

## 고정 상수

| 이름 | 값 | 출처 |
|---|---|---|
| `VOCAB_SIZE` | 30522 | bert-base-uncased 급 워드피스 |
| `DIM` | 64 | potion-base-2M 급, 1.89M 파라미터에 맞춤 |
| 벡터 표현 | int8 테이블 → f32 누산 → L2 정규화 f32[64] | |
| 크기 상한 | 5,242,880 바이트 (gzip 후) | C1 |
| 지연 상한 | p95 50ms (Tier 0 경로만) | C2 |

## 아티팩트 포맷 (바이너리, 리틀엔디언)

```
artifacts/embedding.bin
  magic  : b"PLAIDEMB"        (8 bytes)
  version: u32 = 1 | 2        (v1 도 계속 읽는다)

  [v1 — 전역 스케일 int8]
  vocab  : u32
  dim    : u32
  scale  : f32                (전역 dequant 스케일: f = i8 * scale)
  table  : i8[vocab * dim]    (row-major)

  [v2 — 행별 스케일 + 비트폭 (2026-08-28)]
  vocab  : u32
  dim    : u32
  bits   : u32 = 4 | 8        (요소당 비트)
  scales : f32[vocab]         (행별 dequant: f = q * scales[row])
  table  : packed[vocab * row_bytes]
           row_bytes = ceil(dim * bits / 8)   ← 행은 **바이트 경계에서 시작**한다
           bits=8 : i8 그대로 (부호 있는 2의 보수)
           bits=4 : 니블 2개/바이트, **하위 니블이 먼저**.
                    저장값 u = q + 7  (q ∈ [-7, +7] → u ∈ [0, 14]; 15 는 미사용)
```

### v2 는 왜 생겼나 (2026-08-28 실측)

⛔ **전역 스케일 하나로는 4bit 이 안 선다.** 행마다 노름이 크게 다르므로 전역 스케일을
쓰면 작은 행이 전부 0 으로 뭉갠다. 행별 f32 스케일은 vocab×4 바이트(31,497 행 = 126 KB)를
더 쓰지만 4bit 품질이 그 위에 서 있다 — 아래 수치가 전부 그 포맷으로 났다.

| 구성 | L1 gzip | KLUE-STS | YNAT | NSMC |
|---|---:|---:|---:|---:|
| 31,497 × 64 int8 (v1, 현행) | 2,276,098 | 0.5806 | 0.5132 | 0.7215 |
| 31,497 × 128 int8 | 4,117,221 | 0.6058 | 0.6170 | 0.7500 |
| **31,497 × 512 @4bit** | **4,002,327** | 0.5754 | **0.6870** | **0.7750** |

⭐ 512@4bit 이 128@int8 보다 **작으면서** 분류가 7pp 높다. 근거·재현은
`docs/spec/11-dimension-decision.md` · `experiments/dim-vs-vocab/`.

⛔ **3bit 은 만들지 마라.** YNAT 이 512d 에서 0.687 → 0.648 로 절벽이다. 그래서 `bits ∈ {4,8}`
이고 다른 값은 파서가 거부한다.

### Matryoshka 절단 — 빌드 타임이 정본이다

PCA 성분이 분산 순이라 앞 k 차원만 잘라도 k 로 직접 증류한 것과 같다(실측 4점 전부 일치).
그러나 **v2 의 `scales` 는 전체 `dim` 에서 계산된 값**이므로, 런타임에 k 로 자르면 스케일이
그 k 에 대해 최적이 아니다(상위 차원이 노름을 지배해 대개 근사하지만 보장은 없다).

⇒ **기기 사양별 아티팩트는 빌드 타임에 자르고 그 k 에서 스케일을 다시 계산해 굽는다.**
런타임 절단은 허용하되 근사임을 호출부가 안다. 정확한 런타임 절단이 필요해지면 블록별
스케일(128 차원 블록당 f32)로 가야 하고, 그건 vocab×(dim/128)×4 바이트를 더 쓴다 — 지금은
필요 없어서 안 한다.

```
artifacts/projection.bin
  magic  : b"PLAIDPRJ"        (8 bytes)
  version: u32 = 1
  dim    : u32 = 64
  scale  : f32
  weights: i8[dim * dim]      (row-major, v_out = P · v_in)

artifacts/catalog.bin
  magic  : b"PLAIDCAT"        (8 bytes)
  version: u32 = 2            (v1 도 읽는다 — 그때는 label=id, leaf=없음)
  count  : u32
  dim    : u32 = 64
  scale  : f32
  entries: { id_len: u16, id: utf8,
             label_len: u16, label: utf8,   # 사람이 읽는 소재명 (v2)
             leaf: u16,                     # taxonomy.bin 의 리프 인덱스 (v2)
             vec: i8[dim] } * count

artifacts/taxonomy.bin        # 2,960 소분류 카테고리 공간
  magic  : b"PLAIDTAX"        (8 bytes)
  version: u32 = 1
  dim    : u32 = 64
  scale  : f32
  n_l1   : u32 ; per: { len: u16, utf8 }
  n_l2   : u32 ; per: { len: u16, utf8, parent_l1: u16 }
  n_leaf : u32 ; per: { len: u16, utf8, parent_l2: u16, vec: i8[dim] }
  # ⛔ 리프 id 는 저장하지 않는다 — **순서가 곧 id**(`cat-%04d`, 1-based).
  #    data/taxonomy_ko.py 에 항목을 중간에 끼워 넣으면 기기에 저장된 프로필 이력의
  #    인덱스가 통째로 어긋난다. 추가는 반드시 뒤에.

artifacts/user_embedding.bin  # 유저 임베딩 W — (나) 구조의 "큰 절반"
  magic   : b"PLAIDUEM"       (8 bytes)
  version : u32 = 1 | 2       # 1 = 두 채널 · 2 = 지각 채널(2·3층) 포함
  dim     : u32 = 128         # = DIM. 임베딩 테이블과 달리 절단을 허용하지 않는다
  hist_len: u32               # W_h 의 행 수 = n_l1 + n_l2 (택소노미가 소유)
  n_l1    : u32               # 앞 n_l1 행이 대분류 질량, 나머지가 중분류 질량
  scales_h: f32[hist_len]     # 행별 스케일 (int8 역양자화)
  q_h     : i8[hist_len * dim]
  scales_c: f32[dim]          # W_c 는 dim × dim
  q_c     : i8[dim * dim]
  scales_p: f32[hist_len]     # v2 에만. W_p = 지각 채널, 레이아웃은 W_h 와 같다
  q_p     : i8[hist_len * dim]

  # ⛔ **v1 을 계속 읽는다.** v1 을 실은 기기 = 지각 채널이 없는 기기이고, 그 상태가
  #    곧 Tier 1 없는 세계다. 옛 포맷을 거부하면 그 세계를 배포 경로에서 표현할 수
  #    없게 되고, 그러면 `gates/tier1_optional_gate.py` 가 요구하는 `off` 팔이
  #    "코드에 없는 상태"가 된다. 미지 버전(3 이상)은 거부한다.
  # ⛔ W_p 는 W_h 와 **다른 행렬**이다. 같은 것을 재사용하면 두 채널이 같은 방향만
  #    내고, 그건 채널을 합쳤을 때 콘텐츠 기여가 정확히 0 이 됐던 그 실패(모달리티
  #    붕괴)와 같은 모양이다. 입력 레이아웃만 같고 사영은 따로 배운다.

  # ⛔ 이 파일은 **기기에서 학습되지 않는다.** (나) 구조(2026-08-31 사용자 결정):
  #    큰 W 는 동의받은 패널로 우리가 학습해 배포하고(일반 유저에게서 ε 0), 작은
  #    결합기 θ 만 연합한다. 그래서 sdk-core 는 **추론만** 구현한다 — backward 도,
  #    학습용 기저 행렬(dim × n_leaf ≈ 1.5MB)도 기기에 없다. 리프 내적은
  #    `Taxonomy::scores` 가 이미 하는 그것이다.
  # ⛔ 행별 int8 이다. 전역 스케일을 쓰지 마라 — 히스토그램 행마다 질량 규모가 다르다.
  # ⚠️ 양자화는 공짜에 가깝다는 실측이 있다(잡음 σ/rms 1.6 에서도 이득 +4.59pp 유지,
  #    `experiments/user-vector-gap/split_fusion_probe.json`). 그 실측이 int8 을 고른 근거다.

artifacts/image_projection.bin  # 2층 조밀 이미지 → 우리 공간 직사각 투영 (⛔ 아직 배포 안 함)
  magic     : b"PLAIDIPJ"     (8 bytes)
  version   : u32 = 1
  vision_rev: u32             # OS 임베더 리비전 (macOS/iOS17 rev2 = 768d, iOS16 = 2048d)
  src_dim   : u32             # OS 임베딩 차원 (rev2 = 768)
  dst_dim   : u32 = 128       # = DIM. 다르면 파싱 거부
  rows      : dst_dim × { scale: f32, q: i8[src_dim] }   # 행별 int8, y[d] = scale_d·Σ q[d][s]·x[s]

  # 적용 의미(하네스와 동일): y = normalize(W·x), 정규화 바닥 1e-9
  #   (train_projection_contrastive.py 의 `normalize(x @ W)` 와 같은 식 — 전치는 export 가 한다)
  # ⛔ **리비전은 호출부가 대조한다** — `project(x, revision)` 이 기기 OS 가 보고한
  #    리비전을 받아 vision_rev 와 다르면 거부한다. iOS16(2048d)에 rev2 W 를 조용히
  #    적용하는 길을 타입이 막는다. 미지 version 도 거부(스펙 19 §5-4).
  # ⛔ 이 파일은 **아직 artifacts/ 에 실리지 않는다.** 다버티컬 재판정 + 파일럿 on−off
  #    통과 전에는 리더만 존재한다(spec 19 §5 사전 선언). 실을 때 양자화 델타를
  #    export 셀프체크(코사인)로 재고 결과를 원장에 남긴다.

artifacts/affinity.bin        # 인구통계·시간대별 대분류 친화도 (⚠️ 전부 합성)
  magic  : b"PLAIDAFF"        (8 bytes)
  version: u32 = 1
  n_l1 : u32 · n_age : u32 = 6 · n_gender : u32 = 3 · n_hour : u32 = 6
  age    : f32[n_age * n_l1]
  gender : f32[n_gender * n_l1]
  hour   : f32[n_hour * n_l1]
```

`artifacts/vocab.txt` — 한 줄에 토큰 하나, 줄 번호(0-based)가 토큰 id.

## 임베딩 알고리즘 (Tier 0) — 이 순서를 바꾸지 마라

1. 소문자화 → 유니코드 공백 분할 → 문장부호 분리
2. 각 단어를 greedy longest-match 워드피스로 분해 (`##` 접두 연속). 미매칭은 `[UNK]`
3. 각 토큰 id 의 행을 f32 로 dequant 해 누산
4. 토큰 수로 나눔 (평균 풀링). 토큰 0개면 영벡터
5. L2 정규화 (노름 0 이면 영벡터 그대로)

## 언어 간 패리티 (⛔ I1 강제)

Rust `sdk-core` 와 TS 참조 구현은 **같은 입력에 같은 벡터**를 낸다.
`gates/parity_gate.py` 가 `gates/fixtures/parity.jsonl` 의 입력에 대해 두 구현을 돌려
요소별 차이 `< 1e-5` 를 요구한다. 실패하면 배포 금지.

## 나가는 바이트 (⛔ I2)

`fl-client` 가 만드는 페이로드의 최상위 키는 `gates/budgets.json:egress.allowed_fields`
뿐이다. 그 외 키는 직렬화 단계에서 거부한다(런타임 assert + 테스트).

## 행동 이벤트 (⛔ I3 — 기기를 떠나지 않는다)

`Event` 는 **12바이트 고정 크기 POD** 다: `t_ms u32 · kind u8 · screen_id u16 ·
bucket u8 · cat_hint u16`. 힙 포인터도 문자열도 없다.

- ⛔ `Event` 에 `String`/`Vec`/`Other(String)` 을 추가하지 마라. 원문이 들어갈 자리를
  만드는 것이고, `size_of::<Event>() == 12` 테스트가 그 규율을 지킨다.
- ⛔ `EventLog` 는 `sdk-core` 에 산다. 그 크레이트의 `[dependencies]` 는 비어 있어
  serde 를 구현할 방법이 없고, 따라서 `RoundUpload` 의 필드가 **될 수 없다**
  (`InterestMemory`·`DeviceProfile`·`DeviceParams` 와 같은 수법).
- ⛔ 동의는 **두 개의 다른 타입**이다(`ServiceConsent` / `BehavioralConsent`).
  `EventLog::new` 는 `BehavioralConsent::Granted` 없이는 로그를 만들지 못한다 —
  번들 동의가 표현 불가능하다(2026-07-22 개인정보위 틱톡 처분의 1번 사유).
- **`cat_hint` 는 1-기반이다**: `0 = 없음`, 힌트 `h` 는 리프 `h-1` 을 가리킨다.
  이 오프셋을 조용히 흡수하면 리프/대분류 0 번이 통째로 사라지거나 없는 관측이 생긴다.

## OpenRTB 광고 호출 (⛔ I4 — 요청에는 의도 요약만 산다)

정본은 `docs/plans/2026-09-01/openrtb-sdk-protocol.md` §2·§4 이고 게이트는 `gates/egress_gate.py`
(C3 확장) + `gates/budgets.json::rtb` 다. 구현은 `crates/ad-rtb`(serde 가 있는 fl-client 층) —
⛔ `sdk-core` 에는 JSON 도 이 구조체도 두지 않는다(무의존 불변식).

- `BidRequest` 는 **폐쇄 구조체**다. `user` 에는 `data`·`ext` 만, `device` 에는
  `devicetype·os·osv·language·lmt` 만 필드가 있다. `user.id`·`buyeruid`·`eids`·`geo`·
  `device.ifa`·`ip`·`ua` 는 **타입에 없다** — 값을 넣을 자리가 없어서 못 나간다.
- `user.ext.OICR` v1 = `{v, sdk, tier, intent[≤ max_intent_k]{l1,l2,l2_id,conf(2자리),iab}, floor_hit}`.
  같은 배열이 `user.data[]` 의 `OICR-l2`(자체) · `OICR-iab`(`ext.segtax=7`, Content 3.0) 두
  Data 객체에도 실린다 — 한 번 만든 배열에서 둘 다 생성.
- 확신 하한 미달(`floor_hit=true`)이면 `intent=[]` 다. 약한 추측은 신호가 아니다.
- `BidResponse` 파서는 **관대**(모르는 필드 무시), HTTP 204 = 무입찰(소재 0, 에러 아님).
  `bid.ext.OICR.leaf` 는 선택이며 없으면 슬롯 탭의 `cat_hint` 는 0 이다.
- 카나리아(`rtb.canaries`)는 C3 의 것과 **다른 집합**이다 — `임신` 이 L2 `임신출산` 의 부분
  문자열이라 C3 카나리아를 그대로 쓰면 정상 요청이 FAIL 한다. `crates/ad-rtb/tests` 가
  "카나리아 ∉ 어느 L1/L2 라벨" 을 지킨다.
- 전송 헤더 `x-openrtb-version: 2.6`. 가짜 익스체인지는 `fl-server` `POST /openrtb2/bid`.
