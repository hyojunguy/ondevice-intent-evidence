//! I2: the egress key allowlist, enforced structurally rather than by convention.

use std::collections::BTreeSet;

use oicr-fl-client::{DeviceBucket, ImpressionHistogram, RoundUpload, budgets, param_len};
use serde_json::Value;

fn sample_upload() -> RoundUpload {
    RoundUpload::new(
        1,
        [0xABu8; 16],
        vec![0.01f32; param_len()],
        3,
        DeviceBucket::Mid,
        // ⛔ 예전에는 여기에 raw 광고 ID 를 넣었다 — 지금은 타입이 그걸 못 받는다(C10).
        // ⛔ 축이 28(택소노미 대분류)로 줄었으므로 예전 픽스처의 42 는 이제 범위 밖이다.
        //    가드는 정상 동작했다 — 고칠 것은 테스트 데이터다.
        ImpressionHistogram::from_buckets(&[7, 21, 11]).unwrap(),
    )
}

#[test]
fn top_level_keys_match_egress_allowlist_exactly() {
    let value = serde_json::to_value(sample_upload()).unwrap();
    let obj = value
        .as_object()
        .expect("RoundUpload serializes to a JSON object");

    let got: BTreeSet<&str> = obj.keys().map(String::as_str).collect();
    let want: BTreeSet<&str> = budgets()
        .egress
        .allowed_fields
        .iter()
        .map(String::as_str)
        .collect();

    assert_eq!(
        got, want,
        "RoundUpload's serialized keys must equal gates/budgets.json:egress.allowed_fields exactly"
    );
}

#[test]
fn no_field_can_carry_arbitrary_user_text() {
    let value = serde_json::to_value(sample_upload()).unwrap();
    let obj = value.as_object().unwrap();

    // round_id, epochs_completed: plain numbers, never strings.
    assert!(matches!(obj["round_id"], Value::Number(_)));
    assert!(matches!(obj["epochs_completed"], Value::Number(_)));

    // delta, impression_buckets: arrays of numbers only, nothing else can hide inside.
    for field in ["delta", "impression_buckets"] {
        let arr = obj[field]
            .as_array()
            .unwrap_or_else(|| panic!("{field} must be an array"));
        assert!(
            arr.iter().all(Value::is_number),
            "{field} must contain only numbers"
        );
    }

    // device_bucket: exactly one of three fixed strings, not free text.
    let bucket = obj["device_bucket"]
        .as_str()
        .expect("device_bucket is a string enum on the wire");
    assert!(
        ["low", "mid", "high"].contains(&bucket),
        "device_bucket escaped its closed enum: {bucket:?}"
    );

    // client_nonce: exactly 32 lowercase hex chars -- sixteen random bytes, not a message.
    let nonce = obj["client_nonce"]
        .as_str()
        .expect("client_nonce is hex on the wire");
    assert_eq!(
        nonce.len(),
        32,
        "client_nonce must be exactly 16 bytes hex-encoded"
    );
    assert!(
        nonce.chars().all(|c| c.is_ascii_hexdigit()),
        "client_nonce must be pure hex, not free text"
    );

    // And structurally: RoundUpload::new takes no String/&str parameter at all, so
    // there is no call that could smuggle a user sentence into this struct in the
    // first place -- client_nonce is [u8; 16], not String; the only string-shaped
    // field on the wire (device_bucket) is produced by an enum, never accepted as
    // free text on the way in.
}

#[test]
fn oversized_upload_is_rejected_by_the_client_side_guard() {
    let upload = sample_upload();
    let size = upload.serialized_size();
    assert!(
        upload.guard_size(size).is_ok(),
        "exactly-at-limit must pass"
    );
    assert!(
        upload.guard_size(size - 1).is_err(),
        "one byte over must fail"
    );
}

// ---------------------------------------------------------------------
// FedPer: 개인 슬라이스는 업로드에 담길 수 **없어야** 한다 (구조적 보장)
// ---------------------------------------------------------------------

use oicr-fl-client::{SplitDelta, federated_len, personal_len};

#[test]
fn split_delta_partitions_exactly_and_loses_nothing() {
    let full: Vec<f32> = (0..param_len()).map(|i| i as f32).collect();
    let s = SplitDelta::from_full(&full).expect("정상 길이");
    assert_eq!(s.federated().len(), federated_len());
    assert_eq!(s.personal().len(), personal_len());
    // 이어붙이면 원본과 같아야 한다 — 분할이 데이터를 흘리지 않는다.
    let rejoined: Vec<f32> = s.federated().iter().chain(s.personal()).copied().collect();
    assert_eq!(rejoined, full);
}

#[test]
fn split_delta_rejects_wrong_length() {
    assert!(SplitDelta::from_full(&vec![0.0; param_len() - 1]).is_err());
    assert!(SplitDelta::from_full(&[]).is_err());
}

#[test]
fn try_new_refuses_a_full_length_delta() {
    // ⛔ 이게 핵심이다. 전체 길이를 넘기면 개인 파라미터가 서버에 간다 —
    //    "안 보내기로 했다"가 아니라 **보낼 수 없다**를 타입이 보장한다.
    let full = vec![0.1f32; param_len()];
    let err = oicr-fl-client::RoundUpload::try_new(
        1,
        [0u8; 16],
        full,
        1,
        oicr-fl-client::DeviceBucket::Mid,
        ImpressionHistogram::empty(),
    )
    .unwrap_err();
    assert_eq!(err.got, param_len());
    assert_eq!(err.want, federated_len());
}

#[test]
fn try_new_accepts_the_federated_slice() {
    let full: Vec<f32> = (0..param_len()).map(|i| i as f32 * 0.001).collect();
    let split = SplitDelta::from_full(&full).unwrap();
    let up = oicr-fl-client::RoundUpload::try_new(
        1,
        [0u8; 16],
        split.into_federated(),
        1,
        oicr-fl-client::DeviceBucket::Mid,
        ImpressionHistogram::empty(),
    )
    .expect("연합분 길이는 통과해야 한다");
    assert_eq!(up.delta.len(), federated_len());
}

/// C10 — 업로드에서 개별 광고 ID 를 복원할 수 없다
/// (docs/spec/06-attribution-and-billing.md).
///
/// ⛔ 2026-08-26 이전에는 `impression_buckets` 가 `Vec<u32>` raw 광고 ID 목록이었다.
///    스펙(04-threat-model.md #5)은 "집계 버킷으로 완화"한다고 이미 적고 있었는데
///    코드가 그렇게 하지 않았다 — 계획에 적은 방어가 구현 안 된 형태다. 이 테스트가
///    그 간극이 다시 벌어지지 않게 한다.
#[test]
fn ad_ids_cannot_enter_the_attribution_channel() {
    use oicr-fl-client::{ImpressionError, ImpressionHistogram, attribution_buckets};

    // 광고 ID 는 의도 버킷 축(60) 밖의 임의의 수다. 타입이 애초에 안 받는다.
    let ad_id = attribution_buckets() + 941;
    assert!(matches!(
        ImpressionHistogram::from_buckets(&[ad_id]),
        Err(ImpressionError::BucketOutOfRange { .. })
    ));

    // 길이도 고정이다 — 희소 목록은 길이로 활동량을 누설한다(A3).
    let h = ImpressionHistogram::from_buckets(&[5]).unwrap();
    let v = serde_json::to_value(&h).unwrap();
    assert_eq!(
        v.as_array().unwrap().len(),
        attribution_buckets(),
        "노출 수와 무관하게 길이가 같아야 한다"
    );

    // 노출이 없을 때와 있을 때의 **모양**이 같다.
    let empty = serde_json::to_value(ImpressionHistogram::empty()).unwrap();
    assert_eq!(empty.as_array().unwrap().len(), v.as_array().unwrap().len());
}

/// 민감도가 유계다 — C9 회계의 전제. 이게 없으면 ε 을 계산할 수 없다.
#[test]
fn attribution_l2_sensitivity_is_bounded_by_the_caps() {
    use oicr-fl-client::{ImpressionHistogram, attribution_max_active_buckets, l2_sensitivity};

    let k = attribution_max_active_buckets();
    let full: Vec<usize> = (0..k).collect();
    let h = ImpressionHistogram::from_buckets(&full).unwrap();

    let norm: f64 = h
        .counts()
        .iter()
        .map(|c| (*c as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        norm <= l2_sensitivity() + 1e-9,
        "어떤 유효 히스토그램도 선언된 민감도를 넘을 수 없다: {norm} > {}",
        l2_sensitivity()
    );
}
