"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework.   ★

Test:  pytest tests/test_gate.py -q

Cây quyết định Day 07 (slide Day 09: "Lỗi bắt buộc chặn pipeline; lỗi cảnh báo chuyển người xem"):

  còn ≥ 1 lỗi bắt buộc (severity ERROR)                          -> "Rework"
  không còn lỗi bắt buộc, nhưng tỉ lệ ẢNH còn cảnh báo
      >= cfg["gate"]["warn_ratio_hold"]                          -> "Hold"
  còn lại                                                        -> "Release"

Tỉ lệ cảnh báo = (số ẢNH khác nhau có ≥ 1 cảnh báo) / (tổng số ảnh).  Chú ý:
- chỉ đếm issue có `image` nằm trong `image_stems` (bỏ "<dataset>", "data.yaml");
- bỏ qua ảnh nằm trong `reviewed` (người đã xem, cảnh báo của ảnh đó coi như đã xử lý) — nhưng LỖI BẮT BUỘC thì không bao giờ được bỏ qua;
- mẫu số là số ảnh, không phải số issue.

Trả về dict: decision, n_errors, n_warn_images, n_images, warn_ratio (làm tròn 4 chữ số), reasons (list[str], giải thích bằng tiếng Việt, có số liệu).
"""
from ..common import ERROR, WARN


def decide(issues, image_stems, cfg, reviewed=frozenset()):
    images = set(image_stems)
    n_errors = 0
    flagged = set()

    for issue in issues:
        if issue.severity == ERROR:
            # Ảnh đã duyệt vẫn phải tính lỗi bắt buộc.
            n_errors += 1
        elif (
            issue.severity == WARN
            and issue.image in images
            and issue.image not in reviewed
        ):
            # Một ảnh có nhiều cảnh báo cũng chỉ tính một lần.
            flagged.add(issue.image)

    ratio = len(flagged) / len(images) if images else 0.0
    limit = cfg["gate"]["warn_ratio_hold"]

    if n_errors:
        decision = "Rework"
        reason = f"Còn {n_errors} lỗi bắt buộc."
    else:
        decision = "Hold" if ratio >= limit else "Release"
        reason = (
            f"Có {len(flagged)}/{len(images)} ảnh còn cảnh báo chưa duyệt, "
            f"ngưỡng Hold là {limit:.0%}."
        )

    return {
        "decision": decision,
        "n_errors": n_errors,
        "n_warn_images": len(flagged),
        "n_images": len(images),
        "warn_ratio": round(ratio, 4),
        "reasons": [reason],
    }