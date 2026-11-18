"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line


def run(ds, cfg):
    limits = cfg["label_format"]
    issues = []

    for path in ds.label_files():
        stem = path.stem
        size = ds.image_size(stem)
        previous = []

        for line, text in ds.read_label_lines(stem):
            def report(severity, code, message, **extra):
                issues.append(
                    Issue("label_format", severity, code, stem, message, line, extra)
                )

            try:
                cls, x, y, w, h = parse_yolo_line(text)
            except (ValueError, OverflowError) as exc:
                report(ERROR, "BAD_FORMAT", f"Dòng YOLO không hợp lệ: {exc}")
                continue

            if not 0 <= cls < len(ds.classes):
                report(ERROR, "BAD_CLASS_ID", f"ID lớp ngoài danh sách: {cls}")
                continue

            if w <= 0 or h <= 0:
                report(ERROR, "BOX_ZERO", "Chiều rộng hoặc chiều cao không dương")
                continue

            box = (x, y, w, h)

            # Lấy mức tràn lớn nhất trong bốn cạnh trái, trên, phải, dưới.
            overflow = max(
                w / 2 - x,
                h / 2 - y,
                x + w / 2 - 1,
                y + h / 2 - 1,
            )
            if overflow > limits["overflow_tol"]:
                report(ERROR, "BOX_OUT_OF_IMAGE", "Box vượt biên ảnh", overflow=overflow)

            if any(
                cls == old_cls and iou_xywh(box, old_box) > limits["dup_box_iou"]
                for old_cls, old_box in previous
            ):
                report(ERROR, "DUP_BOX", "Box trùng một box cùng lớp ở dòng trước")
            previous.append((cls, box))

            if size and min(w * size[0], h * size[1]) < limits["min_box_px"]:
                report(WARN, "BOX_TINY", "Cạnh ngắn của box dưới ngưỡng pixel")

            if w * h > limits["max_box_frac"]:
                report(WARN, "BOX_HUGE", "Box chiếm diện tích quá lớn")

    return issues