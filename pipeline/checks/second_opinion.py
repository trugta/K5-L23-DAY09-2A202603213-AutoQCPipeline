"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    limits = cfg["second_opinion"]
    issues = []

    for path in ds.image_files():
        stem = path.stem
        lines = ds.read_label_lines(stem)

        if stem not in preds or lines is None:
            continue

        # Giữ số dòng gốc để chỉ đúng vị trí khi phát hiện box nghi thừa.
        labels = []
        for line, text in lines:
            try:
                cls, *box = parse_yolo_line(text)
            except (ValueError, OverflowError):
                continue

            if 0 <= cls < len(ds.classes) and min(box[2:]) > 0:
                labels.append((line, cls, box))

        size = ds.image_size(stem)

        # Chiều 1: với mỗi vật model thấy, tìm box nhãn tương ứng.
        for cls, conf, *box in preds[stem]:
            if conf < limits["conf"] or size is None:
                continue

            area_px = box[2] * size[0] * box[3] * size[1]
            if area_px < limits["min_area_px"]:
                continue

            matches = [
                (label_cls, iou_xywh(box, label_box))
                for _, label_cls, label_box in labels
            ]
            best = max((iou for _, iou in matches), default=0)
            same = max(
                (iou for label_cls, iou in matches if label_cls == cls),
                default=0,
            )

            if best < limits["iou_missing"]:
                code = "POSSIBLE_MISSING"
                message = "Model thấy vật chưa có box tương ứng"
            elif same >= limits["iou_missing"]:
                if same >= limits["iou_loose"]:
                    continue
                code = "POSSIBLE_LOOSE_BOX"
                message = "Box cùng lớp khớp chưa sát"
                best = same
            elif best >= limits["iou_match"]:
                code = "POSSIBLE_WRONG_CLASS"
                message = "Box khớp vị trí nhưng khác lớp"
            else:
                continue

            issues.append(
                Issue(
                    "second_opinion", WARN, code, stem, message,
                    extra={"conf": conf, "iou": best},
                )
            )

        # Chiều 2: với mỗi box nhãn, kiểm tra model có thấy vật ở đó không.
        # Bước này dùng fp_conf riêng và xét cả các dự đoán nhỏ.
        candidates = [p[2:] for p in preds[stem] if p[1] >= limits["fp_conf"]]

        for line, _, box in labels:
            best = max(
                (iou_xywh(box, candidate) for candidate in candidates),
                default=0,
            )

            if best < limits["iou_missing"]:
                issues.append(
                    Issue(
                        "second_opinion", WARN, "POSSIBLE_FALSE_POS", stem,
                        "Nhãn không khớp dự đoán nào", line, {"iou": best},
                    )
                )

    return issue

