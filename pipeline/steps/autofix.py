"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict

import yaml

from ..common import ERROR, parse_yolo_line


DROP = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}


def _clip(text):
    """Cắt box theo biên ảnh rồi chuyển về dạng YOLO."""
    cls, x, y, w, h = parse_yolo_line(text)

    left = max(0, x - w / 2)
    right = min(1, x + w / 2)
    top = max(0, y - h / 2)
    bottom = min(1, y + h / 2)

    if right <= left or bottom <= top:
        return None

    values = (
        (left + right) / 2,
        (top + bottom) / 2,
        right - left,
        bottom - top,
    )
    return f"{cls} " + " ".join(str(value) for value in values)


def fix(ds, issues, cfg):
    groups = defaultdict(list)
    for issue in issues:
        if issue.severity == ERROR:
            groups[issue.image].append(issue)

    changes = []

    def record(stem, line, action, before, after):
        changes.append({
            "image": stem,
            "line": line,
            "action": action,
            "before": before,
            "after": after,
        })

    for stem, errors in groups.items():
        codes = {issue.code for issue in errors}

        if "BAD_CLASS_NAME" in codes:
            before = ds.yaml_names()
            expected = dict(enumerate(cfg["classes"]))

            if before != expected:
                path = ds.yaml_path
                if path.exists():
                    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                else:
                    data = {}

                data["names"] = expected
                path.write_text(
                    yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )
                record(stem, None, "rename_classes", str(before), str(expected))
            continue

        path = ds.label_path(stem)

        # Xử lý lỗi cần loại ảnh trước các lỗi dòng nhãn của cùng ảnh.
        if codes & REMOVE or any(code.startswith("DUP_IMAGE_") for code in codes):
            image = ds.image_path(stem)
            if image:
                image.unlink()
                path.unlink(missing_ok=True)
                record(stem, None, "remove_image", image.name, "")
            continue

        if not path.exists():
            if "MISSING_LABEL_FILE" in codes and ds.image_path(stem):
                record(stem, None, "needs_label", "(không có file)", "chuyển người gán")
            continue

        # Gom mã lỗi theo số dòng gốc để việc xóa không làm lệch dòng sau.
        by_line = defaultdict(set)
        for issue in errors:
            if issue.line is not None:
                by_line[issue.line].add(issue.code)

        original = path.read_text(encoding="utf-8", errors="replace").splitlines()
        updated = []

        for line, text in enumerate(original, 1):
            flags = by_line.get(line, set())

            if flags & DROP:
                after = None
            elif "BOX_OUT_OF_IMAGE" in flags:
                after = _clip(text)
            else:
                updated.append(text)
                continue

            if after is None:
                record(stem, line, "drop_line", text, "")
            else:
                updated.append(after)
                record(stem, line, "clip_box", text, after)

        if updated == original:
            path.write_text(
                "".join(text + "\n" for text in updated),
                encoding="utf-8",
            )

    return changes
