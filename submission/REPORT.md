# Báo cáo Day 09 — Lê Trung Tá, 2A202603213

## 1. Kết luận

**Quyết định cuối cùng là Hold.** Pipeline tiếp nhận 224 ảnh, loại 19 ảnh và xuất 205 ảnh vào bộ `clean_v1` để train thử. Sau khi người duyệt mô phỏng kiểm tra 80 ảnh, lô dữ liệu còn 0 lỗi bắt buộc. Tuy nhiên, vẫn còn 38/205 ảnh có cảnh báo chưa được duyệt, chiếm 18,54%, vượt ngưỡng Hold 10% của lab.

## 2. Lỗi và phân việc

Em tổng hợp các mã xuất hiện trong `submission/reports/02_error_list_v1.csv`. Các số dưới đây là số bản ghi theo từng mã. Một ảnh có thể xuất hiện ở nhiều mã.

| Mã lỗi và số bản ghi | Ai xử lý | Cách xử lý và lý do |
|---|---|---|
| `CORRUPT_IMAGE` 2, `BAD_SIZE` 3, `DUP_IMAGE_EXACT` 8, `DUP_IMAGE_NEAR` 2, `LEAK_TEST` 4 | Máy | Loại 19 ảnh ở bước tiếp nhận do ảnh hỏng, quá nhỏ, trùng trong lô hoặc trùng với tập tham chiếu. |
| `BAD_FORMAT` 3, `BAD_CLASS_ID` 6, `BOX_ZERO` 4, `DUP_BOX` 7 | Máy | Xóa dòng nhãn sai định dạng, có ID lớp không hợp lệ, box không có diện tích hoặc box lặp. Ghi lại số dòng gốc trong nhật ký sửa. |
| `BOX_OUT_OF_IMAGE` 8, `BAD_CLASS_NAME` 2 | Máy | Cắt box về biên ảnh và chuẩn hóa hai tên lớp trong `data.yaml`. |
| `MISSING_LABEL_FILE` 6, `EMPTY_LABEL` 12 | Người duyệt | Ảnh thiếu file cần được gán nhãn. Với file rỗng, người duyệt kiểm tra xem ảnh có thực sự chỉ chứa nền hay đã bỏ sót vật thể. |
| `BOX_TINY` 6, `BOX_HUGE` 3, `CLASS_RARE` 4 | Người xem xét | Sáu cảnh báo box nhỏ nằm trên bốn ảnh. Kích thước box và tần suất lớp giúp xác định trường hợp cần kiểm tra. |
| `POSSIBLE_MISSING` 53, `POSSIBLE_LOOSE_BOX` 28, `POSSIBLE_WRONG_CLASS` 18, `POSSIBLE_FALSE_POS` 25 | Người duyệt | Đối chiếu ảnh với nhãn để quyết định có thiếu vật, lệch box, nhầm lớp hay thừa box hay không. |

Sau bước sửa tự động, hàng đợi có 125 ảnh. `img_0020` đứng đầu với 11,524 điểm vì cùng có ba tín hiệu: một dòng nhãn đã bị xóa, một cảnh báo nghi thiếu vật và một cảnh báo nghi sai lớp. Tổng điểm này cao hơn mức 10 của ảnh chỉ có `NEEDS_LABEL`.

Đối chiếu `04_error_list_v2.csv` với `06_review_log.csv`, 80 ảnh được duyệt bao phủ **94/123 cảnh báo ngữ nghĩa**. Các cảnh báo này nằm trên 74 ảnh. Con số 94 cho biết số cảnh báo đã được xem xét; một ảnh có thể có nhiều cảnh báo.

## 3. Calibration cho hai ngưỡng

Em thử các ngưỡng trên 39 ảnh trong `data/calib/` và đối chiếu với các lỗi đã chèn trong `MANIFEST.csv`.

| Tham số | Khởi đầu → đã chọn | Precision / Recall trước → sau | Lý do chọn |
|---|---|---|---|
| `duplicates.phash_dist` | 2 → 4 | Ảnh gần trùng: 1,00 / 0,50 → 1,00 / 0,67 | Số ảnh gần trùng phát hiện đúng tăng từ 3/6 lên 4/6. Các ngưỡng từ 6 đến 14 cũng chỉ phát hiện 4/6, nên em chọn 4 để hạn chế nguy cơ loại nhầm ảnh. |
| `second_opinion.conf` | 0,25 → 0,60 | Thiếu vật: 0,50 / 1,00 → 0,57 / 1,00. Sai lớp: 0,56 / 0,83 → 0,62 / 0,83. | Ở ngưỡng 0,60, checker vẫn phát hiện đủ 8/8 ca thiếu vật và 5/6 ca sai lớp, đồng thời giảm cảnh báo giả ở hai nhóm này. Đánh đổi là chỉ còn phát hiện 1/5 ca box lỏng, so với 2/5 ban đầu. |

Một trường hợp cho thấy giới hạn của tập calib là ngưỡng pHash 16. Ngưỡng này phát hiện 5/6 ảnh gần trùng trên calib với precision 1,00. Tuy nhiên, trên raw, pipeline ghép `img_0011` với `img_0008` vì khoảng cách pHash của cặp ảnh bằng 16. Khi mở ảnh, một ảnh là cảnh chơi tennis, ảnh còn lại là cảnh trượt tuyết. Vì một cảnh báo trùng có thể khiến pipeline tự loại ảnh, em giữ ngưỡng 4.

## 4. Cửa chặn và điểm mù của pipeline

Để kiểm tra gate, em chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb`. Khi ngân sách duyệt bằng 0, sáu ảnh thiếu file nhãn vẫn còn nguyên lỗi. Pipeline trả **Rework**, mã thoát **20**, và không xuất `work/exp_nb/manifest.json`. Trong lần chạy chính với ngân sách 80 ảnh, cả sáu lỗi bắt buộc đã được xử lý. Kết quả chuyển sang **Hold** vì còn 38/205 ảnh có cảnh báo chưa duyệt.

Em ghi nhận hai điểm mù:

- **Box đúng lớp nhưng vẽ lỏng.** Ở `second_opinion.conf = 0,60`, checker chỉ phát hiện 1/5 ca box lỏng đã chèn. Bốn ca còn lại cho thấy người duyệt vẫn cần nhìn đường biên vật thể, kể cả khi model và nhãn đồng ý về lớp.
- **Ảnh gần trùng đã qua biến đổi.** Ở `phash_dist = 4`, pipeline bỏ sót 2/6 ảnh gần trùng trên calib. Tăng ngưỡng có thể bắt thêm ảnh biến đổi, nhưng cũng có thể loại nhầm hai cảnh khác nhau như cặp tennis và trượt tuyết ở mục 3.

## 5. So sánh train trên raw và clean

Em train `yolo26n.pt` trong 20 epoch cho mỗi lượt, với seed 0 và 1, kích thước ảnh 640 px và batch size 16. Hai biến thể dùng chung 40 ảnh validation và 100 ảnh test. Số ảnh train trong bảng là số ảnh đầu vào mà script ghi nhận trước khi bộ nạp dữ liệu xử lý các mẫu lỗi.

| Bộ dữ liệu | Ảnh đầu vào | Số seed | mAP50 (TB ± độ lệch chuẩn) | mAP50–95 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 224 | 2 | 0,726 ± 0,000 | 0,512 ± 0,004 | 0,764 ± 0,061 | 0,649 ± 0,028 |
| Clean | 205 | 2 | 0,700 ± 0,030 | 0,486 ± 0,013 | 0,715 ± 0,073 | 0,615 ± 0,017 |

Ở seed 0, mAP50 của raw là 0,7256 và của clean là 0,6788, chênh **−0,0468**. Ở seed 1, hai giá trị lần lượt là 0,7259 và 0,7219, chênh **−0,0040**. Tính từ bốn file JSON, mAP50 trung bình của clean thấp hơn raw **0,0254**, tương đương **2,54 điểm phần trăm**. Độ lệch chuẩn giữa hai seed của clean là 0,0305, lớn hơn mức chênh trung bình. Vì vậy, raw đạt điểm cao hơn ở cả hai lượt đo, nhưng với hai seed em chưa đủ căn cứ để kết luận QC làm giảm hiệu năng nói chung. Xét theo lớp, `truck` giảm nhiều nhất, với AP50 trung bình từ 0,6485 xuống 0,5588.

Em đề xuất kiểm tra hai giả thuyết:

1. **Điểm raw có thể chịu ảnh hưởng của rò rỉ dữ liệu.** Báo cáo tiếp nhận ghi nhận bốn ảnh raw trùng với `test_009`, `test_033`, `test_044` và `test_085`. Có thể đánh giá lại cả hai mô hình trên 96 ảnh test còn lại để xem mức chênh thay đổi bao nhiêu.
2. **Nhãn còn sót lỗi có thể ảnh hưởng kết quả của clean.** Bộ clean vẫn còn 38 ảnh có cảnh báo chưa duyệt. Có thể xem lại nhóm này, ưu tiên các ca dễ nhầm giữa `car`, `truck` và `bus`, rồi train lại với cùng cấu hình và thêm seed để kiểm tra mức giảm ở lớp `truck` có lặp lại hay không.

## 6. Quyết định cuối và ghi chú sử dụng AI

Em giữ quyết định **Hold**. Với 205 ảnh và ngưỡng Hold từ 10% trở lên, lô chỉ đủ điều kiện Release khi còn tối đa 20 ảnh có cảnh báo chưa duyệt và không còn lỗi bắt buộc. Nếu số ảnh giữ nguyên, cần xử lý thêm ít nhất 18 trong 38 ảnh còn cảnh báo, sau đó chạy lại gate để xác nhận.

- **Yêu cầu AI:** Hỗ trợ viết bốn module theo đặc tả của lab, giữ số dòng gốc khi sửa nhãn và giải thích cách chọn hai ngưỡng bằng dữ liệu calibration.
- **Cách kiểm:** Chạy bộ test trên máy cá nhân được `79 passed in 37.97s`. Sau đó đối chiếu với lỗi chèn trên tập calib và thử gate với ngân sách duyệt bằng 0.
- **Vướng mắc thực tế:** Khi làm theo hướng dẫn PowerShell, em đã gọi `& $py` trước khi gán biến nên pipeline chưa chạy. Em khắc phục bằng cách gọi trực tiếp `.\.venv\Scripts\python.exe`. Qua các phép thử trên, em chưa ghi nhận lỗi logic cụ thể trong bốn module.