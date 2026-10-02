Benchmark chạy trên CPU với 284,807 giao dịch, gồm 227,845 mẫu train và 56,962 mẫu test.
Thời gian nạp dữ liệu là 2.68 giây; thời gian huấn luyện LightGBM là 2.75 giây.
Model đạt AUC-ROC 0.9024, cho thấy khả năng xếp hạng giao dịch gian lận khá tốt.
Accuracy đạt 0.9773 nhưng F1 chỉ 0.1162 và Precision 0.0623, nên còn nhiều dự đoán gian lận sai.
Recall đạt 0.8673, model phát hiện được phần lớn giao dịch gian lận trong tập test.
Best iteration bằng 1; kết quả cần được diễn giải thận trọng vì dữ liệu mất cân bằng và model dừng sớm.
Độ trễ dự đoán một dòng trung bình là 1.208 ms; throughput batch 1,000 dòng đạt khoảng 668,154 dòng/giây.
