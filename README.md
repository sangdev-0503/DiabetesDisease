link app: https://diabetesdisease-app.streamlit.app/

🩺 Hệ Thống Chẩn Đoán Tiểu Đường Đa Phương Thức (Multimodal AI)
Dự án này được chia làm 2 giai đoạn phát triển chính nhằm tối ưu hóa khả năng dự đoán dựa trên cả chỉ số lâm sàng và hình ảnh y tế.

-  Giai đoạn 1: Dự đoán dựa trên Dữ liệu Bảng (Tabular Data)
Trong giai đoạn này, chúng tôi tập trung xử lý bộ dữ liệu BRFSS (Behavioral Risk Factor Surveillance System) để tìm ra mô hình Machine Learning tối ưu nhất cho các chỉ số sức khỏe.

1. Tiền xử lý dữ liệu (Data Engineering)
Làm sạch:bộ dữ liệu đã được xử lý sơ bộ trước khi đưa vào python,bây giờ chỉ bao gồm Xử lý giá trị thiếu và gộp nhãn (0, 1, 2) thành nhãn nhị phân (0, 1).

Cân bằng dữ liệu: Xử lý tình trạng mất cân bằng giữa nhóm người bệnh và người khỏe mạnh.

Chuẩn hóa: Sử dụng StandardScaler để đồng bộ hóa thang đo cho các biến như BMI, Age, v.v.

2. Thử nghiệm mô hình (Baseline Models)
Chúng tôi đã huấn luyện và so sánh hiệu suất của các thuật toán phổ biến:

Logistic Regression: Thiết lập mốc tham chiếu (Baseline).

K-Nearest Neighbors (KNN): Phân loại dựa trên sự tương đồng đặc trưng.

Random Forest / XGBoost: Kiểm tra tầm quan trọng của các biến (Feature Importance).
- Giai đoạn 2: Tích hợp Deep Learning & Ảnh Võng Mạc (Multimodal)
Sau khi có bộ khung dữ liệu bảng, chúng tôi nâng cấp hệ thống bằng cách tích hợp thêm Ảnh võng mạc (Retinal Images) để tăng độ chính xác trong chẩn đoán lâm sàng.

1. Khớp dữ liệu (Data Matching)
Thực hiện kỹ thuật Data Merging: Khớp chính xác ID của từng bệnh nhân trong file CSV với đường dẫn file ảnh (image_path) tương ứng.

Đảm bảo mỗi bộ chỉ số sức khỏe đều có một hình ảnh minh chứng thực tế từ đáy mắt.

2. Kiến trúc mô hình Đa phương thức (Hybrid Architecture)
Hệ thống sử dụng mạng Neural kết hợp (Late Fusion):

Nhánh Xử lý Ảnh: Sử dụng MobileNetV2 (CNN) để trích xuất các dấu hiệu bệnh lý từ võng mạc.

Nhánh Xử lý Số: Sử dụng Multi-Layer Perceptron (MLP) để phân tích các chỉ số sức khỏe.

Hợp nhất (Concatenation): Kết nối hai nguồn đặc trưng để đưa ra kết luận cuối cùng qua hàm Sigmoid.

- Triển khai Ứng dụng (Deployment)
Toàn bộ dự án được đóng gói vào ứng dụng Streamlit, cho phép:

    + Nhập các chỉ số sức khỏe tương tự như các mô hình ở Giai đoạn 1.

    + Upload ảnh võng mạc để hệ thống phân tích sâu hơn ở Giai đoạn 2.

    + Đưa ra xác suất mắc bệnh (%) dựa trên sự tổng hợp của cả hai nguồn dữ liệu.