link app: https://diabetesdisease-app.streamlit.app/
Dự án Dự đoán Nguy cơ Tiểu đường sử dụng các thuật toán học máy tiên tiến để phân tích các chỉ số sức khỏe thực tế. Mục tiêu cốt lõi là xây dựng một công cụ hỗ trợ sàng lọc sớm bệnh lý dựa trên bằng chứng khoa học và số liệu cụ thể.

Về quy trình xử lý, mình đã thực hiện phân tích tương quan giữa các biến số như chỉ số cơ thể, huyết áp và độ tuổi để tìm ra những yếu tố ảnh hưởng mạnh nhất. Dữ liệu sau đó được làm sạch và chuẩn hóa bằng kỹ thuật phân tỷ lệ chuẩn để đảm bảo các mô hình đạt được hiệu suất tối ưu nhất.

Hệ thống đánh giá bao gồm nhiều mô hình khác nhau từ hồi quy logistic đến các phương pháp kết hợp hiện đại như cây quyết định và tăng cường độ dốc. Việc thử nghiệm đa dạng mô hình giúp mình có cái nhìn khách quan về khả năng dự báo của từng thuật toán trên cùng một bộ dữ liệu.

Để nâng cao độ chính xác, mình đã áp dụng kỹ thuật tìm kiếm lưới để tối ưu hóa các tham số cốt lõi cho từng mô hình. Quá trình này giúp tinh chỉnh thuật toán đạt đến trạng thái hoạt động tốt nhất thay vì chỉ sử dụng các thiết lập mặc định thông thường.

Hệ thống kiến thức kỹ thuật (Technical Skills)
Để vận hành các mô hình trên, bạn đã thực hiện các kỹ thuật chuyên sâu sau:

Exploratory Data Analysis (EDA): Khám phá dữ liệu, vẽ biểu đồ tương quan (Heatmap) để hiểu mối quan hệ giữa các biến số sức khỏe.

Data Preprocessing: Xử lý dữ liệu thô, chuẩn hóa thang đo (StandardScaler) để các biến như BMI và Độ tuổi có vai trò ngang nhau khi tính toán.

Hyperparameter Tuning: Sử dụng GridSearchCV để máy tự động "dò" ra bộ tham số tốt nhất cho từng thuật toán.

Cross-Validation: Kỹ thuật kiểm chứng chéo (K-Fold) để đảm bảo kết quả đạt được là khách quan, không phụ thuộc vào việc chia dữ liệu ngẫu nhiên.

Overfitting Check: Phân tích khoảng cách (Gap) giữa kết quả trên tập huấn luyện và tập kiểm tra để đảm bảo mô hình có khả năng dự báo thực tế.

Evaluation Metrics: Đánh giá đa chiều qua các chỉ số Accuracy (Độ chính xác) và F1-Score (Sự cân bằng giữa độ nhạy và độ đặc hiệu).
