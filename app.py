import streamlit as st 
import pandas as pd 
import joblib
import numpy as np

model=joblib.load("models/Logistic_Regresstion_Diabetes.pkl")
scaler=joblib.load("models/scaler.pkl")
expected_columns=joblib.load("models/columns.pkl")



# st.title("Dự đoán khả năng mắc bệnh tiểu đường")
# st.markdown("cung cấp các chỉ số cần thiết!!!")


# def yes_no_to_int(label):
#     return 1 if label == "yes" else 0


# Sex = st.selectbox("Giới tính", [0, 1], format_func=lambda x: "Nữ" if x == 0 else "Nam")
# Age = st.slider("Độ tuổi (Nhóm 1-13)", 1, 13, 5) # Lưu ý: Data của bạn Age thường chia theo nhóm
# BMI = st.slider("Chỉ số BMI", 10, 60, 25)
# MentHlth = st.slider("Số ngày sức khỏe tinh thần không tốt (0-30)", 0, 30, 0)
# PhysHlth = st.slider("Số ngày sức khỏe thể chất không tốt (0-30)", 0, 30, 0)
# GenHlth = st.slider("Tình trạng sức khỏe chung (1: Rất tốt - 5: Tệ)", 1, 5, 2)
# Education = st.slider("Trình độ học vấn (1-6)", 1, 6, 4)
# Income = st.slider("Mức thu nhập (1-8)", 1, 8, 5)

# # Cột 2: Các giá trị Yes/No (Dùng Selectbox)
# # Lưu ý: Đã thêm dấu phẩy giữa tiêu đề và List lựa chọn
# HighBP = yes_no_to_int(st.selectbox("Có bị cao huyết áp không?", ["no", "yes"]))
# HighChol = yes_no_to_int(st.selectbox("Có bị mỡ máu cao không?", ["no", "yes"]))
# CholCheck = yes_no_to_int(st.selectbox("Có kiểm tra mỡ máu trong 5 năm qua?", ["no", "yes"]))
# Smoker = yes_no_to_int(st.selectbox("Đã từng hút ít nhất 100 điếu thuốc?", ["no", "yes"]))
# Stroke = yes_no_to_int(st.selectbox("Có tiền sử bị đột quỵ không?", ["no", "yes"]))
# HeartDiseaseorAttack = yes_no_to_int(st.selectbox("Bị bệnh tim mạch hoặc nhồi máu cơ tim?", ["no", "yes"]))
# PhysActivity = yes_no_to_int(st.selectbox("Có hoạt động thể chất trong 30 ngày qua?", ["no", "yes"]))
# Fruits = yes_no_to_int(st.selectbox("Có ăn trái cây hàng ngày?", ["no", "yes"]))
# Veggies = yes_no_to_int(st.selectbox("Có ăn rau xanh hàng ngày?", ["no", "yes"]))
# HvyAlcoholConsump = yes_no_to_int(st.selectbox("Có uống nhiều rượu bia không?", ["no", "yes"]))
# AnyHealthcare = yes_no_to_int(st.selectbox("Có bảo hiểm y tế không?", ["no", "yes"]))
# NoDocbcCost = yes_no_to_int(st.selectbox("Không đi khám vì chi phí cao?", ["no", "yes"]))
# DiffWalk = yes_no_to_int(st.selectbox("Có gặp khó khăn khi đi lại/leo cầu thang?", ["no", "yes"]))


# # Gom tất cả vào mảng để dự đoán
# # CẢNH BÁO: Thứ tự các biến phải khớp hoàn toàn với thứ tự cột lúc bạn Train model
# features = [HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, 
#             HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, 
#             HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, 
#             MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income]

# if st.button("Dự đoán kết quả"):
#     # Chuyển về mảng 2 chiều
#     input_array = np.array([features])
    
#     # 1. Chuẩn hóa bằng scaler.pkl
#     input_scaled = scaler.transform(input_array)
    
#     # 2. Dự đoán bằng Logistic Regresstion_Diabetes.pkl
#     prediction = model.predict(input_scaled)
    
#     if prediction[0] == 1:
#         st.error("Cảnh báo: Bạn có nguy cơ cao bị tiểu đường!")
#     else:
#         st.success("Chúc mừng: Bạn có nguy cơ thấp với bệnh tiểu đường.")


st.markdown("""
    <style>
    /* 🎨 1. NỀN TỔNG THỂ - Xám cực nhẹ để làm nổi bật các Card trắng */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* 🎨 2. TIÊU ĐỀ CHÍNH - Nền Xanh đậm, Chữ Trắng để nổi bật */
    .main-title {
        color: #ffffff; 
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 800;
        padding: 30px;
        margin-bottom: 30px;
        background: linear-gradient(135deg, #1e4d2b 0%, #2e7d32 100%); /* Gradient Xanh Lá */
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(46, 125, 50, 0.2);
    }

    /* 🎨 3. CÁC KHỐI CARD - Nền TRẮNG (thay vì Đen), Viền mỏng */
    .category-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        height: 100%;
        border-top: 5px solid #66bb6a; /* Viền trên Xanh tươi */
        margin-bottom: 20px;
        border-left: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }

    /* 🎨 4. MÀU CHỮ TIÊU ĐỀ CON TRONG CARD */
    .category-card h3 {
        color: #1b5e20 !important; /* Xanh lá cực đậm */
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 15px;
        border-bottom: 2px solid #f0f4f0;
        padding-bottom: 10px;
    }

    /* 🎨 5. MÀU CHỮ CHO NHÃN (LABELS) - Xám đậm chuyên nghiệp */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* 🎨 6. NÚT DỰ ĐOÁN - Gradient hiện đại */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%);
        color: white !important;
        border-radius: 12px;
        width: 100%;
        height: 55px;
        font-size: 18px;
        font-weight: 700;
        border: none;
        margin-top: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.3);
    }
    
    /* Hiệu ứng Hover nút */
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(46, 125, 50, 0.4);
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        color: #ffffff !important;
    }

    /* 🎨 7. CHỈNH MÀU CÁC CỘT INPUT - Giúp text bên trong rõ hơn */
    input {
        color: #1e293b !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🩺 Hệ Thống Dự Đoán Tiểu Đường</h1>", unsafe_allow_html=True)

# 3. HÀM HỖ TRỢ
def yes_no_to_int(label):
    return 1.0 if label == "yes" else 0.0

# 4. GIAO DIỆN NHẬP LIỆU CHIA CỘT
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Chỉ số cơ bản")
    Sex = st.selectbox("Giới tính", [0, 1], format_func=lambda x: "Nữ" if x == 0 else "Nam")
    Age = st.slider("Nhóm tuổi (1-13)", 1, 13, 5)
    BMI = st.slider("Chỉ số BMI", 10, 100, 30)
    GenHlth = st.slider("Sức khỏe chung (1:Tốt - 5:Tệ)", 1, 5, 2)
    Education = st.slider("Học vấn (1-6)", 1, 6, 4)
    Income = st.slider("Thu nhập (1-8)", 1, 8, 5)
    MentHlth = st.slider("Số ngày sức khỏe tinh thần kém (0-30)", 0, 30, 0)
    PhysHlth = st.slider("Số ngày sức khỏe thể chất kém (0-30)", 0, 30, 0)


with col2:
    st.subheader("Tiền sử bệnh lý")
    HighBP = yes_no_to_int(st.selectbox("Cao huyết áp?", ["no", "yes"]))
    HighChol = yes_no_to_int(st.selectbox("Mỡ máu cao?", ["no", "yes"]))
    CholCheck = yes_no_to_int(st.selectbox("Kiểm tra mỡ máu (5 năm)?", ["no", "yes"]))
    Stroke = yes_no_to_int(st.selectbox("Từng bị đột quỵ?", ["no", "yes"]))
    HeartDiseaseorAttack = yes_no_to_int(st.selectbox("Bệnh tim mạch?", ["no", "yes"]))
    DiffWalk = yes_no_to_int(st.selectbox("Khó khăn khi đi lại?", ["no", "yes"]))
    

with col3:
    st.subheader("Thói quen ")
    Smoker = yes_no_to_int(st.selectbox("Có hút thuốc?", ["no", "yes"]))
    PhysActivity = yes_no_to_int(st.selectbox("Hoạt động thể chất?", ["no", "yes"]))
    Fruits = yes_no_to_int(st.selectbox("Ăn trái cây hàng ngày?", ["no", "yes"]))
    Veggies = yes_no_to_int(st.selectbox("Ăn rau xanh hàng ngày?", ["no", "yes"]))
    HvyAlcoholConsump = yes_no_to_int(st.selectbox("Uống nhiều rượu bia?", ["no", "yes"]))
    AnyHealthcare = yes_no_to_int(st.selectbox("Có bảo hiểm y tế?", ["no", "yes"]))
    NoDocbcCost = yes_no_to_int(st.selectbox("Ngại đi khám vì phí cao?", ["no", "yes"]))

# Hai biến này để cuối cho gọn

st.write("---")

# 5. XỬ LÝ DỰ ĐOÁN
# Cực kỳ quan trọng: Thứ tự các biến trong list này phải khớp hoàn toàn với dữ liệu lúc train
features = [HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, 
            HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, 
            HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, 
            MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income]

if st.button("BẮT ĐẦU PHÂN TÍCH"):
    # Chuyển đổi sang định dạng Numpy và Scale
    input_array = np.array([features])
    input_scaled = scaler.transform(input_array)
    
    # Dự đoán
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled) # Lấy xác suất nếu model hỗ trợ

    st.subheader("Kết quả phân tích:")
    if prediction[0] == 1:
        st.error(f"CẢNH BÁO: Bạn có nguy cơ cao mắc bệnh tiểu đường (Xác suất: {probability[0][1]:.2%})")
        st.write("Lời khuyên: Bạn nên đến cơ sở y tế gần nhất để kiểm tra chuyên sâu.")
    else:
        st.success(f"AN TOÀN: Nguy cơ mắc bệnh tiểu đường của bạn thấp (Xác suất: {probability[0][1]:.2%})")
        st.write("Lời khuyên: Duy trì chế độ ăn uống và tập luyện điều độ nhé!")