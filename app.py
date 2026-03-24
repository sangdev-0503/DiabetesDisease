# import streamlit as st 
# import pandas as pd 
# import joblib
# import numpy as np

# model=joblib.load("models/Logistic_Regresstion_Diabetes.pkl")
# scaler=joblib.load("models/scaler.pkl")
# expected_columns=joblib.load("models/columns.pkl")

# st.markdown("""
#     <style>
#     /* 🎨 1. NỀN TỔNG THỂ - Xám cực nhẹ để làm nổi bật các Card trắng */
#     .stApp {
#         background-color: #f8fafc;
#     }
    
#     /* 🎨 2. TIÊU ĐỀ CHÍNH - Nền Xanh đậm, Chữ Trắng để nổi bật */
#     .main-title {
#         color: #ffffff; 
#         text-align: center;
#         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#         font-weight: 800;
#         padding: 30px;
#         margin-bottom: 30px;
#         background: linear-gradient(135deg, #1e4d2b 0%, #2e7d32 100%); /* Gradient Xanh Lá */
#         border-radius: 20px;
#         box-shadow: 0 10px 25px rgba(46, 125, 50, 0.2);
#     }

#     /* 🎨 3. CÁC KHỐI CARD - Nền TRẮNG (thay vì Đen), Viền mỏng */
#     .category-card {
#         background-color: #ffffff;
#         padding: 25px;
#         border-radius: 15px;
#         box-shadow: 0 4px 15px rgba(0,0,0,0.05);
#         height: 100%;
#         border-top: 5px solid #66bb6a; /* Viền trên Xanh tươi */
#         margin-bottom: 20px;
#         border-left: 1px solid #e2e8f0;
#         border-right: 1px solid #e2e8f0;
#         border-bottom: 1px solid #e2e8f0;
#     }

#     /* 🎨 4. MÀU CHỮ TIÊU ĐỀ CON TRONG CARD */
#     .category-card h3 {
#         color: #1b5e20 !important; /* Xanh lá cực đậm */
#         font-size: 1.2rem;
#         font-weight: 700;
#         margin-bottom: 15px;
#         border-bottom: 2px solid #f0f4f0;
#         padding-bottom: 10px;
#     }

#     /* 🎨 5. MÀU CHỮ CHO NHÃN (LABELS) - Xám đậm chuyên nghiệp */
#     .stSelectbox label, .stSlider label, .stNumberInput label {
#         color: #334155 !important;
#         font-weight: 600 !important;
#         font-size: 0.95rem !important;
#     }

#     /* 🎨 6. NÚT DỰ ĐOÁN - Gradient hiện đại */
#     div.stButton > button:first-child {
#         background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%);
#         color: white !important;
#         border-radius: 12px;
#         width: 100%;
#         height: 55px;
#         font-size: 18px;
#         font-weight: 700;
#         border: none;
#         margin-top: 10px;
#         transition: all 0.3s ease;
#         box-shadow: 0 4px 12px rgba(46, 125, 50, 0.3);
#     }
    
#     /* Hiệu ứng Hover nút */
#     div.stButton > button:first-child:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 8px 20px rgba(46, 125, 50, 0.4);
#         background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
#         color: #ffffff !important;
#     }

#     /* 🎨 7. CHỈNH MÀU CÁC CỘT INPUT - Giúp text bên trong rõ hơn */
#     input {
#         color: #1e293b !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# st.markdown("<h1 class='main-title'>🩺 Hệ Thống Dự Đoán Tiểu Đường</h1>", unsafe_allow_html=True)


# def yes_no_to_int(label):
#     return 1.0 if label == "yes" else 0.0

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.subheader("Chỉ số cơ bản")
#     Sex = st.selectbox("Giới tính", [0, 1], format_func=lambda x: "Nữ" if x == 0 else "Nam")
#     Age = st.slider("Nhóm tuổi (1-13)", 1, 13, 5)
#     BMI = st.slider("Chỉ số BMI", 10, 100, 30)
#     GenHlth = st.slider("Sức khỏe chung (1:Tốt - 5:Tệ)", 1, 5, 2)
#     Education = st.slider("Học vấn (1-6)", 1, 6, 4)
#     Income = st.slider("Thu nhập (1-8)", 1, 8, 5)
#     MentHlth = st.slider("Số ngày sức khỏe tinh thần kém (0-30)", 0, 30, 0)
#     PhysHlth = st.slider("Số ngày sức khỏe thể chất kém (0-30)", 0, 30, 0)


# with col2:
#     st.subheader("Tiền sử bệnh lý")
#     HighBP = yes_no_to_int(st.selectbox("Cao huyết áp?", ["no", "yes"]))
#     HighChol = yes_no_to_int(st.selectbox("Mỡ máu cao?", ["no", "yes"]))
#     CholCheck = yes_no_to_int(st.selectbox("Kiểm tra mỡ máu (5 năm)?", ["no", "yes"]))
#     Stroke = yes_no_to_int(st.selectbox("Từng bị đột quỵ?", ["no", "yes"]))
#     HeartDiseaseorAttack = yes_no_to_int(st.selectbox("Bệnh tim mạch?", ["no", "yes"]))
#     DiffWalk = yes_no_to_int(st.selectbox("Khó khăn khi đi lại?", ["no", "yes"]))
    

# with col3:
#     st.subheader("Thói quen ")
#     Smoker = yes_no_to_int(st.selectbox("Có hút thuốc?", ["no", "yes"]))
#     PhysActivity = yes_no_to_int(st.selectbox("Hoạt động thể chất?", ["no", "yes"]))
#     Fruits = yes_no_to_int(st.selectbox("Ăn trái cây hàng ngày?", ["no", "yes"]))
#     Veggies = yes_no_to_int(st.selectbox("Ăn rau xanh hàng ngày?", ["no", "yes"]))
#     HvyAlcoholConsump = yes_no_to_int(st.selectbox("Uống nhiều rượu bia?", ["no", "yes"]))
#     AnyHealthcare = yes_no_to_int(st.selectbox("Có bảo hiểm y tế?", ["no", "yes"]))
#     NoDocbcCost = yes_no_to_int(st.selectbox("Ngại đi khám vì phí cao?", ["no", "yes"]))



# st.write("---")

# features = [HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, 
#             HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, 
#             HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, 
#             MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income]

# if st.button("BẮT ĐẦU PHÂN TÍCH"):
#     input_array = np.array([features])
#     input_scaled = scaler.transform(input_array)
    
#     prediction = model.predict(input_scaled)
#     probability = model.predict_proba(input_scaled)

#     st.subheader("Kết quả phân tích:")
#     if prediction[0] == 1:
#         st.error(f"CẢNH BÁO: Bạn có nguy cơ cao mắc bệnh tiểu đường (Xác suất: {probability[0][1]:.2%})")
#         st.write("Lời khuyên: Bạn nên đến cơ sở y tế gần nhất để kiểm tra chuyên sâu.")
#     else:
#         st.success(f"AN TOÀN: Nguy cơ mắc bệnh tiểu đường của bạn thấp (Xác suất: {probability[0][1]:.2%})")
#         st.write("Lời khuyên: Duy trì chế độ ăn uống và tập luyện điều độ nhé!")


import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import joblib
import os
from PIL import Image

st.set_page_config(page_title="AI Chẩn đoán Tiểu đường", layout="wide", page_icon="🩺")


st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 Hệ thống Chẩn đoán Tiểu đường Đa phương thức")
st.write("Dự án ứng dụng Deep Learning kết hợp Ảnh võng mạc và Chỉ số sức khỏe (BRFSS)")

@st.cache_resource
def load_assets():
    
    model_path = 'Notebook/models/best_multimodal_model.h5'
    scaler_path = 'Notebook/models/scaler.pkl'
    cols_path = 'Notebook/models/columns.pkl'
    
    if not os.path.exists(model_path):
        st.error("Không tìm thấy file model trong thư mục 'models/'.")
        return None, None, None
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    columns = joblib.load(cols_path)
    return model, scaler, columns

model, scaler, columns = load_assets()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Nhập chỉ số sức khỏe")
    user_inputs = {}

    for col in columns:
        if col in ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack']:
            user_inputs[col] = st.selectbox(f"{col} (0: Không, 1: Có)", [0, 1])
        elif col == 'BMI':
            user_inputs[col] = st.number_input(f"Chỉ số {col}", min_value=10.0, max_value=60.0, value=25.0)
        elif col == 'Age':
            user_inputs[col] = st.slider(f"Nhóm tuổi (1-13)", 1, 13, 5)
        else:
            user_inputs[col] = st.number_input(f"{col}", value=0.0)

with col2:
    st.subheader("Phân tích Ảnh Võng mạc")
    uploaded_file = st.file_uploader("Tải lên ảnh đáy mắt (Retinal Image)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Ảnh đã tải lên', width=350)
        
        if st.button("BẮT ĐẦU PHÂN TÍCH"):
            with st.spinner('Đang xử lý dữ liệu...'):
  
                img_array = np.array(img.convert('RGB'))
                img_input = cv2.resize(img_array, (224, 224))
                img_input = img_input / 255.0
                img_input = np.expand_dims(img_input, axis=0)

                # 2. Tiền xử lý dữ liệu bảng
                df_input = pd.DataFrame([user_inputs])
                tab_input = scaler.transform(df_input)

                # 3. Dự đoán từ mô hình Multimodal
                prediction = model.predict({
                    "image_input": img_input, 
                    "tabular_input": tab_input
                })
                prob = prediction[0][0]

    
                st.divider()
                if prob > 0.5:
                    st.error(f"### KẾT QUẢ: CÓ DẤU HIỆU BỆNH")
                    st.write(f"**Độ tin cậy của AI:** {prob*100:.2f}%")
                    st.warning("Chú ý: Đây là kết quả tham khảo từ AI, bạn cần tham vấn ý kiến bác sĩ chuyên khoa.")
                else:
                    st.success(f"### KẾT QUẢ: BÌNH THƯỜNG")
                    st.write(f"**Độ tin cậy của AI:** {(1-prob)*100:.2f}%")
                    st.info("Không phát hiện dấu hiệu bất thường trên ảnh và chỉ số sức khỏe.")