import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import joblib
import os
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Ẩn các cảnh báo thừa
import tensorflow as tf
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
        if col in ['Sex', 'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack','PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'DiffWalk']:
            user_inputs[col] = st.selectbox(f"{col} (0: Không, 1: Có)", [0, 1])
        elif col == 'BMI':
            user_inputs[col] = st.slider("Chỉ số BMI",10,60,25)
        elif col == 'Age':
            user_inputs[col] = st.slider(f"Nhóm tuổi (1-13)", 1, 13, 5)
        elif col == 'Education':
            user_inputs[col] = st.slider(f"Trình độ học vấn (1-6)", 1, 6, 3)    
        elif col == 'Income':
            user_inputs[col] = st.slider(f"Thu nhập (1-8)", 1, 8, 4)        
        elif col == 'PhysHlth':
            user_inputs[col] = st.slider(f"Số ngày sức khỏe thể chất không tốt trong 30 ngày qua", 0, 30, 5)
        elif col == 'MentHlth':
            user_inputs[col] = st.slider(f"Số ngày sức khỏe tinh thần không tốt trong 30 ngày qua", 0, 30, 5)   
            
    

with col2:
    st.subheader("Phân tích Ảnh Võng mạc (Không bắt buộc)")
    uploaded_file = st.file_uploader("Tải lên ảnh đáy mắt...", type=["jpg", "jpeg", "png"])
    
    # Hiển thị ảnh nếu có
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Ảnh đã tải lên', width=350)
    else:
        st.info("Chưa có ảnh. Hệ thống sẽ phân tích dựa trên chỉ số sức khỏe.")

    # Nút bấm nằm ngoài điều kiện if uploaded_file
    if st.button("BẮT ĐẦU PHÂN TÍCH"):
        with st.spinner('Đang xử lý dữ liệu...'):
            # --- XỬ LÝ ẢNH ---
            if uploaded_file is not None:
                # Nếu có ảnh: Xử lý ảnh thật
                img_array = np.array(img.convert('RGB'))
                img_input = cv2.resize(img_array, (224, 224))
            else:
                # Nếu KHÔNG có ảnh: Tạo ảnh đen (Dummy Input)
                img_input = np.zeros((224, 224, 3), dtype=np.float32)
            
            img_input = img_input / 255.0
            img_input = np.expand_dims(img_input, axis=0)

            # --- XỬ LÝ DỮ LIỆU BẢNG ---
            df_input = pd.DataFrame([user_inputs])
            tab_input = scaler.transform(df_input)

            # --- DỰ ĐOÁN ---
            prediction = model.predict({
                "image_input": img_input, 
                "tabular_input": tab_input
            })
            prob = prediction[0][0]

            # --- HIỂN THỊ KẾT QUẢ ---
            st.divider()
            if prob > 0.5:
                st.error(f"### KẾT QUẢ: CÓ DẤU HIỆU BỆNH")
                st.write(f"**Độ tin cậy:** {prob*100:.2f}%")
            else:
                st.success(f"### KẾT QUẢ: BÌNH THƯỜNG")
                st.write(f"**Độ tin cậy:** {(1-prob)*100:.2f}%")

