import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import datetime

# --- CONFIG ---
st.set_page_config(page_title="Used Bike Price Predictor Pro", layout="wide")

# --- 1. DATA LOADING & CLEANING (Dữ liệu thực tế) ---
@st.cache_data
def load_and_clean_data():
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv('gia_xe_may.csv')
    
    # Xử lý các khoảng trắng thừa ở dữ liệu dạng text (Data Cleaning)
    df['Place'] = df['Place'].str.strip()
    df['Hang_xe'] = df['Hang_xe'].str.strip()
    df['Dong_xe'] = df['Dong_xe'].str.strip()
    df['Phu_tung'] = df['Phu_tung'].str.strip()
    
    # Quy đổi đơn vị Giá sang Triệu VNĐ để dễ nhìn và model dễ học hơn
    df['Gia_Trieu'] = df['Gia'] / 1000000.0
    
    # Feature Engineering: Tính tuổi xe
    current_year = datetime.datetime.now().year
    df['Tuoi_Xe'] = current_year - df['Year']
    
    # Xóa bỏ các dòng có dữ liệu bị thiếu (nếu có)
    df = df.dropna()
    
    return df

# Tải dữ liệu
df = load_and_clean_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Bike ML Engine")
page = st.sidebar.selectbox("Chọn giai đoạn", ["1. EDA (Phân tích)", "2. ML Pipeline & Evaluation", "3. Demo Predictor"])

# --- PAGE 1: EDA ---
if page == "1. EDA (Phân tích)":
    st.title("📊 Phân tích Dữ liệu Khám phá (EDA)")
    st.write(f"Đang sử dụng dữ liệu thực tế: **{len(df)} mẫu xe**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phân phối giá bán (Triệu VNĐ)")
        fig, ax = plt.subplots()
        sns.histplot(df['Gia_Trieu'], kde=True, ax=ax, color='blue')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Giá vs Tình trạng phụ tùng")
        fig, ax = plt.subplots()
        sns.boxplot(x='Phu_tung', y='Gia_Trieu', data=df, ax=ax, palette="Set2")
        st.pyplot(fig)
    
    st.subheader("Tương quan giữa các biến số")
    fig, ax = plt.subplots(figsize=(10, 5))
    # Tính tương quan cho các biến dạng số
    sns.heatmap(df[['Year', 'Km_da_chay', 'Score', 'Gia_Trieu', 'Tuoi_Xe']].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# --- PAGE 2: ML PIPELINE ---
elif page == "2. ML Pipeline & Evaluation":
    st.title("⚙️ Machine Learning Pipeline (Linear Regression)")
    
    # Định nghĩa Features và Target
    features = ['Hang_xe', 'Dong_xe', 'Km_da_chay', 'Score', 'Phu_tung', 'Place', 'Tuoi_Xe']
    X = df[features]
    y = df['Gia_Trieu'] # Target là cột Giá đã quy đổi ra Triệu
    
    # Phân loại biến
    categorical_features = ['Hang_xe', 'Dong_xe', 'Phu_tung', 'Place']
    numeric_features = ['Km_da_chay', 'Score', 'Tuoi_Xe']
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Pipeline tích hợp
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Chia tập Train/Test và Huấn luyện
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Hiển thị chỉ số
    st.subheader("Kết quả đánh giá mô hình (Metrics)")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("MAE (Sai số tuyệt đối)", f"{mean_absolute_error(y_test, y_pred):.2f} Triệu")
    m_col2.metric("RMSE (Căn sai số BP)", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f} Triệu")
    m_col3.metric("R² Score (Độ khớp)", f"{r2_score(y_test, y_pred):.2f}")

    # Biểu đồ sai số
    st.subheader("Phân tích sai số (Thực tế vs Dự đoán)")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Giá Thực tế (Triệu)")
    plt.ylabel("Giá Dự đoán (Triệu)")
    st.pyplot(fig)

# --- PAGE 3: DEMO ---
else:
    st.title("🔮 Dự đoán giá xe bằng Data Thực tế")
    st.write("Mô hình đã được học từ file CSV của bạn. Hãy nhập thông số để thử nghiệm:")
    
    # Huấn luyện mô hình với TOÀN BỘ dữ liệu để ứng dụng vào thực tế
    features = ['Hang_xe', 'Dong_xe', 'Km_da_chay', 'Score', 'Phu_tung', 'Place', 'Tuoi_Xe']
    X = df[features]
    y = df['Gia_Trieu']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['Km_da_chay', 'Score', 'Tuoi_Xe']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Hang_xe', 'Dong_xe', 'Phu_tung', 'Place'])
        ])
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    model_pipeline.fit(X, y)
    
    # Lấy danh sách Hãng xe độc nhất từ CSV
    danh_sach_hang_xe = df['Hang_xe'].unique().tolist()
    
    # Chọn Hãng xe (Đưa ra ngoài st.form để update Dòng xe linh hoạt)
    brand = st.selectbox("Hãng xe", danh_sach_hang_xe)
    
    with st.form("input_form"):
        c1, c2 = st.columns(2)
        
        # Tự động lọc các Dòng xe thuộc về Hãng xe đã chọn ở trên
        danh_sach_dong_xe = df[df['Hang_xe'] == brand]['Dong_xe'].unique().tolist()
        model_name = c1.selectbox("Dòng xe", danh_sach_dong_xe)
        
        # Lấy min/max năm sản xuất từ CSV
        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())
        year = c1.slider("Năm sản xuất", min_year, max_year, min_year + int((max_year-min_year)/2))
        
        km = c1.number_input("Số KM đã đi", min_value=0, value=15000, step=1000)
        
        cond = c2.slider("Điểm chất lượng (Score)", 1, 10, 8)
        
        # Lấy danh sách trạng thái phụ tùng từ CSV
        danh_sach_phu_tung = df['Phu_tung'].unique().tolist()
        parts = c2.radio("Tình trạng phụ tùng", danh_sach_phu_tung, horizontal=True)
        
        # Lấy danh sách các Quận/Huyện từ CSV
        danh_sach_khu_vuc = sorted(df['Place'].unique().tolist())
        loc = c2.selectbox("Khu vực (Place)", danh_sach_khu_vuc)
        
        submitted = st.form_submit_button("Tính giá ngay")
        
        if submitted:
            current_year = datetime.datetime.now().year
            age = current_year - year
            input_df = pd.DataFrame([[brand, model_name, km, cond, parts, loc, age]], 
                                    columns=['Hang_xe', 'Dong_xe', 'Km_da_chay', 'Score', 'Phu_tung', 'Place', 'Tuoi_Xe'])
            
            res = model_pipeline.predict(input_df)[0]
            st.success(f"### Mức giá tham khảo: {res:.2f} Triệu VNĐ")
            st.caption("*(Lưu ý: Mức giá này được ước tính dựa trên dữ liệu từ file gia_xe_may.csv)*")
            st.balloons()