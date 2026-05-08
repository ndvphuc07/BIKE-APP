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

# --- 1. DATA GENERATION & CLEANING (Simulated) ---
@st.cache_data
def load_and_clean_data():
    np.random.seed(42)
    n = 1000
    brands = ['Honda', 'Yamaha']
    models = {'Honda': ['Vision', 'Wave', 'Air Blade', 'SH'], 'Yamaha': ['Sirius', 'Exciter', 'Grande']}
    
    data = []
    for _ in range(n):
        brand = np.random.choice(brands)
        model = np.random.choice(models[brand])
        year = np.random.randint(2010, 2024)
        km = np.random.randint(500, 100000)
        condition = np.random.randint(1, 11)
        location = np.random.choice(['Hà Nội', 'TP.HCM', 'Đà Nẵng'])
        parts_replaced = np.random.choice(['Có', 'Chưa']) # Đã bổ sung biến thay phụ tùng
        
        # Base price logic
        price = 20 if brand == 'Honda' else 18
        price += (year - 2010) * 2 - (km / 5000) + (condition * 1.5)
        
        # Trừ giá nếu đã thay phụ tùng
        if parts_replaced == 'Có':
            price -= 2.0 
            
        price += np.random.normal(0, 3)
        data.append([brand, model, year, km, condition, parts_replaced, location, max(5, price)])

    df = pd.DataFrame(data, columns=['Brand', 'Model', 'Year', 'KM', 'Condition', 'Parts_Replaced', 'Location', 'Price'])
    
    # Feature Engineering: Tính tuổi xe
    current_year = datetime.datetime.now().year
    df['Bike_Age'] = current_year - df['Year']
    
    return df

df = load_and_clean_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Bike ML Engine")
page = st.sidebar.selectbox("Chọn giai đoạn", ["1. EDA (Phân tích)", "2. ML Pipeline & Evaluation", "3. Demo Predictor"])

# --- PAGE 1: EDA ---
if page == "1. EDA (Phân tích)":
    st.title("📊 Exploratory Data Analysis (EDA)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phân phối giá bán")
        fig, ax = plt.subplots()
        sns.histplot(df['Price'], kde=True, ax=ax, color='blue')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Giá vs Tình trạng thay phụ tùng")
        fig, ax = plt.subplots()
        sns.boxplot(x='Parts_Replaced', y='Price', data=df, ax=ax, palette="Set2")
        st.pyplot(fig)
    
    st.subheader("Tương quan giữa các biến số")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# --- PAGE 2: ML PIPELINE ---
elif page == "2. ML Pipeline & Evaluation":
    st.title("⚙️ Machine Learning Pipeline")
    
    # Feature Selection (Bổ sung Parts_Replaced)
    features = ['Brand', 'Model', 'KM', 'Condition', 'Parts_Replaced', 'Location', 'Bike_Age']
    X = df[features]
    y = df['Price']
    
    # Preprocessing Pipeline
    categorical_features = ['Brand', 'Model', 'Parts_Replaced', 'Location']
    numeric_features = ['KM', 'Condition', 'Bike_Age']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Model Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Split & Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    st.subheader("Kết quả đánh giá mô hình (Metrics)")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("MAE (Sai số tuyệt đối)", f"{mean_absolute_error(y_test, y_pred):.2f} Tr")
    m_col2.metric("RMSE (Căn sai số bình phương)", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f} Tr")
    m_col3.metric("R² Score (Độ khớp)", f"{r2_score(y_test, y_pred):.2f}")

    # Residual Plot
    st.subheader("Phân tích sai số (Residual Plot)")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    st.pyplot(fig)

# --- PAGE 3: DEMO ---
else:
    st.title("🔮 Dự đoán giá xe theo thời gian thực")
    st.write("Nhập thông số xe của bạn để hệ thống định giá:")
    
    # Re-train model for prediction (using full pipeline)
    features = ['Brand', 'Model', 'KM', 'Condition', 'Parts_Replaced', 'Location', 'Bike_Age']
    X = df[features]
    y = df['Price']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['KM', 'Condition', 'Bike_Age']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Brand', 'Model', 'Parts_Replaced', 'Location'])
        ])
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    model_pipeline.fit(X, y)
    
    # TỪ ĐIỂN MAP HÃNG XE VÀ DÒNG XE
    models_dict = {
        'Honda': ['Vision', 'Wave', 'Air Blade', 'SH'], 
        'Yamaha': ['Sirius', 'Exciter', 'Grande']
    }

    # FIX LỖI: Đưa "Hãng xe" ra khỏi st.form để nó cập nhật UI ngay lập tức khi thay đổi
    brand = st.selectbox("Hãng xe", ['Honda', 'Yamaha'])
    
    with st.form("input_form"):
        c1, c2 = st.columns(2)
        
        # Dòng xe giờ đây sẽ gọi list dựa theo khóa của dict ở trên
        model_name = c1.selectbox("Dòng xe", models_dict[brand])
        
        year = c1.slider("Năm sản xuất", 2010, 2024, 2020)
        km = c1.number_input("Số KM đã đi", min_value=0, value=15000)
        
        cond = c2.slider("Tình trạng ngoại hình (1-10)", 1, 10, 8)
        parts = c2.radio("Xe đã thay phụ tùng chưa?", ['Chưa', 'Có'], horizontal=True)
        loc = c2.selectbox("Khu vực", ['Hà Nội', 'TP.HCM', 'Đà Nẵng'])
        
        submitted = st.form_submit_button("Tính giá ngay")
        
        if submitted:
            age = 2024 - year
            input_df = pd.DataFrame([[brand, model_name, km, cond, parts, loc, age]], 
                                    columns=['Brand', 'Model', 'KM', 'Condition', 'Parts_Replaced', 'Location', 'Bike_Age'])
            res = model_pipeline.predict(input_df)[0]
            st.success(f"### Giá đề xuất: {res:.2f} Triệu VNĐ")
            if parts == 'Có':
                st.info("💡 Lưu ý: Mức giá đã được điều chỉnh giảm do xe đã từng thay thế phụ tùng.")
            st.balloons()