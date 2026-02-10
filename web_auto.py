import os
import subprocess
import sys

# 检查是否有 pkg_resources，没有则强制安装 setuptools
try:
    import pkg_resources
except ImportError:
    print("pkg_resources not found. Installing setuptools...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"])
    import pkg_resources

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap

@st.cache_resource

def VSpace(px):
    """一个简单的函数，用于在 Streamlit 中创建指定像素的垂直空间"""
    st.markdown(f'<div style="margin-top: {px}px;"></div>', unsafe_allow_html=True)


# Load the trained AutoGluon model
# 模型路径：./Result_auto_DKD_s73_try6/
predictor = TabularPredictor.load('./DKD_model_WEB')  
best_model = "LightGBM_BAG_L1/T3_FULL"  # 最佳模型名称

# Define the feature options
Gender_options = {
    0: 'Female',  
    1: 'Male'  
}
Inj_Freq_options = {
    0: 'No Insulin Therapy',
    1: '1 injection/day',
    2: '2 injections/day',
    3: '≥ 3 injections/day' 
}

# Streamlit UI
st.title("Diabetic Kidney Disease (DKD) Risk Predictor")  

# 如果有图片，可以取消注释
# image = Image.open("Snipaste_2025-07-01_13-45-35.png")
# st.image(image)


# Sidebar for input options
st.sidebar.header("Input Patient Data")  # 侧边栏输入样本数据

Age = st.sidebar.number_input("Age:", min_value=18, max_value=100, value=70)
Gender = st.sidebar.selectbox("Gender:", options=list(Gender_options.keys()), format_func=lambda x: Gender_options[x])
DM_Duration = st.sidebar.number_input("DM Duration (years):", min_value=0.0, max_value=50.0, value=4.0, step=0.5)
Inj_Freq = st.sidebar.selectbox("Insulin Use Freq (Day):", options=list(Inj_Freq_options.keys()), format_func=lambda x: Inj_Freq_options[x])

st.sidebar.subheader("Laboratory Tests")
Glu = st.sidebar.number_input("Glucose (Glu, mmol/L):", min_value=0.0, max_value=30.0, value=4.49, step=0.1)
HbA1c = st.sidebar.number_input("HbA1c (%):", min_value=4.0, max_value=15.0, value=5.7, step=0.1)
Cr = st.sidebar.number_input("Creatinine (Cr, μmol/L):", min_value=0.0, max_value=500.0, value=49.1, step=1.0)
SBP = st.sidebar.number_input("Systolic BP (SBP, mmHg):", min_value=80, max_value=220, value=119)
TC = st.sidebar.number_input("Total Cholesterol (TC, mmol/L):", min_value=0.0, max_value=15.0, value=3.62, step=0.1)
LDL_C = st.sidebar.number_input("LDL-C (mmol/L):", min_value=0.0, max_value=10.0, value=3.80, step=0.1)
ALT = st.sidebar.number_input("ALT (U/L):", min_value=0.0, max_value=500.0, value=27.0, step=1.0)
AST = st.sidebar.number_input("AST (U/L):", min_value=0.0, max_value=500.0, value=23.0, step=1.0)
PLT = st.sidebar.number_input("Platelet (PLT, 10^9/L):", min_value=0.0, max_value=800.0, value=237.0, step=1.0)

# 添加一个 50 像素的垂直空白
VSpace(50)

st.subheader("Process the input and make a prediction")
# Process the input and make a prediction
# 注意：特征顺序需要与训练时一致
feature_values = [Age, int(Gender), DM_Duration, int(Inj_Freq), Glu, HbA1c, Cr, SBP, 
                  TC, LDL_C,  ALT, AST, PLT ]
feature_names = ["Age", "Gender", "DM_Duration", "Inj_Freq", "Glu", "HbA1c", "Cr", 
                 "SBP", "TC", "LDL_C",  "ALT", "AST", "PLT" ]
features = pd.DataFrame([feature_values], columns=feature_names) 

if st.button("Make Prediction"):  # 如果点击了预测按钮
    # Predict the class and probabilities using AutoGluon
    predicted_proba_df = predictor.predict_proba(features, model=best_model)
    predicted_proba = predicted_proba_df.values[0]  # [prob_class_0, prob_class_1]
    predicted_class = predictor.predict(features, model=best_model).values[0].astype(int)  # 预测结果

    # Display the prediction results
    st.write(f"**Predicted Class (0 = Non-DKD, 1 = DKD):** {predicted_class}")  # 显示预测类别
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  # 显示各类别的预测概率

    # Generate advice based on the prediction result
    probability = predicted_proba[predicted_class] * 100  # 根据预测类别获取对应的概率，并转化为百分比


    # Visualize the prediction probabilities
    sample_prob = {
        'No DKD': predicted_proba[0],  # DKD不发生的概率
        'DKD': predicted_proba[1]  # DKD发生的概率
    }
    
    VSpace(20)
    # Set figure size
    plt.figure(figsize=(8, 1))  # 设置图形大小
    plt.rc('ytick', labelsize=8) # 设置所有Y轴刻度的字体大小
    plt.rc('xtick', labelsize=8) # 设置所有X轴刻度的字体大小
    # Create bar chart
    bars = plt.barh(['No DKD', 'DKD'], 
                    [sample_prob['No DKD'], sample_prob['DKD']], 
                    height=0.6, edgecolor="black", color=['#81abd3','#fcd6d3'])  # 绘制水平条形图

    # Add title and labels, set font bold and increase font size
    plt.title("Prediction Probability for DKD", fontsize=9, fontweight='bold')  # 添加图表标题，并设置字体大小和加粗
    plt.xlabel("Probability", fontsize=7 )  # 添加X轴标签，并设置字体大小和加粗

    # Add probability text labels, adjust position to avoid overlap, set font bold
    for i, v in enumerate([sample_prob['No DKD'], sample_prob['DKD']]):  # 为每个条形图添加概率文本标签
        plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=6, color='black' )  # 设置标签位置、字体加粗

    # Hide other axes (top, right, bottom)
    plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
    plt.gca().spines['right'].set_visible(False)  # 隐藏右边框

    # Show the plot
    st.pyplot(plt, use_container_width=True)  # 显示图表
    

    if predicted_class == 1:  # 如果预测为DKD发生，给出相关建议
        advice = (
            f"**Recommendation:** According to our model, the probability of Diabetic Kidney Disease (DKD) is {probability:.1f}%, which is considered **High risk**. "
            f"We recommend you discuss these findings with your doctor or nephrologist as soon as possible to determine the next steps for kidney[Mam- protection and treatment."
        )  
    else:  # 如果预测为DKD低风险
        advice = (
            f"**Recommendation:** According to our model, the patient is at **low risk** for DKD. "
            f"The probability of **not developing DKD** is **{probability:.1f}%**. "
            "However, it is still important to continue regular monitoring of kidney function and blood glucose control. "
            "Please maintain good diabetes management and have regular check-ups."
        )  

    st.write(advice)  # 显示建议


    VSpace(50)

    st.subheader("Feature importance")
    import os
    import joblib

    # 尝试多种方式获取底层 模型
    model_estimator = None
    # 方法 1: 直接加载底层模型文件 (绕过 AutoGluon 包装器)
    # 路径基于目录结构: ./DKD_model_WEB/models/LightGBM_BAG_L1/T3_FULL/S1F1/model.pkl
    direct_model_path = os.path.join("./DKD_model_WEB", "models", "LightGBM_BAG_L1", "T3_FULL", "S1F1", "model.pkl")
    
    if os.path.exists(direct_model_path):
        loaded_obj = joblib.load(direct_model_path)
        # AutoGluon 的模型包装器通常把真实模型放在 .model 属性中
        if hasattr(loaded_obj, 'model'):
            model_estimator = loaded_obj.model
        else:
            model_estimator = loaded_obj

    # # 方法 2: 如果文件加载失败，尝试通过 predictor 获取
    # if model_estimator is None:
    #     model_obj = predictor._trainer.load_model(best_model)
    #     # 检查是否为 Bagged 模型，尝试提取第一个 fold
    #     if hasattr(model_obj, 'models') and model_obj.models:
    #         sub_model_name = model_obj.models[0]
    #         sub_model_obj = predictor._trainer.load_model(sub_model_name)
    #         if hasattr(sub_model_obj, 'model'):
    #             model_estimator = sub_model_obj.model
    #     # 普通模型
    #     elif hasattr(model_obj, 'model'):
    #         model_estimator = model_obj.model
    
    X_transformed = predictor.transform_features(features, model=best_model)
    explainer = shap.TreeExplainer(model_estimator)
    shap_values = explainer.shap_values(X_transformed)#features.values)
    shap_values = shap_values[1]

    if shap_values is not None:
        # 处理 expected_value
        if isinstance(explainer.expected_value, list):
            base_val = explainer.expected_value[1]
        else:
            base_val = explainer.expected_value

        # 1. Waterfall Plot
        # st.markdown("**1. Waterfall Plot**")
        # try:
        #     fig_waterfall = plt.figure(figsize=(6, 3)) 
        #     shap_exp = shap.Explanation(
        #         values=shap_values[0] if len(shap_values.shape) > 1 else shap_values,
        #         base_values=base_val,
        #         data=features.values[0],
        #         feature_names=features.columns.tolist()
        #     )
        #     shap.plots.waterfall(shap_exp, max_display=6, show=False)
            
        #     # 样式调整
        #     plt.tick_params(axis='x', labelsize=12)
        #     plt.tick_params(axis='y', labelsize=12)
        #     plt.savefig("shap_waterfall_plot.png", bbox_inches='tight')#, dpi=300)
        #     plt.close(fig_waterfall)
        #     st.image("shap_waterfall_plot.png", use_column_width=True)
            
        # except Exception as e:
        #     st.error(f"Waterfall plot failed: {str(e)}")

        # 2. Force Plot
        st.markdown("**Force Plot**")
        try:
            # Force Plot 需要 matplotlib=True
            fig_force = plt.figure()
            shap.plots.force(
                base_val,
                shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                features,
                matplotlib=True,
                link="logit",
                plot_cmap="viridis",
                show=False
            )
            plt.savefig("shap_force_plot.png", bbox_inches='tight')#, dpi=300)
            plt.close(fig_force)
            st.image("shap_force_plot.png", use_column_width=True)
            
        except Exception as e:
            st.error(f"Force plot failed: {str(e)}")


