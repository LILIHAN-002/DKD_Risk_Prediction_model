# DKD prediction Model
# AutoGluon DKD Risk Predictor Web App

## 概述
基于 AutoGluon 训练的糖尿病肾病（DKD）风险预测 Web 应用。

## 模型信息
- **训练框架**: AutoGluon TabularPredictor
- **最佳模型**: LightGBM_BAG_L1/T3
- **模型路径**: `../autoglon/Result_auto_DKD_s73_try6/`

## 特征列表（共19个）
### 基本信息
1. Age (年龄)
2. Gender (性别): 0=女性, 1=男性
3. DM_Duration (糖尿病病程, 年)
4. Inj_Freq (胰岛素使用): 0=否, 1=是
5. BMI (体重指数)

### 实验室检查
6. Glu (空腹血糖, mmol/L)
7. HbA1c (糖化血红蛋白, %)
8. Cr (肌酐, μmol/L)
9. SBP (收缩压, mmHg)
10. DBP (舒张压, mmHg)
11. TC (总胆固醇, mmol/L)
12. LDL_C (低密度脂蛋白, mmol/L)
13. HDL_C (高密度脂蛋白, mmol/L)
14. TG (甘油三酯, mmol/L)
15. ALT (谷丙转氨酶, U/L)
16. AST (谷草转氨酶, U/L)
17. PLT (血小板, 10^9/L)
18. HGB (血红蛋白, g/L)
19. WBC (白细胞, 10^9/L)

## 安装依赖

```bash
pip install -r requirements_autogluon.txt
```

## 运行应用

```bash
streamlit run web_autogluon.py
```

## 重要说明

### 1. 模型路径配置
确保模型路径正确指向训练好的模型：
```python
predictor = TabularPredictor.load('../autoglon/Result_auto_DKD_s73_try6/')
```

### 2. 特征顺序
特征顺序必须与训练时保持一致：
```python
feature_names = ["Age", "Gender", "DM_Duration", "Inj_Freq", "Glu", "HbA1c", "Cr", 
                 "SBP", "DBP", "TC", "LDL_C", "HDL_C", "TG", "ALT", "AST", 
                 "PLT", "HGB", "WBC", "BMI"]
```

### 3. 类别特征
性别（Gender）和胰岛素使用情况（Inj_Freq）需要转换为字符串类型以匹配训练时的数据类型。

### 4. 预测结果
- 0 = Non-DKD (无糖尿病肾病)
- 1 = DKD (糖尿病肾病)

## 功能特性

### 1. 风险预测
- 输入患者临床数据
- 预测 DKD 发生风险
- 显示各类别的概率

### 2. 可视化
- 概率柱状图
- SHAP 特征重要性瀑布图

### 3. 临床建议
根据预测结果提供个性化的临床建议

## 故障排查

### 1. 模型加载失败
检查模型路径是否正确：
```bash
ls ../autoglon/Result_auto_DKD_s73_try6/
```

### 2. SHAP 解释失败
如果 SHAP 解释报错，程序会捕获异常并显示错误信息，不会影响预测功能。

### 3. 特征数量不匹配
确保输入的特征数量为 19 个，且顺序与训练时一致。

## 与原 PyCaret 版本的主要区别

1. **导入库**: 
   - 旧: `from pycaret.classification import *`
   - 新: `from autogluon.tabular import TabularPredictor`

2. **模型加载**:
   - 旧: `model = load_model('best_model')`
   - 新: `predictor = TabularPredictor.load('path')`

3. **预测方式**:
   - 旧: `predict_model(model, data=features)`
   - 新: `predictor.predict(features, model=best_model)`

4. **SHAP 解释**:
   - 旧: `model.named_steps['actual_estimator']`
   - 新: `predictor._trainer.load_model(best_model)`

## 作者与版权
本应用基于 autogluon_2C_v6.ipynb 训练的模型构建。

