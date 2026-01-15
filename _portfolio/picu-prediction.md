---
title: "临床数据（PIC数据）分析和患者死亡风险预测"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/picu-prediction
date: 2026-01-14
excerpt: "本项目对临床PIC数据进行深度分析，运用逻辑回归和XGBoost模型预测患者死亡风险，通过调参优化模型，最终实现对新患者的预测并提供临床建议。"
header:
  teaser: /images/portfolio/picu-prediction/logistic_regression_visualization.png
tags:
  - 临床数据分析
  - 逻辑回归
  - XGBoost
  - 死亡风险预测
tech_stack:
  - name: Python
  - name: Scikit-learn
  - name: XGBoost
  - name: SHAP
---
# 项目背景 (Background)
在临床医疗领域，准确预测患者的死亡风险对于制定治疗方案和提高医疗质量具有重要意义。本项目基于临床 PIC 数据，通过数据分析和机器学习模型，旨在构建能够有效预测患者在医院死亡风险的模型，并为临床决策提供支持。
# 核心实现 (Implementation)
数据读取和初步分析
        
        import pandas as pd
        # 读取数据
        file_path = "E:/博一/python学习洪/data.xlsx"
        data = pd.read_excel(file_path)
        
        # 根据是否在医院死亡分组
        grouped = data.groupby('HOSPITAL_EXPIRE_FLAG')
        
        # 定义要分析的生理指标列
        lab_columns = ['lab_5237_min', 'lab_5227_min', 'lab_5225_range', 'lab_5235_max', 'lab_5257_min']
        
        # 检查是否有缺失值
        print(f"\n数据缺失情况：")
        print(data[lab_columns].isnull().sum())
        
        # 为每组计算统计量
        results = {}
        for group_value, group_data in grouped:
            group_name = f"死亡组" if group_value == 1 else f"存活组"
            print(f"\n{'='*50}")
            print(f"{group_name} (HOSPITAL_EXPIRE_FLAG = {group_value})")
            print(f"样本数量: {len(group_data)}")
        
            # 计算每个指标的统计量
            stats_dict = {}
            for col in lab_columns:
                # 移除缺失值
                values = group_data[col].dropna()
        
                if len(values) > 0:
                    stats = {
                        '平均值': np.mean(values),
                        '中位数': np.median(values),
                        '方差': np.var(values, ddof=0),  # 总体方差
                        '样本方差': np.var(values, ddof=1),  # 样本方差（无偏估计）
                        '标准差': np.std(values, ddof=1),
                    }
                    stats_dict[col] = stats
        
                    # 打印每个指标的统计结果
                    print(f"\n{col}:")
                    print(f"  平均值: {stats['平均值']:.4f}")
                    print(f"  中位数: {stats['中位数']:.4f}")
                    print(f"  方差: {stats['方差']:.4f}")
                else:
                    print(f"\n{col}: 无有效数据")
                    stats_dict[col] = None
            
此部分代码主要完成了数据的读取和初步的分组统计分析，通过按患者是否在医院死亡进行分组，计算了各个生理指标的统计量，为后续的分析和建模提供了基础。

# 逻辑回归模型

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc
        
        # 初始化模型
        model = LogisticRegressionModel(random_state=42)
        
        # 加载数据
        train_path = "E:/博一/python学习洪/PIC_split_data/train/PIC_train.xlsx"
        validation_path = "E:/博一/python学习洪/PIC_split_data/validation/PIC_validation.xlsx"
        test_path = "E:/博一/python学习洪/PIC_split_data/test/PIC_test.xlsx"
        model.load_data(train_path, validation_path, test_path)
        
        # 数据预处理
        model.preprocess_data()
        
        # 步骤1: 在训练集上训练，验证集上调整参数
        model.hyperparameter_tuning()
        
        # 步骤2: 使用最佳参数在训练集上重新训练模型
        model.train_model()
        
        # 步骤3: 在测试集上评估最终模型
        model.evaluate_on_test_set()
        
        # 创建性能汇总
        model.create_performance_summary()
        
        # 可视化结果
        model.visualize_results()
        
        # 保存结果
        model.save_results()
        
逻辑回归模型的实现过程包括数据加载、预处理、超参数调优、模型训练、评估和可视化等步骤。通过网格搜索在验证集上选择最佳参数，然后使用最佳参数在训练集上重新训练模型，最后在测试集上评估模型性能。

# XGBoost 模型

        import xgboost as xgb
        from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
        import shap
        
        # 加载数据
        train_path = "E:/博一/python学习洪/PIC_split_data/train/PIC_train.xlsx"
        val_path = "E:/博一/python学习洪/PIC_split_data/validation/PIC_validation.xlsx"
        test_path = "E:/博一/python学习洪/PIC_split_data/test/PIC_test.xlsx"
        train_data, val_data, test_data = load_data(train_path, val_path, test_path)
        
        # 数据预处理
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(train_data, val_data, test_data)
        
        # 训练和调参
        best_model, best_params = train_and_tune_xgb(X_train, y_train, X_val, y_val)
        
        # 在测试集上评估最终模型
        test_results = evaluate_model(best_model, X_test, y_test, "测试集")
        
        # 可视化结果
        feature_names = X_train.columns.tolist()
        plot_results(test_results, feature_names, best_model, X_test)
        
XGBoost 模型的实现同样包括数据加载、预处理、超参数调优、模型训练、评估和可视化等步骤。通过遍历参数网格选择最佳参数，使用最佳模型在测试集上进行评估，并对模型结果进行可视化展示。

# 分析结果 (Results & Analysis)
- **逻辑回归模型**
![逻辑回归模型可视化结果](/images/portfolio/picu-prediction/logistic_regression_visualization.png)
逻辑回归模型的可视化结果展示了模型在不同数据集上的性能指标对比、特征重要性、ROC 曲线等信息。从图中可以直观地看到模型在训练集、验证集和测试集上的表现，以及各个特征对模型预测的影响程度。
- **XGBoost模型**
![XGBoost模型可视化结果](/images/portfolio/picu-prediction/xgboost_visualization.png)
XGBoost 模型的可视化结果包括 ROC 曲线、PR 曲线、混淆矩阵图表。通过这些图表可以评估模型的性能。
- **XGBoost 详细 SHAP 图**
![XGBoost详细SHAP图](/images/portfolio/picu-prediction/xgboost_shap_detail.png)
SHAP 详细图展示了每个特征对模型预测结果的影响方向和程度，有助于深入理解模型的决策过程。
