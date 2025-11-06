# House Prices Prediction - Kaggle Competition

A comprehensive machine learning solution for predicting residential home sale prices in Ames, Iowa. This project demonstrates advanced feature engineering, model selection, and ensemble techniques to achieve competitive performance on the Kaggle leaderboard.

## 🎯 Project Overview

- **Competition**: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Evaluation Metric**: Root Mean Squared Logarithmic Error (RMSLE)
- **Final Model**: Stacking Regressor with multiple base learners
- **Cross-Validation Score**: RMSE ~0.12 (on log-transformed target)

## 🔑 Key Features

### Feature Engineering
- **Ordinal Encoding**: Applied to 21 quality/condition features with inherent ordering
- **One-Hot Encoding**: Used for low-cardinality nominal features
- **Feature Selection**: Mutual information-based selection keeping top 50% of features
- **Target Transformation**: Log transformation to normalize distribution and align with RMSLE metric

### Models Explored
- Linear Models (Ridge Regression with hyperparameter tuning)
- K-Nearest Neighbors
- Support Vector Regression (Linear & RBF kernels)
- Tree-based Models (Decision Tree, Random Forest)
- Gradient Boosting (AdaBoost, Gradient Boosting, XGBoost)
- **Ensemble Methods** (Voting & Stacking Regressors) ⭐

## 📊 Results

| Model | CV Score (RMSE) | Notes |
|-------|----------------|-------|
| Decision Tree (Baseline) | 0.2108 | Initial benchmark |
| Ridge Regression | 0.1397 | Best linear model |
| Random Forest | 0.1628 | Solid tree-based performance |
| Gradient Boosting | 0.1270 | Strong boosting performance |
| XGBoost | 0.1474 | With early stopping |
| **Stacking Regressor** | **0.12336** | **Best overall performance** |

## 🛠️ Technical Stack

- **Python 3.12+**
- **Core Libraries**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Advanced Models**: XGBoost
- **Feature Engineering**: Custom preprocessing pipelines

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis

1. Clone this repository:
```bash
git clone https://github.com/yourusername/house-prices-prediction.git
cd house-prices-prediction
```

2. Open and run the Jupyter notebook:
```bash
jupyter notebook notebook.ipynb
```

3. The notebook will:
   - Load data from the provided URLs
   - Perform comprehensive feature engineering
   - Train and evaluate multiple models
   - Generate predictions for submission

## 📁 Project Structure
```
├── notebook.ipynb          # Main analysis notebook
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── submission_final.csv   # Final predictions (generated)
└── .gitignore            # Git ignore rules
```

## 🎓 Key Learnings

1. **Feature Engineering is Critical**: Proper encoding of ordinal features and feature selection improved performance significantly
2. **Ensemble Methods Win**: Stacking multiple diverse models outperformed individual algorithms
3. **Target Transformation Matters**: Log transformation aligned with the evaluation metric and stabilized predictions
4. **Cross-Validation is Essential**: Reliable performance estimation prevented overfitting

## 📈 Model Pipeline
```
Raw Data
   ↓
Feature Engineering (Ordinal/One-Hot/Imputation)
   ↓
Feature Selection (Mutual Information)
   ↓
Model Training (Stacking Ensemble)
   ↓
Predictions (Exponential Transform)
   ↓
Submission
```

## 🔮 Future Improvements

- [ ] Experiment with deep learning approaches (Neural Networks)
- [ ] Implement automated feature engineering (Featuretools)
- [ ] Add more sophisticated feature interactions
- [ ] Optimize ensemble weights through meta-learning
- [ ] Deploy model as a web service

## 📫 Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your Name](https://linkedin.com/in/yourprofile)
- **Email**: your.email@example.com

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Kaggle for hosting the competition
- Dean De Cock for creating the dataset
- The scikit-learn and XGBoost communities

---

⭐ If you found this project helpful, please consider giving it a star!
