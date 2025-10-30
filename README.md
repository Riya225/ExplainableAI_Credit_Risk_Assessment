# Credit Risk Prediction with Explainable AI

A beginner-friendly machine learning project that predicts whether a loan will be paid back or defaulted. Uses XGBoost for predictions and SHAP for explanations.

## 📋 Project Overview

This project helps predict credit risk (whether a borrower will default on a loan) using real data from LendingClub. It includes:
- **Machine Learning Model**: XGBoost classifier
- **Explainability**: SHAP values to understand why predictions are made
- **Web Interface**: Streamlit app for easy interaction
- **Public Dataset**: LendingClub loan data (2007-2018)

## 🎯 Objective

Develop a model with high predictive accuracy for credit risk and implement explainability methods to justify model decisions using the LendingClub dataset.

## 📁 Files in This Project

```
mINOR/
├── train_model.py          # Script to train the model
├── app.py                  # Streamlit web application
├── best_model.pkl          # Trained model (generated)
├── feature_names.pkl       # Feature list (generated)
├── best_threshold.pkl      # Optimal threshold (generated)
└── credit-risk-xai/
    └── data/
        └── accepted_2007_to_2018Q4.csv.gz  # Dataset
```

## 🚀 Getting Started

### Step 1: Install Required Packages

Open your terminal and run:

```bash
pip install streamlit pandas numpy scikit-learn xgboost shap
```

Or if you have a `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

Run the training script to create the model:

```bash
python train_model.py
```

This will:
- Load the data (10,000 samples)
- Create features from the raw data
- Train an XGBoost model
- Evaluate the model performance
- Save the model files

**Expected Output:**
- Training accuracy ~80-90%
- Test accuracy ~75-85%
- AUC score ~0.70-0.85

### Step 3: Run the Web App

Start the Streamlit app:

```bash
streamlit run app.py
```

This will open a web browser with the interactive interface where you can:
- Input loan details
- Get instant predictions
- See SHAP explanations

## 📊 How to Use the Web App

1. **Enter Loan Details:**
   - Loan Amount: How much the borrower wants
   - Interest Rate: Annual interest rate
   - Annual Income: Borrower's yearly income
   - Monthly Payment: The monthly installment
   - Loan Term: 36 or 60 months
   - Home Ownership: OWN, MORTGAGE, or RENT

2. **Get Prediction:**
   - Click "Predict Credit Risk"
   - See if it's a good or bad loan
   - View the default probability

3. **Understand the Prediction:**
   - See which features influenced the decision
   - SHAP values show feature importance
   - Positive values → bad loan
   - Negative values → good loan

## 🔍 Features Used

The model uses 9 features:

1. **loan_amnt** - Original loan amount
2. **int_rate** - Annual interest rate
3. **annual_inc** - Borrower's annual income
4. **installment** - Monthly payment amount
5. **loan_to_income** - Loan amount / Income (risk ratio)
6. **log_annual_inc** - Log of income (normalized)
7. **installment_to_income** - Monthly burden ratio
8. **term_numeric** - Loan term in months
9. **home_ownership_numeric** - Encoded home ownership

## 🎓 Understanding Key Concepts

### What is Credit Risk?
The risk that a borrower won't repay the loan. We predict this using historical data.

### What is XGBoost?
A powerful machine learning algorithm that combines many decision trees to make accurate predictions.

### What are SHAP Values?
SHAP (SHapley Additive exPlanations) tells you:
- Which features matter most
- How each feature pushes toward good or bad loan
- Why the model made a specific decision

### What is a Threshold?
The probability cutoff for making decisions:
- Low threshold (0.3) = lenient (more loans approved)
- High threshold (0.7) = strict (fewer loans approved)
- Optimal threshold = best balance (from training)

## 📈 Model Performance Metrics

The model reports several metrics:

- **Accuracy**: Percentage of correct predictions
- **AUC Score**: How well the model separates good and bad loans (0-1, higher is better)
- **F1 Score**: Balance between precision and recall
- **Confusion Matrix**: Shows correct and incorrect predictions

Example performance:
```
Training Accuracy: 85%
Test Accuracy:     80%
AUC Score:         0.75
F1 Score:          0.60
```

## 🐛 Troubleshooting

### Error: "Model files not found"
**Solution:** Run `python train_model.py` first to create the model files.

### Error: "Module not found"
**Solution:** Install missing packages:
```bash
pip install pandas numpy scikit-learn xgboost shap streamlit
```

### Low Accuracy
**Solution:** The model uses 10,000 samples. For better accuracy:
- Increase to 50,000+ samples in `train_model.py`
- Tune hyperparameters
- Add more features

### SHAP Not Working
**Solution:** Ensure SHAP is installed:
```bash
pip install shap
```

## 💡 Tips for Beginners

1. **Start Simple**: The model uses 9 features and 10k rows - easy to understand
2. **Read the Comments**: Each script has detailed comments explaining the code
3. **Experiment**: Try different loan scenarios in the web app
4. **Understand SHAP**: SHAP values show "why" - the most important part!
5. **Adjust Threshold**: Use the slider in the app to see how threshold affects predictions

## 📚 What You Can Learn From This Project

- Data preprocessing and feature engineering
- Training machine learning models
- Model evaluation and validation
- Explainable AI techniques (SHAP)
- Building interactive web apps
- Working with real-world financial data

## 🔄 Improving the Model

To make the model better, you can:

1. **Use More Data**: Change `nrows=10000` to `50000+` in train_model.py
2. **Add More Features**: Use more columns from the dataset
3. **Tune Hyperparameters**: Adjust XGBoost parameters
4. **Try Other Models**: Random Forest, Neural Networks, etc.
5. **Feature Selection**: Remove irrelevant features

## 📞 Need Help?

If you're stuck:
1. Check the error message carefully
2. Make sure all packages are installed
3. Verify the data file exists
4. Read the comments in the code

## 📄 License

This is an educational project using public LendingClub data.

## 🙏 Acknowledgments

- Dataset: LendingClub public loan data
- Libraries: scikit-learn, XGBoost, SHAP, Streamlit

---

**Made for beginners in Machine Learning and Explainable AI** 🚀

