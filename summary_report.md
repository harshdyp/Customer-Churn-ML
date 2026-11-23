# Executive Summary Report: Customer Churn Prediction

## 1. Business Problem and Dataset Summary
Customer churn poses a major financial challenge for telecom operators like Jio and Airtel. Because Customer Acquisition Cost (CAC) is five to seven times higher than the cost of retention, losing paying subscribers directly damages Customer Lifetime Value (CLV). Predicting churn allows the business to intervene before subscribers leave.

The project utilizes the IBM Telco Customer Churn dataset, containing 7,043 subscriber records across 21 demographic, service, and account attributes. The target is binary (`Churn`: Yes or No). The dataset exhibits moderate class imbalance: 73.46% of subscribers stayed, while 26.54% churned (~3:1 ratio). In addition, 11 customer records contained blank whitespace entries in `TotalCharges` corresponding to brand new subscribers with zero tenure; these were converted to numeric values and imputed using the median (₹1,397.48).

## 2. Preprocessing and Class Imbalance Strategy
Data preprocessing followed a rigorous, leak-free protocol:
* Dropped the non-predictive `customerID` column.
* Engineered three business features: `tenure_group` (ordinal bins: New, Mid, Senior, Loyal), `num_services` (count of 6 subscribed add-on services), and `AutoPay` (binary flag for automatic payments).
* Encoded categorical attributes: binary features mapped to 0/1, `tenure_group` label-encoded to preserve order, and multi-class nominal features one-hot encoded.
* Applied an 80/20 stratified train-test split (`random_state=42`) to maintain equal class proportions.
* Normalized continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`, `num_services`) using `StandardScaler` fitted strictly on training data.
* Handled class imbalance using SMOTE exclusively on the training split (`X_train`, `y_train`), balancing classes to 4,139 samples each. The test set remained completely unaugmented to ensure realistic evaluation.

## 3. Model Recommendation and Metric Evidence
Four supervised learning classifiers were evaluated on the 1,409 holdout test customers: K-Nearest Neighbours, Gaussian Naive Bayes, Support Vector Machine (RBF kernel), and Decision Tree Classifier.

Because failing to detect a churner (False Negative) loses all future customer revenue, **Recall for the Churn class** is the primary metric. While KNN achieved the highest raw Recall (80.75%), it suffered from lower Precision (48.17%) and high latency. 

We recommend deploying the **Decision Tree Classifier (depth=7)**. It achieves a strong Recall of **76.20%**, the best Precision (**52.49%**), the highest F1-Score (**0.6216**), and the highest overall Accuracy (**75.37%**). Crucially, Decision Trees provide clear, human-interpretable rules that frontline retention agents can explain, along with sub-second inference speed (< 0.02s) suitable for production systems.

## 4. Top 3 Churn Signals and Retention Actions
1. **Month-to-Month Contracts:** Churn rate reaches 42.71% for month-to-month contracts vs under 11% for annual plans.  
   * *Action:* Offer targeted discounts (e.g., 15% off) to migrate month-to-month users onto annual contracts.
2. **First-Year Tenure Risk:** Customers in their first 12 months churn at 47.44%.  
   * *Action:* Introduce a proactive 90-day onboarding program with customer check-in calls.
3. **High Monthly Charges on Fiber Optic:** High bills without bundled security drive churn.  
   * *Action:* Bundle free tech support and security add-ons to enhance perceived plan value.

## 5. Next Steps
Future iterations should incorporate live call-drop telemetry, explore gradient boosting models (LightGBM/XGBoost) for enhanced boundary precision, and deploy the pipeline as a containerized REST API for real-time churn scoring.
