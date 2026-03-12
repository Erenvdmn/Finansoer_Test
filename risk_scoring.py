import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from feature_engineering import FeatureEngineer


class RiskScorer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.features = ['Daily_Return', 'Volatility', 'RSI_14', 'MACD', 'SMA_20', 'SMA_50']
        self.target = 'Target_Downside'

    
    def prepare_data(self):
        print("Datas pulling and XGBoost preparing...")
        engineer = FeatureEngineer(self.filepath)
        df = engineer.get_processed_data()

        train_df = df.dropna(subset=[self.target])
        latest_data = df[df[self.target].isna()].copy()

        X = train_df[self.features]
        y = train_df[self.target]

        return X, y, latest_data, df.index[-1]
    

    def train_and_explain(self):
        X, y, latest_data, latest_date = self.prepare_data()

        print("XGBoost Model is training...")

        base_model = xgb.XGBClassifier(
            n_estimators = 100,
            learning_rate = 0.05,
            max_depth=4,
            random_state = 42,
            eval_metric = 'logloss'
        )

        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
        calibrated_model.fit(X, y)
        print("Model Calibiraiton is complated!")

        if not latest_data.empty:
            X_latest = latest_data[self.features].iloc[[-1]]

            risk_prob = calibrated_model.predict_proba(X_latest)[0][1]
            print(f"{latest_date} Risk Analyze:")
            print(f"Posibility of Sharp Decline Risk Score: %{risk_prob * 100:.2f}")

            print("SHAP Description (Factors that effects Risk Score) preparing...")

            trained_xgb = calibrated_model.calibrated_classifiers_[0].estimator
            explainer = shap.TreeExplainer(trained_xgb)
            shap_values = explainer.shap_values(X_latest)

            print("-" * 40)
            for feature_name, shap_val, actual_val in zip(self.features, shap_values[0], X_latest.values[0]):
                effect_direction = "🔺 Increase Risk" if shap_val > 0 else "🔻 Decrease Risk"
                print(f"{feature_name:<15} | Value: {actual_val:>8.4f} | Effect: {effect_direction} (Power: {abs(shap_val):.4f})")
            print("-" * 40)

            shap.force_plot(explainer.expected_value, shap_values[0], X_latest, matplotlib=True, show=False)
            plt.savefig("shap_explanaiton.png", bbox_inches='tight')
            print("SHAP image saved as 'shap_explanaiton.png'")

            return risk_prob * 100
        # If there is no current data
        return None

# for manuel testing of this file
if __name__=="__main__":
    scorer = RiskScorer("data/AAPL_daily.csv")
    scorer.train_and_explain()