"""
Risk Profile Prediction Node
Predicts risk profile and risk scores using ML model.
"""

import joblib
import os
<<<<<<< HEAD
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
=======
from sklearn.ensemble import RandomForestClassifier
>>>>>>> bedffafef0f7bda9b6501e9a959edb41aaefe771
from state import WorkflowState


def train_risk_model():
    """Train risk profile model using the available dataset"""

    model_path = "ml/models/risk_profile_model.pkl"
<<<<<<< HEAD
    encoder_path = "ml/models/label_encoders.pkl"

    if os.path.exists(model_path) and os.path.exists(encoder_path):
        print(f"✅ Risk model already exists at {model_path}")
        return True

    try:
        print("📖 Loading prospects dataset...")
        data = pd.read_csv("data/input_data/prospects.csv")

        if len(data) == 0:
            print("⚠️ Prospects dataset is empty")
            return False

        # Create target variable based on age and experience level
        # High risk: Young and inexperienced, Low risk: Experienced or older
        def assign_risk_level(row):
            age = row['age']
            experience = row['investment_experience_level']

            if age < 30 and experience == "Beginner":
                return "High"
            elif age > 50 and experience in ["Advanced", "Intermediate"]:
                return "Low"
            else:
                return "Moderate"

        data['risk_level'] = data.apply(assign_risk_level, axis=1)

        # Select features for training
        feature_columns = [
            'age', 'annual_income', 'current_savings',
            'target_goal_amount', 'investment_horizon_years',
            'number_of_dependents'
        ]

        # Encode categorical features
        label_encoders = {}
        X = data[feature_columns].copy()
        y = data['risk_level']

        # Encode target variable
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        label_encoders['target'] = le_target

        # Handle any missing values
        X = X.fillna(X.mean())

        # Train the model
        print("🔧 Training RandomForest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        model.fit(X, y_encoded)

        # Create models directory if it doesn't exist
        os.makedirs("ml/models", exist_ok=True)

        # Save model and encoders using pickle protocol 4 for compatibility
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, protocol=4)
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoders, f, protocol=4)

        print(f"✅ Risk profile model trained and saved to {model_path}")
        return True

    except FileNotFoundError:
        print("❌ prospects.csv not found in data/input_data/")
        return False
    except Exception as e:
        print(f"❌ Error training risk model: {str(e)}")
        return False
=======

    if os.path.exists(model_path):
        print(f"✅ Risk model already exists at {model_path}")
        return True

    print("⚠️ Risk profile model training data not available")
    print("Please provide training dataset in data/ folder")
    return False
>>>>>>> bedffafef0f7bda9b6501e9a959edb41aaefe771


async def predict_risk_profile(state: WorkflowState) -> WorkflowState:
    """Predict risk profile using ML model or rule-based fallback"""

    try:
        model_path = "ml/models/risk_profile_model.pkl"
        encoder_path = "ml/models/label_encoders.pkl"

        # Try to load trained model
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            model = joblib.load(model_path)
            label_encoder = joblib.load(encoder_path)

            print("🎯 ML Prediction: Risk profile loaded from model")

        else:
            # Train model if it doesn't exist
            print("🔄 Model not found, attempting to train...")
            if train_risk_model():
                # Retry prediction after training
                return await predict_risk_profile(state)
            else:
                # Fallback to rule-based prediction
                print("⚠️ Using rule-based fallback for risk prediction")

                # Simple rule-based risk assessment
                risk_score = 50  # Default medium risk
                if state.prospect.age < 30:
                    risk_score += 20
                elif state.prospect.age > 60:
                    risk_score -= 20

                if state.prospect.investment_experience_level == "Beginner":
                    risk_score -= 15
                elif state.prospect.investment_experience_level == "Expert":
                    risk_score += 15

                state.prospect.risk_score = max(0, min(100, risk_score))
                state.prospect.risk_level = "High" if risk_score > 70 else "Low" if risk_score < 30 else "Medium"

        state.current_step = "risk_assessed"

    except Exception as e:
        print(f"❌ Risk prediction error: {str(e)}")
        # Fallback prediction
        state.prospect.risk_score = 50
        state.prospect.risk_level = "Medium"
        state.current_step = "risk_assessed"

    return state


# Auto-train model if this file is run directly
if __name__ == "__main__":
    train_risk_model()
