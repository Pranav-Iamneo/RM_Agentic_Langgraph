"""
Goal Success Prediction Node
Predicts probability of goal success using ML model.
"""

import joblib
import os
<<<<<<< HEAD
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
=======
from sklearn.ensemble import RandomForestClassifier
>>>>>>> bedffafef0f7bda9b6501e9a959edb41aaefe771
from state import WorkflowState


def train_goal_model():
    """Train goal success prediction model using the available dataset"""

    model_path = "ml/models/goal_success_model.pkl"
<<<<<<< HEAD
    encoder_path = "ml/models/goal_success_label_encoders.pkl"

    if os.path.exists(model_path) and os.path.exists(encoder_path):
        print(f"✅ Goal success model already exists at {model_path}")
        return True

    try:
        print("📖 Loading prospects dataset...")
        data = pd.read_csv("data/input_data/prospects.csv")

        if len(data) == 0:
            print("⚠️ Prospects dataset is empty")
            return False

        # Create target variable based on goal feasibility
        # Success: Good savings rate and long horizon, Failure: Low savings and short horizon
        def assess_goal_success(row):
            savings_ratio = row['current_savings'] / row['target_goal_amount']
            horizon = row['investment_horizon_years']

            if savings_ratio > 0.4 and horizon >= 10:
                return "Achievable"
            elif savings_ratio < 0.2 and horizon < 5:
                return "Challenging"
            else:
                return "Moderate"

        data['goal_success'] = data.apply(assess_goal_success, axis=1)

        # Select features for training
        feature_columns = [
            'age', 'annual_income', 'current_savings',
            'target_goal_amount', 'investment_horizon_years',
            'number_of_dependents'
        ]

        # Encode categorical features
        label_encoders = {}
        X = data[feature_columns].copy()
        y = data['goal_success']

        # Encode target variable
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        label_encoders['target'] = le_target

        # Handle any missing values
        X = X.fillna(X.mean())

        # Train the model
        print("🔧 Training RandomForest model for goal success...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        model.fit(X, y_encoded)

        # Create models directory if it doesn't exist
        os.makedirs("ml/models", exist_ok=True)

        # Save model and encoders using pickle protocol 4 for compatibility
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, protocol=4)
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoders, f, protocol=4)

        print(f"✅ Goal success model trained and saved to {model_path}")
        return True

    except FileNotFoundError:
        print("❌ prospects.csv not found in data/input_data/")
        return False
    except Exception as e:
        print(f"❌ Error training goal success model: {str(e)}")
        return False
=======

    if os.path.exists(model_path):
        print(f"✅ Goal success model already exists at {model_path}")
        return True

    print("⚠️ Goal success model training data not available")
    print("Please provide training dataset in data/ folder")
    return False
>>>>>>> bedffafef0f7bda9b6501e9a959edb41aaefe771


async def predict_goal_success(state: WorkflowState) -> WorkflowState:
    """Predict probability of achieving investment goal using ML model or rule-based fallback"""

    try:
        model_path = "ml/models/goal_success_model.pkl"
        encoder_path = "ml/models/goal_success_label_encoders.pkl"

        # Try to load trained model
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            model = joblib.load(model_path)
            label_encoder = joblib.load(encoder_path)

            print("🎯 ML Prediction: Goal success probability loaded from model")

        else:
            # Train model if it doesn't exist
            print("🔄 Model not found, attempting to train...")
            if train_goal_model():
                # Retry prediction after training
                return await predict_goal_success(state)
            else:
                # Fallback to rule-based prediction
                print("⚠️ Using rule-based fallback for goal success prediction")

                # Simple rule-based goal success assessment
                success_probability = 0.5  # Default 50%

                # Adjust based on investment horizon
                if state.prospect.investment_horizon_years >= 10:
                    success_probability += 0.25
                elif state.prospect.investment_horizon_years >= 5:
                    success_probability += 0.15
                elif state.prospect.investment_horizon_years < 2:
                    success_probability -= 0.20

                # Adjust based on savings rate
                if state.prospect.annual_income > 0:
                    savings_rate = state.prospect.current_savings / (state.prospect.annual_income * state.prospect.investment_horizon_years + 0.01)
                    if savings_rate > 0.3:
                        success_probability += 0.15
                    elif savings_rate < 0.1:
                        success_probability -= 0.15

                state.prospect.goal_success_probability = max(0.0, min(1.0, success_probability))

        state.current_step = "goal_assessed"

    except Exception as e:
        print(f"❌ Goal success prediction error: {str(e)}")
        # Fallback prediction
        state.prospect.goal_success_probability = 0.5
        state.current_step = "goal_assessed"

    return state


# Auto-train model if this file is run directly
if __name__ == "__main__":
    train_goal_model()
