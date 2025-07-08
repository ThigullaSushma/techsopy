import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    df = pd.read_csv("data/sample_user_logs.csv")
    X = df[['sleep_hours', 'work_hours', 'screen_time', 'exercise_minutes', 'mood_score']]
    y = df['stress_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, 'models/stress_predictor.pkl')
    print("✅ Model trained and saved.")

if __name__ == "__main__":
    train_model()
