from flask import Flask, jsonify, abort
from flask_cors import CORS
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data.questions import question_mapping

app = Flask(__name__)
CORS(app)

# Load the LOTR data
fileName = 'data/lotr_data.csv'
lotr_data = pd.read_csv(fileName, sep=';', header=0)

# Extract features and target variable
target_column = 2
target = lotr_data.iloc[:, target_column].values

# Exclude the target column from features
features = lotr_data.drop(columns=[lotr_data.columns[target_column]])

# One-hot encode categorical columns
features = pd.get_dummies(features)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=100)

# Replace DecisionTreeClassifier with RandomForestClassifier
clf_random_forest = RandomForestClassifier(n_estimators=100, random_state=100)
clf_random_forest.fit(X_train, y_train)

y_pred = clf_random_forest.predict(X_test)

# After predicting y_pred with your model
try:
    print(classification_report(y_test, y_pred))

except Exception as e:
    print("Error calculating metrics:", e)

feature_importance = clf_random_forest.feature_importances_
print("Feature Importance:")
for feature, importance in zip(features.columns, feature_importance):
    print(f"{feature}: {importance}")

accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy * 100)

for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted}")

def ask_question():
    # Pick a random question
    random_question_id = random.choice(list(question_mapping.keys()))
    random_question_info = question_mapping.get(random_question_id, None)

    if random_question_info:
        question = random_question_info["question"]

        return {
            "question_id": random_question_id,
            "question": question,
        }
    else:
        return {"error": "Question not found in the mapping."}

@app.route('/')
def index():
    return 'Flask is running!'

@app.route('/ask_question', methods=['GET', 'POST'])
def get_question():
    try:
        question_info = ask_question()
        if not question_info:
            raise ValueError("No questions available.")

        return jsonify(question_info)
    except ValueError as e:
        print(f"Error getting question: {e}")
        abort(500)

if __name__ == '__main__':
    app.run(debug=True)
