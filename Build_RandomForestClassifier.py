import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier as SklearnRF

class RandomForestClassifier:
    def __init__(self, n_estimators=100, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = SklearnRF(n_estimators=self.n_estimators, random_state=self.random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"n_estimators": self.n_estimators,
                "random_state": self.random_state }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for param, value in params.items():
            setattr(self, param, value)
        # Update the internal model with new parameters
        self.model = SklearnRF(n_estimators=self.n_estimators, random_state=self.random_state)
        return self



#Main function
if __name__ == "__main__":
    data = pd.read_csv("LUSCexpfile.csv", sep = ";", low_memory=False)

    data = data.T
    data.columns = data.iloc[0]
    data = data[1:]
    data = data.rename(columns = {np.nan:'Class'})
    data.columns.name = None

    X = data.drop("Class", axis=1)  # Independent variables
    y = data["Class"]               # Dependent variable

    # Standardizing the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(np.float32))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    # Apply SMOTE for oversampling the minority class
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Initialize LabelEncoder for the categorical labels
    label_encoder = LabelEncoder()
    y_train_resampled = label_encoder.fit_transform(y_train_resampled)
    y_test = label_encoder.transform(y_test)

    # Verify the new class distribution
    print("\nBefore SMOTE: \n", pd.Series(y_train).value_counts())
    print("After SMOTE: \n", pd.Series(y_train_resampled).value_counts())


    # Initialize RandomForestClassifier
    rfc_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc_model.fit(X_train_resampled, y_train_resampled)

    #Perform Cross Validation
    scores = cross_val_score(rfc_model, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
    print(f"\nCross-Validation Accuracy Scores: {scores}")
    print("Mean cross-validation score:", np.mean(scores))

    with open("./models/classification.txt", "a") as file:
        file.write(f"random_forest: {np.mean(scores)}\n")
    file.close()

    # save the model
    with open('models/random_forest_classifier.pkl', 'wb') as f:
        pickle.dump(rfc_model, f)
    print(f"\nModel saved as models/random_forest_classifier.pkl")
    