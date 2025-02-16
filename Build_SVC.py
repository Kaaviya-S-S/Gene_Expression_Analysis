import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE


class SVC:
    """
    SVM Classifier
    Parameters: learning_rate, lambda_param, n_iters
    Attributes: learning_rate, lambda_param, weights, biases
    """
    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_param = lambda_param
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        # Training the model
        for _ in range(self.num_iterations):
            for idx, x_i in enumerate(X):
                # Compute the margin
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # If the condition is satisfied (correctly classified)
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)  # Regularization
                else:
                    # If the condition is not satisfied (misclassified)
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w-np.dot(x_i, y[idx]))  # Regularization
                    self.b -= self.learning_rate * y[idx]


    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.where(linear_output >= 0, 1, 0)


    def get_params(self, deep=True):     # Implement get_params
        """Get parameters for this estimator."""
        return {"learning_rate": self.learning_rate,
                "lambda_param": self.lambda_param,
                "num_iterations": self.num_iterations}

    def set_params(self, **parameters):  # Implement set_params
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


# Main function
if __name__ == "__main__":
    data = pd.read_csv("LUSCexpfile.csv", sep = ";", low_memory=False)
    data.info()

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
    print("Before SMOTE: \n", pd.Series(y_train).value_counts())
    print("After SMOTE: \n", pd.Series(y_train_resampled).value_counts())

    svc_model = SVC(learning_rate=0.001,  lambda_param=0.01, num_iterations=1000)

    svc_model.fit(X_train_resampled, y_train_resampled)
    print("\nModel is trained successfully!")

    # Model Evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5 folds
    scores = cross_val_score(svc_model, X_train_resampled, y_train_resampled, cv=skf, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {scores}")
    print("Mean cross-validation score:", np.mean(scores))

    with open("./models/classification.txt", "a") as file:
        file.write(f"svm: {np.mean(scores)}\n")
    file.close()

    # Save the model
    with open('models/svc_model.pkl', 'wb') as f:
        pickle.dump(svc_model, f)
    print("\nModel saved!")
