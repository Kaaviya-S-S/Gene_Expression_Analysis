import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE


class LogisticRegression:
    """
    Logistic Regression Classifier
    Parameters: learning_rate, num_iterations, threshold
    Attributes: learning_rate, num_iterations, threshold, weights, biases
    """
    def __init__(self, learning_rate=0.0001, num_iterations=100, threshold=0.5):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.threshold = threshold
        self.w = None
        self.b = None

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))


    def fit(self, X, y):
        # Initialize weights
        self.w = np.zeros(X.shape[1])
        self.b = 0
        m = X.shape[0]

        #Gradient Descent
        for _ in range(self.num_iterations):
            z = np.dot(X, self.w) + self.b
            A = self.sigmoid(z)
            #cost = - (1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
            dw = (1 / m) * np.dot(X.T, (A - y))
            db = (1 / m) * np.sum(A - y)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db


    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        A = self.sigmoid(z)
        y_pred = np.where(A > self.threshold, 1, 0)
        return y_pred


    def get_params(self, deep=True):     # Implement get_params
        """Get parameters for this estimator."""
        return {"learning_rate": self.learning_rate,
                "num_iterations": self.num_iterations,
                "threshold": self.threshold}

    def set_params(self, **parameters):  # Implement set_params
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self



#Main function
if __name__ == "__main__":

    data = pd.read_csv("LUSCexpfile.csv", sep = ";", low_memory=False)
    data.info()

    data = data.T
    data.columns = data.iloc[0]
    data = data[1:]
    data = data.rename(columns = {np.nan:'Class'})
    data.columns.name = None

    #Preprocessing
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


    #Create a model
    model = LogisticRegression(learning_rate=0.00001, num_iterations=100, threshold=0.4)

    model.fit(X_train_resampled, y_train_resampled)
    print("\nModel is trained successfully!")

    #Evaluate the model
    scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
    print(f"\nCross-Validation Accuracy Scores: {scores}")
    print("Mean cross-validation score:", np.mean(scores))

    with open("./models/classification.txt", "a") as file:
        file.write(f"logistic: {np.mean(scores)}\n")
    file.close()

    # Save the model
    with open('models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved!")
