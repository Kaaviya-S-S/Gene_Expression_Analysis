import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA)
    Parameters: n_components
    Attributes: n_components, components, mean
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0).astype(np.float64) 
        X_centered = X - self.mean

        covariance_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.components = eigenvectors[:, :self.n_components]


    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)


    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

    def get_params(self, deep=True):
        return {"n_components": self.n_components}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    