import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
from skopt.space import Real
from sklearn.base import BaseEstimator, RegressorMixin

# Load the diabetes dataset as an example
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter search space (learning rate)
param_space = {
    'learning_rate': Real(1e-5, 1e-1, prior='log-uniform')
}

# Define a custom PyTorch model
class LinearRegressionNN(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionNN, self).__init__()
        self.fc = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return self.fc(x)

# Create a custom PyTorch estimator
class CustomEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, X, y):
        input_size = X.shape[1]
        self.model = LinearRegressionNN(input_size)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        for epoch in range(100):  # You can adjust the number of epochs
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs.view(-1), y_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y_pred = self.model(X_tensor).numpy()
        return y_pred

# Create a Bayesian optimization object with the correct parameter
bayes_search = BayesSearchCV(
    CustomEstimator(),  # Use the custom PyTorch estimator
    search_spaces=param_space,
    n_iter=20,  # Number of optimization steps
    n_jobs=-1,
    cv=5,  # Number of cross-validation folds
    scoring='neg_mean_squared_error',  # Metric to optimize (negative MSE)
    verbose=1
)

# Perform Bayesian optimization
bayes_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding negative MSE
print("Best Learning Rate: ", bayes_search.best_params_['learning_rate'])
print("Best Negative MSE: ", bayes_search.best_score_)

'''
MSE measures the average squared difference between the true values and the predicted values. It is typically a non-negative value, where smaller values indicate better model performance.

However, in some optimization libraries like scikit-optimize (skopt) and Hyperopt, the objective function is formulated as something to maximize. For example, minimizing the negative MSE is equivalent to maximizing the negative of the negative MSE, which is just MSE. In such cases, you might see negative values for MSE, but it's still interpreted in the same way: smaller (more negative) values indicate better model performance.

So, when you encounter a negative MSE in the context of hyperparameter optimization, you should interpret it as the standard MSE, where lower (more negative) values represent better model predictions with less error.
'''