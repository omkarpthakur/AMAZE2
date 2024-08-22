# Gradient descent implementation for linear regression
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = 0
        self.b = 0

    def compute_cost(self, X, y):
        n = len(y)
        predictions = self.m * X + self.b
        cost = (1 / (2 * n)) * np.sum((predictions - y) ** 2)
        return cost

    def fit(self, X, y):
        n = len(y)
        for _ in range(self.epochs):
            predictions = self.m * X + self.b
            # Calculate gradients
            dm = (1 / n) * np.sum((predictions - y) * X)
            db = (1 / n) * np.sum(predictions - y)
            # Update parameters
            self.m -= self.learning_rate * dm
            self.b -= self.learning_rate * db

    def predict(self, X):
        return self.m * X + self.b


# Initialize and train the model
model_gd = LinearRegressionGD(learning_rate=0.01, epochs=1000)
model_gd.fit(time, population_density)

# Predict and plot
predicted_population_density_gd = model_gd.predict(time)

plt.scatter(time, population, color='blue', label='Original Data')
plt.plot(time, predicted_population_density_gd, color='red', label='Gradient Descent Fit')
plt.xlabel("Time (years)")
plt.ylabel("Population Density")
plt.title("Time vs. Population Density with Gradient Descent Regression")
plt.legend()
plt.show()

model_gd.m, model_gd.b  # Display the learned parameters
