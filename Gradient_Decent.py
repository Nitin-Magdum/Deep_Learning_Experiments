import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_predicted):
    cost = np.sum((y_true-y_predicted)**2) / len(y_true)
    return cost

def gradient_descent(x, y, iterations = 10, learning_rate = 0.1, 
                     stopping_threshold = 1e-3):
    current_weight = 4
    current_bias = 0.01
    iterations = iterations
    learning_rate = learning_rate
    n = float(len(x))
      
    costs = []
    weights = []
    previous_cost = None
      
    for i in range(iterations):
        y_predicted = (current_weight * x) + current_bias
        current_cost = mean_squared_error(y, y_predicted)
        if previous_cost and abs(previous_cost-current_cost)<=stopping_threshold:
            break
        previous_cost = current_cost
        costs.append(current_cost)
        weights.append(current_weight)
        weight_derivative = -(2/n) * sum(x * (y-y_predicted))
        bias_derivative = -(2/n) * sum(y-y_predicted)
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)
        print(f"Iteration {i+1}: Cost {current_cost}, Weight \
        {current_weight}, Bias {current_bias}")
        plt.figure(figsize = (8,6))
        plt.plot(weights, costs)
        plt.scatter(weights, costs, marker='o', color='red')
        plt.title("Cost vs Weights")
        plt.ylabel("Cost")
        plt.xlabel("Weight")
        plt.show()
    return current_weight, current_bias

def main():
    X = np.array([1, 2, 3, 4])
    Y = np.array([2, 4, 6, 8])
    estimated_weight, eatimated_bias = gradient_descent(X, Y, iterations=10)
    print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {eatimated_bias}")
    Y_pred = estimated_weight*X + eatimated_bias
    plt.figure(figsize = (8,6))
    plt.scatter(X, Y, marker='o', color='red')
    plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue',
             markerfacecolor='red',
             markersize=10,linestyle='dashed')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    Y_pred = estimated_weight*7 + eatimated_bias
    print(Y_pred)
    
if __name__=="__main__":
    main()