from numpy import exp, dot, random, array

def initialize_weights():
    random.seed(1)
    synaptic_weights = random.uniform(low=-1, high=1, size=(3, 1))
    return synaptic_weights


def sigmoid(x):
    return 1 / (1 + exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


def train(inputs, expected_output, synaptic_weights, bias, learning_rate, training_iterations):
    for epoch in range(training_iterations):
        predicted_output = learn(inputs, synaptic_weights, bias)
        error = sigmoid_derivative(predicted_output) * (expected_output - predicted_output)
        weight_factor = dot(inputs.T, error) * learning_rate
        bias_factor = error * learning_rate
        synaptic_weights += weight_factor
        bias += bias_factor

        if ((epoch % 10) == 0):
            print("Epoch", epoch)
            print("Predicted Output = ", predicted_output.T)
            print("Expected Output = ", expected_output.T)
            print()
    return synaptic_weights


def learn(inputs, synaptic_weights, bias):
    return sigmoid(dot(inputs, synaptic_weights) + bias)


if __name__ == "__main__":
    synaptic_weights = initialize_weights()
    inputs = array([[0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1]])
    expected_output = array([[1, 0, 1]]).T
    test = array([1, 0, 1])
    trained_weights = train(inputs, expected_output, synaptic_weights, bias=0.01, learning_rate=0.98,
                            training_iterations=100)
    accuracy = (learn(test, trained_weights, bias=0.01)) * 100
    print("accuracy =", accuracy[0], "%")