import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from genetic_algorithm import Genetic_Algorithm

import copy


class CustomSGDClassifier:
    def __init__(self, n_inputs, n_outputs, learning_rate=0.01) -> None:
        self.weights = np.random.randn(n_inputs, n_outputs) * 0.01
        self.bias = np.zeros(n_outputs)
        self.learning_rate = learning_rate

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.loss_history = []


    def __calculate_gradient(self, X_batch, y_batch):
        logits = np.dot(X_batch, self.weights) + self.bias
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))

        pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        y_true = np.eye(self.n_outputs)[y_batch]
        error = pred - y_true
        
        grad_weights = error.T @ X_batch / len(X_batch)
        grad_bias = np.mean(error, axis=0)

        return grad_weights.T, grad_bias

    def __update_parameters(self, grad_weights, grad_bias):
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias 

    def __genetic_optimizer_weights(self, X_val, y_val, verbose=False):
        def generate_individual():
            return np.random.randn(*self.weights.shape) * 0.01

        def fitness(perturbation):
            candidate_weights = self.weights + perturbation
            logits = np.dot(X_val, candidate_weights) +   self.bias
            preds = np.argmax(logits, axis=1)

            return -accuracy_score(y_val, preds)

        ga = Genetic_Algorithm(
                population_size=1000,
                fitness_function=fitness,
                crossover_function=lambda p1, p2: (0.5 * p1 + 0.5 * p2, 0.5 * p1 + 0.5 * p2),
                mutation_function=lambda x: x + np.random.randn(*x.shape) * 0.01,
                mutation_rate=0.3,
                max_generations=1000,
                )
        best_perturbation, _ = ga.run(generate_individual)

        self.weights += best_perturbation
        if verbose:
            print("GA: Applied weight perturbation")

    def __genetic_optimizer_learning_rate(self, X_val, y_val, verbose=False):
        def generate_individual():
            return np.random.uniform(0.001, 0.1)

        def fitness(lr):
            temp_model = copy.deepcopy(self)
            temp_model.learning_rate = lr
            for _ in range(5): 
                grad_weights, grad_bias = temp_model.__calculate_gradient(X_val[:32], y_val[:32])
                temp_model.__update_parameters(grad_weights, grad_bias)

            return -temp_model.evaluate(X_val, y_val)


        ga = Genetic_Algorithm(
                population_size=1000,
                fitness_function=fitness,
                crossover_function=lambda p1, p2: ((p1 + p2) / 2, (p1 + p2) / 2),
                mutation_function=lambda x: x * np.random.uniform(0.8, 1.2),
                mutation_rate=0.2,
                max_generations=1000
                )
        best_lr, _ = ga.run(generate_individual)

        self.learning_rate = best_lr
        if verbose:
            print(f"GA: Updated learning rate to {best_lr:.6f}")

    def compute_loss(self, X_batch, y_batch):
        """Compute cross-entropy loss for a batch."""
        logits = np.dot(X_batch, self.weights) + self.bias
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        y_true = np.eye(self.n_outputs)[y_batch]
        
        loss = -np.mean(np.sum(y_true * np.log(probs + 1e-10), axis=1))  
        
        return loss

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def fit(self, X_train, y_train, X_val, y_val, n_epochs=100, batch_size=32, ga_interval=10, verbose=True):
        best_fitness = -np.inf
        best_weights = self.weights.copy()
        best_bias = self.bias.copy()
        best_lr = self.learning_rate
        self.loss_history.clear()

        for epoch in range(n_epochs):
            permutation = np.random.permutation(len(X_train))
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            epoch_loss = 0

            for i in range(0, len(X_train), batch_size):
                X_batch = X_shuffled[i: i + batch_size]
                y_batch = y_shuffled[i: i + batch_size]
                
                grad_weights, grad_bias = self.__calculate_gradient(X_batch, y_batch)
                self.__update_parameters(grad_weights, grad_bias)

                batch_loss = self.compute_loss(X_batch, y_batch) 
                epoch_loss += batch_loss


            epoch_loss /= (len(X_train) // batch_size)
            self.loss_history.append(epoch_loss)
            if verbose:
                print(f"Epoch {epoch+1}: Average Epoch Loss = {epoch_loss:.6f}")

            # do genetic optimization every ga_interval
            if ga_interval > 0  and (epoch + 1) % ga_interval == 0:
                pre_ga_weights = self.weights.copy()
                pre_ga_bias = self.bias.copy()
                pre_ga_lr = self.learning_rate
                pre_ga_fitness = self.evaluate(X_val, y_val)

                self.__genetic_optimizer_learning_rate(X_val, y_val)
                self.__genetic_optimizer_weights(X_val, y_val)


                post_ga_fitness = self.evaluate(X_val, y_val)
                if post_ga_fitness < pre_ga_fitness:
                    self.weights = pre_ga_weights
                    self.bias = pre_ga_bias
                    self.learning_rate = pre_ga_lr
                    post_ga_fitness = pre_ga_fitness


                if post_ga_fitness > best_fitness:
                    best_fitness = post_ga_fitness
                    best_weights = self.weights.copy()
                    best_bias = self.bias.copy()
                    best_lr = self.learning_rate

            else:
                current_fitness = self.evaluate(X_val, y_val)
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_weights = self.weights.copy()
                    best_bias = self.bias.copy()
                    best_lr = self.learning_rate

        self.weights = best_weights
        self.bias = best_bias
        self.learning_rate = best_lr

    def evaluate(self, X, y):
        logits = np.dot(X, self.weights) + self.bias
        preds = np.argmax(logits, axis=1)

        return accuracy_score(y, preds)

if __name__ == "__main__":
    # Load dataset
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the model
    model = CustomSGDClassifier(n_inputs=X_train.shape[1], n_outputs=len(np.unique(y)))
    model.fit(X_train, y_train, X_val, y_val, n_epochs=3000, ga_interval=10)

    print(model.evaluate(X_val, y_val))
