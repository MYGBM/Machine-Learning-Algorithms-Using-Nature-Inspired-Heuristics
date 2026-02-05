# Machine Learning Algorithms Using Nature-Inspired Heuristics

This project explores and applies nature-inspired metaheuristic algorithms to optimize machine learning algorithms, with a specific focus on **enhancing Stochastic Gradient Descent (SGD)** performance using **Genetic Algorithms (GA)**.

## 🎯 Objective

The primary goal is to address common limitations in traditional SGD:

- Slow convergence
- Susceptibility to local minima
- Static learning rate

By integrating Genetic Algorithm optimization techniques inspired by natural evolutionary processes, we achieve improved convergence and accuracy compared to standard SGD.

## 📁 Project Structure

```
├── genetic_algorithm.py      # Core GA implementation
├── custom_sgd_classifier.py  # GA-enhanced SGD classifier
├── genetic_algorithm_test.py # Unit tests for GA
├── n_queens_solver.py        # GA demo: N-Queens problem
├── notebook.ipynb            # Full experimental analysis
└── README.md                 # This file
```

## 🧬 Core Components

### 1. Genetic Algorithm (`genetic_algorithm.py`)

A flexible, reusable Genetic Algorithm framework that can be applied to various optimization problems.

**Key Features:**

- Configurable population size, mutation rate, and crossover functions
- Fitness threshold-based or generation-limited termination
- Elitism through surviving population percentage
- Customizable individual generation, crossover, and mutation functions

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `population_size` | Number of individuals in each generation | - |
| `fitness_function` | Callable to evaluate individual fitness | - |
| `crossover_function` | Callable for genetic crossover | - |
| `mutation_function` | Callable for genetic mutation | - |
| `mutation_rate` | Probability of mutation | 0.2 |
| `max_generations` | Maximum iterations | 1000 |
| `fitness_threshold` | Target fitness to stop early | 0.0 |
| `surviving_population` | Percentage surviving to next generation | 0.7 |

### 2. Custom SGD Classifier (`custom_sgd_classifier.py`)

A custom implementation of SGD classification enhanced with Genetic Algorithm optimization.

**How It Works:**

1. **Standard SGD Training**: Computes gradients using softmax cross-entropy loss and updates weights
2. **GA Weight Optimization**: Periodically applies GA to find optimal weight perturbations
3. **GA Learning Rate Optimization**: Uses GA to dynamically optimize the learning rate

**Key Methods:**

- `fit(X_train, y_train, X_val, y_val, n_epochs, batch_size, ga_interval)` - Train the model
- `predict(X)` - Make predictions
- `evaluate(X, y)` - Calculate accuracy

**GA Integration:**

- Every `ga_interval` epochs, the classifier:
  1. Optimizes the learning rate using GA
  2. Finds optimal weight perturbations using GA
  3. Only applies changes if they improve validation accuracy

### 3. N-Queens Solver (`n_queens_solver.py`)

A demonstration of the GA framework solving the classic N-Queens problem.

**Problem:** Place N queens on an N×N chessboard such that no two queens threaten each other.

**Implementation:**

- **Individual Representation**: List of column positions for each row
- **Fitness Function**: Counts attacking pairs (goal: minimize to 0)
- **Crossover**: Single-point crossover with random swaps
- **Mutation**: Random position changes

### 4. Unit Tests (`genetic_algorithm_test.py`)

Comprehensive test suite for the Genetic Algorithm implementation covering:

- Population initialization
- Population evaluation
- Mutation operations
- Reproduction/crossover
- Evolution process
- Full algorithm execution

## 📊 Experimental Results

The notebook (`notebook.ipynb`) demonstrates the GA-SGD classifier on the **Digits Dataset** (handwritten digit classification).

### Performance Comparison

| Metric          | Standard SGD | GA-SGD   |
| --------------- | ------------ | -------- |
| Accuracy        | 96%          | **97%**  |
| Macro Avg F1    | 0.96         | **0.98** |
| Weighted Avg F1 | 0.96         | **0.98** |

### Key Findings

1. **Improved Accuracy**: GA-SGD achieves ~1% higher accuracy than standard SGD
2. **Better Convergence**: Loss curves show smoother, more consistent descent
3. **Robust Optimization**: GA helps escape local minima through population-based search
4. **Adaptive Learning**: Dynamic learning rate optimization improves training efficiency

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy scikit-learn matplotlib seaborn
```

### Usage Examples

#### Using the Genetic Algorithm

```python
from genetic_algorithm import Genetic_Algorithm

# Define your functions
def fitness(individual):
    return -sum(individual)  # Minimize sum

def crossover(p1, p2):
    mid = len(p1) // 2
    return p1[:mid] + p2[mid:], p2[:mid] + p1[mid:]

def mutation(x):
    return [gene + 1 for gene in x]

def generate_individual():
    return [random.randint(0, 10) for _ in range(5)]

# Create and run GA
ga = Genetic_Algorithm(
    population_size=100,
    fitness_function=fitness,
    crossover_function=crossover,
    mutation_function=mutation,
    mutation_rate=0.2,
    max_generations=50
)

best, best_fitness = ga.run(generate_individual)
```

#### Using the GA-SGD Classifier

```python
from custom_sgd_classifier import CustomSGDClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
digits = load_digits()
X, y = digits.data, digits.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
model = CustomSGDClassifier(
    n_inputs=X_train.shape[1],
    n_outputs=len(set(y))
)

model.fit(
    X_train, y_train,
    X_val=X_test, y_val=y_test,
    n_epochs=1000,
    batch_size=32,
    ga_interval=10  # Apply GA optimization every 10 epochs
)

# Evaluate
accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

## 🔬 Algorithm Details

### Genetic Algorithm Evolution Process

```
1. Initialize random population
2. Evaluate fitness of all individuals
3. While not converged:
   a. Sort population by fitness
   b. Select top performers (elitism)
   c. Create offspring through crossover
   d. Apply random mutations
   e. Combine survivors and offspring
   f. Evaluate new generation
4. Return best individual
```

### GA-SGD Training Process

```
1. Initialize weights randomly
2. For each epoch:
   a. Shuffle training data
   b. For each mini-batch:
      - Compute gradients
      - Update weights (standard SGD step)
      - Track batch loss
   c. Every ga_interval epochs:
      - Run GA to optimize learning rate
      - Run GA to find weight perturbations
      - Apply changes only if validation improves
3. Return best weights found during training
```

## 📈 Visualizations

The notebook includes:

- **Loss Curves**: Track training convergence over epochs
- **Confusion Matrices**: Compare classification performance between SGD and GA-SGD
- **Accuracy Comparison**: Bar charts showing model performance across multiple runs
- **Time Analysis**: Comparison of training times

## 🔧 Running Tests

```bash
python genetic_algorithm_test.py
```

## 📚 References

- Holland, J. H. (1992). _Adaptation in Natural and Artificial Systems_
- Mitchell, M. (1998). _An Introduction to Genetic Algorithms_
- Scikit-learn Documentation: SGDClassifier

## 📝 License

This project is for educational purposes as part of exploring nature-inspired optimization techniques in machine learning.

## 🤝 Contributing

Feel free to extend this project by:

- Adding other metaheuristic algorithms (PSO, Simulated Annealing, etc.)
- Implementing additional ML algorithms with GA optimization
- Experimenting with different datasets and problem domains
