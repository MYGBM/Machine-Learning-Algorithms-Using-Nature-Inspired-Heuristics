import random
from genetic_algorithm import Genetic_Algorithm

SIZE = 8

def generate_individual():
    board = []
    used_num = set()
    for _ in range(SIZE):
        picked_num = random.randint(0, SIZE - 1) 
        while picked_num in used_num:
            picked_num = random.randint(0, SIZE - 1) 
        
        board.append(picked_num)
        used_num.add(picked_num)

    return board


def printBoard(board):
    chess_board = [['.' for _ in range(len(board))] for _ in range(len(board))]

    for row, board_val in enumerate(board):
        chess_board[board_val][row] = "Q"

    for row in chess_board:
        print(' '.join(row))

def fitness_function(board):
    fitness_value = 0

    for queen_idx, queen_pos in enumerate(board):
        for another_queen_idx, another_queen_pos in enumerate(board):
            is_same_queen = queen_idx == another_queen_idx
            if is_same_queen:
                continue

            can_attack_horizontally = queen_pos == another_queen_pos
            can_attack_diagonally = abs(queen_idx - another_queen_idx) == abs(queen_pos - another_queen_pos)

            if can_attack_horizontally or can_attack_diagonally:
                fitness_value += 1

    return fitness_value

def cross_over_func(parent1, parent2):
    pivot_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:pivot_point] + parent2[pivot_point:]
    child2 = parent2[:pivot_point] + parent1[pivot_point:]

    for _ in range(3):
        idx1, idx2 = random.randint(0, len(child1) - 1), random.randint(0, len(child1) - 1)
        child1[idx1], child1[idx2] = child1[idx2], child1[idx1]
        child2[idx1], child2[idx2] = child2[idx2], child2[idx1]

    return child1, child2

def mutation_function(individual):
    random_index = random.randint(0, len(individual) - 1)
    random_value = random.randint(0, len(individual) - 1)
    individual[random_index] = random_value

    return individual

ga = Genetic_Algorithm(population_size=16, fitness_function=fitness_function,crossover_function=cross_over_func, mutation_function=mutation_function,mutation_rate=0.2, max_generations=float('inf'))
print('=======================================================')
ga.initialize_population(generate_individual)
printBoard(ga.population[5])
print('=======================================================')
y, _ = ga.run(generate_individual, verbose=True)
printBoard(y)
