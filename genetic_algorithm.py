import random
from typing import Any, Callable, List, Tuple


class Genetic_Algorithm:
    def __init__(
                self, 
                population_size: int,
                fitness_function: Callable[[Any], float],
                crossover_function: Callable[[Any, Any], Tuple[Any, Any]],
                mutation_function: Callable[[Any], Any],
                mutation_rate: float=0.2,
                max_generations: int=1000,
                fitness_threshold: float=0.0, # by default it will run for max_generations
                surviving_population: float=0.7 # percentage of population that will survive to next generation
            ) -> None:

        self.population_size = population_size
        self.fitness_function = fitness_function
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.fitness_threshold = fitness_threshold
        self.surviving_population = surviving_population

        self.population: List[Any] = []
        self.generation = 0


    def run(self, generate_individual: Callable[[], Any], verbose=False)-> Tuple[Any, float]:
        self.initialize_population(generate_individual)

        best_fitness = self.fitness_function(self.population[0])
        self.fitness_threshold = min(best_fitness, self.fitness_threshold)

        if verbose:
            print(f"Generation starting with a fitness of {best_fitness}")

        while not self.should_stop(best_fitness):
            self.evolve()
            best_fitness = self.fitness_function(self.population[0])

            self.generation += 1

            if verbose:
                print(f"Generation{self.generation} with best fitness of {best_fitness}")

        return self.population[0], self.fitness_function(self.population[0])


    def evaluate_population(self) -> List[Tuple[Any, float]]:
        return [(individual, self.fitness_function(individual)) for individual in self.population]


    def evolve(self) -> None:
        self.population.sort(key=lambda x: self.fitness_function(x))

        surviving_population = int(self.surviving_population * self.population_size)

        replaced_population = self.population_size - surviving_population
        next_generation_producers = self.population[:surviving_population]
        reproduced_population = self.next_generation(next_generation_producers, replaced_population)

        self.population = [*next_generation_producers, *reproduced_population]


    def reproduce(self, parent1: Any, parent2: Any) -> List[Any]:
        child1, child2 = self.crossover_function(parent1, parent2)
        crossed_over_children = [self.mutate(child1), self.mutate(child2)]

        return crossed_over_children
    

    def mutate(self, individual: Any) -> Any:
        if random.random() <= self.mutation_rate:
            return self.mutation_function(individual)
        return individual


    def initialize_population(self, generate_individual: Callable[[], Any]) -> None:
        self.population = [generate_individual() for _ in range(self.population_size)]


    def next_generation(self, parents: List[Any], child_size: int) -> List[Any]:
        next_gen = []
        for _ in range(child_size):
            random_parent_1 = parents[self.random_index(len(parents))]
            random_parent_2= parents[self.random_index(len(parents))]

            next_gen.extend(self.reproduce(random_parent_1, random_parent_2))

        return next_gen[:child_size]


    def random_index(self, end: int) -> int:
        return random.randint(0, end - 1) 


    def should_stop(self, best_fitness: float) -> bool:
        return best_fitness <= self.fitness_threshold or self.generation >= self.max_generations

