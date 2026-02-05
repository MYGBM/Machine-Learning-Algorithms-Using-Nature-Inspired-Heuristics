import unittest
import random
from genetic_algorithm import Genetic_Algorithm  # Update this to your module's name


class TestGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        self.fitness_function = lambda x: -sum(x)
        self.crossover_function = lambda p1, p2: (
            p1[: len(p1) // 2] + p2[len(p2) // 2 :],
            p2[: len(p2) // 2] + p1[len(p1) // 2 :],
        )
        self.mutation_function = lambda x: [gene + 1 for gene in x]
        self.generate_individual = lambda: [random.randint(0, 10) for _ in range(5)]

        self.ga = Genetic_Algorithm(
            population_size=10,
            fitness_function=self.fitness_function,
            crossover_function=self.crossover_function,
            mutation_function=self.mutation_function,
            mutation_rate=0.5,
            max_generations=50,
            fitness_threshold=50,
        )

    def test_initialize_population(self):
        self.ga.initialize_population(self.generate_individual)
        self.assertEqual(len(self.ga.population), 10)
        for individual in self.ga.population:
            self.assertIsInstance(individual, list)
            self.assertEqual(len(individual), 5)

    def test_evaluate_population(self):
        self.ga.initialize_population(self.generate_individual)
        evaluated_population = self.ga.evaluate_population()
        self.assertEqual(len(evaluated_population), 10)
        for individual, fitness in evaluated_population:
            self.assertEqual(-fitness, sum(individual))

    def test_mutate(self):
        individual = [1, 2, 3, 4, 5]
        mutated = self.ga.mutate(individual)
        if mutated != individual:
            self.assertNotEqual(mutated, individual)
            self.assertEqual(len(mutated), len(individual))
            self.assertTrue(all(gene == original + 1 for gene, original in zip(mutated, individual)))

    def test_reproduce(self):
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        children = self.ga.reproduce(parent1, parent2)
        self.assertEqual(len(children), 2)
        self.assertTrue(all(len(child) == len(parent1) for child in children))

    def test_evolve(self):
        self.ga.initialize_population(self.generate_individual)
        initial_population = self.ga.population[:]
        self.ga.evolve()
        self.assertNotEqual(self.ga.population, initial_population)

    def test_run(self):
        self.ga.initialize_population(self.generate_individual)
        best_individual, best_fitness = self.ga.run(self.generate_individual)
        self.assertTrue(best_fitness >= self.ga.fitness_threshold)
        self.assertEqual(len(best_individual), 5)


if __name__ == "__main__":
    unittest.main()

