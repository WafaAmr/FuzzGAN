import random
import gan_mutation_manager
from gan_mnist_member import MnistMember
from config import MUTOPPROB
from utils import get_distance


class DigitMutator:

    def __init__(self, digit):
        self.digit = digit
        self.seed = digit.seed

    def mutate(self, reference=None):
        # Select mutation operator.
        rand_mutation_probability = random.uniform(0, 1)
        if rand_mutation_probability >= MUTOPPROB:
            mutation = 1
        else:
            mutation = 2

        condition = True
        counter_mutations = 0
        distance_inputs = 0
        while condition:
            counter_mutations += 1
            mutant = gan_mutation_manager.mutate(self.digit.state, mutation)
            m_image_array = mutant["m_res"].image_array

            distance_inputs = get_distance(self.digit.purified, m_image_array)

            if distance_inputs != 0:
                if reference is not None:
                    distance_inputs = get_distance(reference.purified, m_image_array)
                    if distance_inputs != 0:
                        condition = False
                else:
                    condition = False

        self.digit.state = mutant
        self.digit.purified = m_image_array
        self.digit.predicted_label = None
        self.digit.confidence = None
        self.digit.correctly_classified = None

        return distance_inputs

    def generate(self):
        # Select mutation operator.
        rand_mutation_probability = random.uniform(0, 1)
        if rand_mutation_probability >= MUTOPPROB:
            mutation = 1
        else:
            mutation = 2

        condition = True
        counter_mutations = 0
        distance_inputs = 0
        while condition:
            counter_mutations += 1
            mutant1, mutant2 = gan_mutation_manager.generate(
                self.digit.state,
                mutation)

            distance_inputs = get_distance(mutant1.purified,
                                           mutant2.purified)

            if distance_inputs != 0:
                condition = False

        first_digit = MnistMember(mutant1.state,
                                  self.digit.expected_label,
                                  self.seed)
        second_digit = MnistMember(mutant2.state,
                                   self.digit.expected_label,
                                   self.seed)
        first_digit.purified = mutant1.purified
        second_digit.purified = mutant2.purified
        return first_digit, second_digit, distance_inputs


