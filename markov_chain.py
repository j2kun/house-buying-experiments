'''
A Markov Model that works by discretizing
the continuous space of values, treating each bin
as a state in the markov chain, and then training
the Markov model on sequential pairs of observations.
'''

from binary_search import binary_search
from binary_search import BinarySearchHint
from collections import defaultdict
from dataclasses import dataclass
from model import ExperimentParameters
from model import Model
from model import plot
from model import run_experiment
import numpy as np
import pandas as pd
import random


@dataclass(frozen=True)
class Range:
    lower: float
    upper: float

    def contains(self, x):
        return self.lower <= x < self.upper

    def select_uniform_in_range(self):
        return random.uniform(self.lower, self.upper)


def draw(weights):
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex

        choiceIndex += 1


class MarkovChain:
    def __init__(self, states):
        self.states = states

        # Dict[state, Dict[state, int]]
        # each (state, state) pair counts as one entry of a transition table
        self.transition_frequencies = defaultdict(lambda: defaultdict(int))

    def fit(self, state_pairs):
        for starting_state, ending_state in state_pairs:
            self.transition_frequencies[starting_state][ending_state] += 1

    def sample_next_state(self, starting_state):
        if len(self.transition_frequencies[starting_state]) == 0:
            ending_states = tuple(self.transition_frequencies.keys())
        else:
            ending_states = tuple(
                self.transition_frequencies[starting_state].keys())

        weights = tuple(
            self.transition_frequencies[starting_state][ending_state]
            for ending_state in ending_states)
        index = draw(weights)
        return ending_states[index]

    def sample(self, n):
        current_state = random.choice(
            tuple(self.transition_frequencies.keys()))
        observations = [current_state.select_uniform_in_range()]
        for i in range(n - 1):
            current_state = self.sample_next_state(current_state)
            observations.append(current_state.select_uniform_in_range())
        return observations


def find_bin(bins, value):
    def test(the_bin):
        if the_bin.contains(value):
            return BinarySearchHint(found=True)
        elif the_bin.upper <= value:
            return BinarySearchHint(tooLow=True)
        else:
            return BinarySearchHint(tooLow=False)

    return binary_search(bins, test).value


def fit_markov_chain(data, bin_size=0.005):
    snp_percent_change = data["Percent change"].to_numpy()

    # a fancy way to get a list of sequential pairs from a 1d numpy array
    change_pairs = np.stack(
        [np.roll(snp_percent_change, 1), snp_percent_change]).T[1:]

    change_range = Range(
        lower=np.min(snp_percent_change),
        upper=np.max(snp_percent_change),
    )

    bins = []
    x = change_range.lower
    while x < change_range.upper:
        bins.append(Range(x, x + 0.005))
        x += 0.005

    # change pairs need to be converted to bin pairs, which are pairs of states
    # in the markov model
    bin_pairs = []
    for (first_change, second_change) in change_pairs:
        first_change_bin = find_bin(bins, first_change)
        second_change_bin = find_bin(bins, second_change)
        bin_pairs.append((first_change_bin, second_change_bin))

    mc = MarkovChain(bins)
    mc.fit(bin_pairs)

    return Model(name="Markov Chain", sampler=mc.sample)


if __name__ == "__main__":
    data = pd.read_csv("snp.tsv", sep='\t')
    model = fit_markov_chain(data.head(50 * 12))

    params = ExperimentParameters(
        years=30,
        house_price_usd=1000000,
        on_hand_usd=700000,
        mortgage_rate=0.03,
    )

    plot(params, run_experiment(model, params))
