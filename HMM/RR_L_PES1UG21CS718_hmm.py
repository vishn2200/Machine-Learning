import torch
import numpy as np
class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = {state: i for i, state in enumerate(self.states)}
        self.emissions_dict = {emission: i for i, emission in enumerate(self.emissions)}

    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            Porbability of the hidden state at time t given an obeservation sequence 
        """
        T = len(seq)
        delta = torch.zeros((self.N, T))
        psi = torch.zeros((self.N, T), dtype=torch.long)

        pi_tensor = torch.tensor(self.pi)

        delta[:, 0] = pi_tensor * self.B[:, self.emissions_dict[seq[0]]]

        for t in range(1, T):
            probabilities = delta[:, t-1].unsqueeze(1) * self.A * self.B[:, self.emissions_dict[seq[t]]]

            max_prob, max_idx = torch.max(probabilities, dim=0)

            delta[:, t] = max_prob
            psi[:, t] = max_idx

        path = [torch.argmax(delta[:, -1]).item()]
        for t in range(T - 1, 0, -1):
            path.insert(0, psi[path[0], t].item())

        state_sequence = [self.states[i] for i in path]

        return state_sequence
        