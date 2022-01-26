import numpy as np


class MarkovChain:

    @staticmethod
    def _sanitize_input(prob_states, trans_mat):

        prob_states, trans_mat = np.asanyarray(
            prob_states), np.asanyarray(trans_mat)

        if np.any(trans_mat < 0) or np.any(trans_mat > 1) or not np.allclose(np.sum(trans_mat, axis=1), 1):
            raise ValueError('Invalid transition matrix!')

        if np.any(prob_states < 0) or np.any(prob_states > 1) or not np.allclose(np.sum(prob_states), 1):
            raise ValueError('Invalid probabilities vector!')

        return prob_states, trans_mat

    @staticmethod
    def compute_next_state(prob_states, trans_mat):

        prob_states, trans_mat = MarkovChain._sanitize_input(
            prob_states, trans_mat)
        return prob_states @ trans_mat

    @staticmethod
    def compute_nth_state(start_prob_states, trans_mat, num) -> np.ndarray:

        num = int(num)

        if num <= 0:
            raise ValueError(
                'The state number must be a positive nonzero integer!')

        start_prob_states, trans_mat = MarkovChain._sanitize_input(
            start_prob_states, trans_mat)
        return start_prob_states @ np.linalg.matrix_power(trans_mat, num)


def exercise1():
    trans_matrix = [[0.25, 0.5, 0.25], [0.5, 0.25, 0.25], [0.25, 0.25, 0.5]]
    start_prob = list(map(float, input().split()))
    print(MarkovChain.compute_nth_state(start_prob, trans_matrix, 7))


def exercise2():
    p = float(input())
    start_prob = [1, 0, 0]
    trans_matrix = [[p, 1 - 2 * p, p], [1 - 2 * p, p, p], [p, p, 1 - 2 * p]]
    print(MarkovChain.compute_nth_state(start_prob, trans_matrix, 5)[1])


def exercise3():
    result = [list(map(float, input().split()))]
    trans_matrix = [[0.01, 0.98, 0.01], [0.98, 0.01, 0.01], [0.01, 0.01, 0.98]]
    for i in range(8):
        result.append(MarkovChain.compute_next_state(result[-1], trans_matrix))
    result = [val[2] for val in result[1:]]
    print(' '.join([F'{x:.3f}' for x in result]))


def exercise4():
    n, k = tuple(map(int, input().split()))
    trans_matrix = [list(map(float, input().split())) for _ in range(k)]
    start_prob = [1 / k] * k
    previous_state = MarkovChain.compute_nth_state(
        start_prob, trans_matrix, n - 1)
    last_state = MarkovChain.compute_next_state(previous_state, trans_matrix)
    print(np.allclose(previous_state, last_state, rtol=0, atol=1e-03))
