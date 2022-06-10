from __future__ import annotations

from random import random

import numpy as np


def bintodec(list):
    """Converts a list of binary numbers to a decimal number."""
    output = 0
    for i in range(len(list)):
        output += list[i] * 2 ** (len(list) - i - 1)
    return output


class batch:
    """This class defines a single batch."""

    def __init__(self, d, e, batch_size, p1, p2, r):
        """Set the initial values."""
        # Set the delta and the batch_size.
        self.d = d
        self.e = e
        self.size = batch_size

        # Flip the middle of the strategy due to asymmetry.
        self.p = [p1, [p2[0], p2[2], p2[1], p2[3]]]

        # Set the rewards.
        self.r = [r, [r[0], r[2], r[1], r[3]]]

        # Set the other values.
        self.reset()

    def reset(self):
        # We start in state 0, this has no effect on the Qvalues.
        self.state = 0

        # Count the amount of times a state action pair has been played.
        self.counter = [
            [[0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0]],
        ]

        # Count the rewards gotten for each state action pair.
        self.reward = [
            [[0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0]],
        ]

        # Set the initial valQ and nextQ to zero.
        self.valQ = [
            [[0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0]],
        ]
        self.nextQ = [
            [[0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0]],
        ]

        self.a_aux = [0, 0]

        self.a = [0, 0]

    def run(self):
        """Run the batch."""

        # Run a batch until every state action pair has been visited.
        while not np.all(self.counter):
            # Reset the values for each try.
            self.reset()

            # Simulate the rounds.
            for t in range(self.size):

                # The players play an action according to their strategy
                # but the opposite with probability e.
                for i in range(2):
                    self.a[i] = self.p[i][self.state] if random(
                    ) > self.e else not self.p[i][self.state]

                new_state = bintodec(self.a)

                # Add to the reward and counter
                for i in range(2):
                    self.counter[i][self.state][self.a[i]] += 1
                    self.reward[i][self.state][
                        self.a[i]
                    ] += self.r[i][new_state]

                    # Set the alpha in the batch to how often the state
                    # action pair has come up.
                    self.a_aux[i] = 1 / self.counter[i][self.state][self.a[i]]

                    # Update the nextQ and valQ.
                    self.nextQ[i][self.state][self.a[i]] += max(
                        self.valQ[i][new_state][0], self.valQ[i][new_state][1],
                    )

                    self.valQ[i][self.state][self.a[i]] = (1 - self.a_aux[i]) \
                        * self.valQ[i][self.state][self.a[i]] + self.a_aux[i] \
                        * (
                            self.r[i][new_state] + self.d *
                            max(
                                self.valQ[i][new_state][0],
                                self.valQ[i][new_state][1],
                            )
                        )

                self.state = t % 4

    def return_values(self):
        """Return Qvalues according to the batch."""
        Q = [
            [[0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0]],
        ]

        for i in range(2):
            for s in range(len(Q[i])):
                for a in range(len(Q[i][s])):
                    # print(self.counter1[s][a])
                    # Set the Qvalues.
                    Q[i][s][a] = (
                        self.reward[i][s][a]
                        / self.counter[i][s][a]
                    ) \
                        + self.d * (
                            self.nextQ[i][s]
                            [a] / self.counter[i][s][a]
                        )
        return Q


# t, r, p, s = 1, 5/10, 2/10, 0
# b = batch(9/10, 1/10, 10000, [0,0,0,1], [0,0,0,1], [p, t, s, r])
#  b.run()
# result = b.return_values()
#  print(result[1])
#
#
# new_strat1 = [0, 0, 0, 0]
# new_strat2 = [0, 0, 0, 0]
#  for k in range(len(result[0])):
#     new_strat1[k] = int(result[0][k][0] < result[0][k][1])
#     new_strat2[k] = int(result[1][k][0] < result[1][k][1])
#
# print(new_strat1, new_strat2)
#
