from random import randint, random

def bintodec(list):
    """Converts a list of binary numbers to a decimal number."""
    output = 0
    for i in range(len(list)):
        output += list[i] * 2 ** (len(list) - i - 1)
    return output

class batch:
    """This class defines a single batch."""

    def __init__(self, d, e, batch_size, p2, r1):
        """Set the initial values."""
        # Set the delta and the batch_size.
        self.d = d
        self.e = e
        self.size = batch_size

        # Flip the middle of the strategy due to assymmetry.
        self.p2 = [p2[0], p2[2], p2[1], p2[3]]

        # Set the rewards for player 1.
        self.r1 = r1

        # Set the other values.
        self.reset()

    def reset(self):
        # We start in state 0, this has no effect on the Qvalues.
        self.state = 0

        # Count the amount of times a state action pair has been played.
        self.counter = [[0, 0], [0, 0], [0, 0], [0, 0]]

        # Count the rewards gotten for each state action pair.
        self.reward = [[0, 0], [0, 0], [0, 0], [0, 0]]

        # Set the initial valQ and nextQ to zero.
        self.valQ = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.nextQ = [[0, 0], [0, 0], [0, 0], [0, 0]]

    def run(self):
        """Run the batch."""

        # Run a batch until every state action pair has been visited.
        while any(0 in i for i in self.counter):
            # Reset the values for each try.
            self.reset()

            # Simulate the rounds.
            for _ in range(self.size):

                # Player 1 plays a random action
                a1 = randint(0, 1)
                # Player 2 plays an action according to its strategy
                # but the opposite with probability e.
                a2 = self.p2[self.state] if random(
                ) > self.e else not self.p2[self.state]
                new_state = bintodec([a1, a2])

                # Add to the reward and counter
                self.counter[self.state][a1] += 1
                self.reward[self.state][a1] += self.r1[new_state]

                # Set the alpha in the batch to how often the state
                # action pair has come up.
                a_aux = 1 / self.counter[self.state][a1]

                # Update the nextQ and valQ.
                self.nextQ[self.state][a1] += max(
                    self.valQ[new_state][0], self.valQ[new_state][1])

                self.valQ[self.state][a1] = (1 - a_aux) * \
                    self.valQ[self.state][a1] + a_aux * \
                    (self.r1[new_state] + self.d *
                     max(self.valQ[new_state][0],
                         self.valQ[new_state][1]))

                self.state = new_state

    def return_values(self):
        """Return Qvalues according to the batch."""
        Q1 = [[0, 0], [0, 0], [0, 0], [0, 0]]

        for s in range(len(Q1)):
            for a in range(len(Q1[s])):
                # Set the Qvalues.
                Q1[s][a] = (self.reward[s][a] / self.counter[s][a]) + \
                    self.d * (self.nextQ[s][a] / self.counter[s][a])
        return Q1