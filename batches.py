from random import randint, random
from matplotlib import pyplot as plt
import multiprocessing as mp
from time import time

# Amount of times the batch_size is doubled and e is halved.
doubles = 4

# The amount of rounds in each game.
batch_size = 1000 * (2 ** doubles)

# How often each batch is run for a delta.
iterations = 50

# Rate of updating the Qvalues after the batch.
a_rate = 1

# The values of the rewards.
t, r, p, s = 1.5, 1, 0, -0.5

# The strategy of the opponent.
p2 = [0, 0, 0, 1]

# The probability that player 2 makes a move different to its strategy.
e = 0.1 / (2 ** doubles)

# Interval of deltas in which we find the best counter-strategy.
start = 0
end = 100

# How many (equally spaced) points we check in the interval.
steps = 20


def dectobin(number):
    """Converts a decimal number to a list of binary numbers of length 4."""
    output = [int(x) for x in bin(number)[2:]]
    return [0] * (4 - len(output)) + output


def bintodec(list):
    """Converts a list of binary numbers to a decimal number."""
    output = 0
    for i in range(len(list)):
        output += list[i] * 2 ** (len(list) - i - 1)
    return output


class batch:
    """This class defines a single batch."""

    def __init__(self):
        """Set the initial values."""
        # Set the delta and the batch_size.
        self.d = d
        self.size = batch_size

        # Flip the middle of the strategy due to assymmetry.
        self.p2 = [p2[0], p2[2], p2[1], p2[3]]

        # Set the rewards for player 1.
        self.r1 = [p, t, s, r]

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
                ) > e else not self.p2[self.state]
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
                    (self.r1[new_state] + d *
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
                    d * (self.nextQ[s][a] / self.counter[s][a])
        return Q1


def test_function(_):
    """Function to run in parallel."""
    batch1 = batch()
    batch1.run()
    return batch1.return_values()


# Make a list of the deltas to check.
deltas = [(start + (end - start) * i / steps) / 100 for i in range(steps)]

new_strat1 = [0, 0, 0, 0]

t = time()

# Make a list where we put how many times each strat is chosen as the new one.
new_strat = []
for i in range(len(deltas)):
    new_strat.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for d in deltas:

    # Simultaneously run the batches.
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(test_function, [_ for _ in range(iterations)])
    pool.close()

    # Add the results to new_strat.
    for j in range(len(result)):
        for k in range(len(result[0])):
            new_strat1[k] = int(result[j][k][0] < result[j][k][1])
        new_strat[deltas.index(d)][bintodec(new_strat1)] += 1

    print(d)

# Print the time it took.
print(print(time() - t))

# Print the strats for each delta.
for i in range(len(new_strat)):
    print(deltas[i], new_strat[i])

# Add a line to plot.
G = []
for i in range(len(new_strat)):
    G.append(new_strat[i][0])

# Make the plot.
plt.plot(deltas, G)
plt.legend(["[0,0,0,0]"])
plt.ylim(0, iterations + iterations / 10)
plt.axvline(x=1 / 3, ymin=0, ymax=iterations, color="orange")
plt.title(str(batch_size))
# plt.show()
plt.savefig("Approx/" + str(batch_size) + ".png")
