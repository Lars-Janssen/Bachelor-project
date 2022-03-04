from matplotlib import pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
import batch
import numpy as np

# Amount of times the batch_size is doubled and e is halved.
doubles = 2

# The amount of rounds in each game.
batch_size = 1000 * (2 ** doubles)

# How often each batch is run for a delta.
iterations = 50

# Rate of updating the Qvalues after the batch.
a_rate = 1

# The values of the rewards.
t, r, p, s = 1.5, 1, 0, -0.5
r1 = [p, t, s, r]

# The strategy of the opponent.
p2 = [0, 0, 0, 1]

# The probability that player 2 makes a move different to its strategy.
e = 0.1 / (2 ** doubles)
d=0

# Interval of deltas in which we find the best counter-strategy.
start = 0
end = 100

# How many (equally spaced) points we check in the interval.
steps = 10

def dectobin(number):
    """Converts a decimal number to a list of binary numbers of length 4."""
    output = [int(x) for x in bin(number)[2:]]
    return [0] * (4 - len(output)) + output

def test_delta(delta):
    """Function to run in parallel."""
    batch1 = batch.batch(delta, e, batch_size, p2, r1)
    batch1.run()
    return batch1.return_values()

class counterstrats:
    def __init__(self):
        # Make a list of the deltas to check.
        self.deltas = [(start + (end - start) * i / steps) / 100 for i in range(steps)]
        self.new_strat1 = [0, 0, 0, 0]
        # Make a list where we put how many times each strat is chosen as the new one.
        self.new_strat = np.zeros((len(self.deltas), 16))

    def get_data(self):
        for i in tqdm(range(len(self.deltas))):
            # Simultaneously run the batches.
            pool = mp.Pool(mp.cpu_count())
            result = pool.map(test_delta, [self.deltas[i] for _ in range(iterations)])
            pool.close()

            # Add the results to new_strat.
            for j in range(len(result)):
                for k in range(len(result[0])):
                    self.new_strat1[k] = int(result[j][k][0] < result[j][k][1])
                self.new_strat[i][batch.bintodec(self.new_strat1)] += 1

    def print_and_plot(self):
        # Print the strats for each delta.
        for i in range(len(self.new_strat)):
            print(self.deltas[i], self.new_strat[i])

        # Add a line to plot.
        G = []
        for i in range(len(self.new_strat)):
            G.append(self.new_strat[i][0])

        # Make the plot.
        plt.plot(self.deltas, G)
        plt.legend(["[0,0,0,0]"])
        plt.ylim(0, iterations + iterations / 10)
        plt.axvline(x=1 / 3, ymin=0, ymax=iterations, color="orange")
        plt.title(str(batch_size))
        # plt.show()
        plt.savefig("Approx/" + str(batch_size) + ".png")

if __name__ == "__main__":
    counterstat = counterstrats()
    counterstat.get_data()
    counterstat.print_and_plot()