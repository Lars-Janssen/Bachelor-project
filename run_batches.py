from __future__ import annotations

import csv
import multiprocessing as mp

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import batch


# The amount of rounds in each game.
batch_size = 10000

# How often each batch is run for a delta.
iterations = 50

# Rate of updating the Qvalues after the batch.
a_rate = 1

# The values of the rewards.
t, r, p, s = 1, 9/10, 1/10, 0
r1 = [p, t, s, r]

# The strategies.
p1 = [0, 0, 0, 1]
p2 = [0, 1, 0, 0]

# The probability that player 2 makes a move different to its strategy.
e = 1/100
d = 0


def dectobin(number):
    """Converts a decimal number to a list of binary numbers of length 4."""
    output = [int(x) for x in bin(number)[2:]]
    return [0] * (8 - len(output)) + output


def test_delta(delta):
    """Function to run in parallel."""
    batch1 = batch.batch(delta, e, batch_size, p1, p2, r1)
    batch1.run()
    return batch1.return_values()


def test_graph(pair):
    """Function to run in parallel."""
    p1 = pair[0:4]
    p2 = pair[4:]
    delta = 9/10
    batch1 = batch.batch(delta, e, batch_size, p1, p2, r1)
    batch1.run()
    return batch1.return_values()


class counterstrats:
    def __init__(self, start, end, steps, type):
        self.start = start
        self.end = end
        self.steps = steps
        self.type = type
        # Make a list of the deltas to check.
        self.deltas = [(start + (end - start) * i / steps) /
                       100 for i in range(steps)]
        self.new_strat = [0, 0, 0, 0, 0, 0, 0, 0]
        # Make a list where we put how many times each strat is chosen
        # as the new one.
        if type == 2:
            self.stats = np.zeros((len(self.deltas), 256))
        elif type == 1:
            self.stats = np.zeros((256, 256))

    def get_data(self):
        for i in tqdm(range(len(self.deltas))):
            # Simultaneously run the batches.
            pool = mp.Pool(mp.cpu_count())
            result = pool.map(
                test_delta, [
                    self.deltas[i]
                    for _ in range(iterations)
                ],
            )
            pool.close()

            # Add the results to new_strat.
            for j in range(len(result)):
                for k in range(4):
                    self.new_strat[k] = int(
                        result[j][0][k][0] < result[j][0][k][1],
                    )
                    self.new_strat[k+4] = int(
                        result[j]
                        [1][k][0] < result[j][1][k][1],
                    )
                self.stats[i][batch.bintodec(self.new_strat)] += 1

    def get_graph(self):
        for i in tqdm(range(256)):
            # Simultaneously run the batches.
            pair = dectobin(i)

            pool = mp.Pool(mp.cpu_count())
            result = pool.map(
                test_graph, [
                    pair
                    for _ in range(iterations)
                ],
            )
            pool.close()
            # Add the results to new_strat.
            for j in range(len(result)):
                for k in range(4):
                    self.new_strat[k] = int(
                        result[j][0][k][0] < result[j][0][k][1],
                    )
                    self.new_strat[k+4] = int(
                        result[j]
                        [1][k][0] < result[j][1][k][1],
                    )
                self.stats[i][batch.bintodec(self.new_strat)] += 1

    def print_and_plot(self):
        if self.type == 2:

            # Print the strats for each delta.
            for i in range(len(self.stats)):
                print(self.deltas[i], self.stats[i])

            # Add a line to plot.
            G1 = []
            G2 = []
            G3 = []
            for i in range(len(self.stats)):
                G1.append(self.stats[i][0] / iterations)
                G2.append(self.stats[i][208] / iterations)
                G3.append(self.stats[i][209] / iterations)

            # Make the plot.
            plt.plot(self.deltas, G1)
            plt.plot(self.deltas, G2)
            plt.plot(self.deltas, G3)
            plt.xlabel('\u03B4')
            plt.ylabel('probability')
            plt.legend(['0', '208', '209'])
            plt.axvline(x=1 / 4, ymin=0, ymax=iterations, color='orange')
            plt.axvline(x=3 / 4, ymin=0, ymax=iterations, color='green')
            plt.title(
                'batch-size = ' + str(batch_size) +
                ', \u03B5 = ' + str(e),
            )
            plt.savefig(
                'Approx/' + str(self.start) + '-' + str(self.end) + '-' +
                str(self.steps) + ', ' +
                str(batch_size) + '-' + str(e) + '.png',
            )
            plt.show()

        if self.type == 1:
            arr = self.stats
            print(arr)
            with open('new_file.csv', 'w+') as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=',')
                csvWriter.writerows(arr)


if __name__ == '__main__':
    text = input()
    if text == '1':
        counterstat = counterstrats(0, 0, 0, 1)
        counterstat.get_graph()
        counterstat.print_and_plot()
    if text == '2':
        counterstat = counterstrats(0, 100, 40, 2)
        counterstat.get_data()
        counterstat.print_and_plot()
