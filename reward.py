from __future__ import annotations

import numpy as np

players = 2
actions = 2


def dectobin(number):
    """Converts a decimal number to a list of binary numbers of length 4."""
    output = [int(x) for x in bin(number)[2:]]
    return [0] * ((actions ** players) - len(output)) + output


def bintodec(list):
    """Converts a list of binary numbers to a decimal number."""
    output = 0
    for i in range(len(list)):
        output += list[i] * 2 ** (len(list) - i - 1)
    return output


# p,t,s,r
rewards = [0, 3/2, -1/2, 1]
d = 34/100

for s2 in range(actions ** (actions ** players)):
    p2 = dectobin(s2)
    counter = []
    for start in range(actions ** players):
        amounts = []
        for s1 in range(actions ** (actions ** players)):
            p1 = dectobin(s1)
            loop = []
            s = start
            statelist = [start]
            for j in range(actions ** players):
                s = bintodec([p1[s], p2[s]])
                if s in statelist:
                    loop = statelist[statelist.index(s):]
                    break
                statelist.append(s)
            loop_start = statelist.index(loop[0])
            loop_length = len(statelist) - statelist.index(loop[0])

            reward = sum(
                rewards[statelist[j]] * (d**j)
                for j in range(loop_start)
            )
            reward += (
                d ** loop_start * sum(
                    rewards[statelist[i+loop_start]] * (
                        d ** i
                    ) for i in range(loop_length)
                )
            )/(1-d ** loop_length)

            amounts.append(reward)

        temp = list(np.flatnonzero(amounts == np.max(amounts)))
        best_move = list({dectobin(i)[start] for i in temp})

        if len(best_move) > 1:
            print('Panic')

        counter.append(best_move[0])

    print(p2, counter, bintodec(p2)+1, bintodec(counter)+1)
