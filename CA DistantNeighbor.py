import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

latticeSize = 101
steps = 100
cellDist =10
SRules = [15]

cellA = latticeSize // 2
cellB = cellA + cellDist

def createRule(ruleNumber):
    ruleBinary = np.array([int(x) for x in np.binary_repr(ruleNumber, width=8)])

    def rule(left, center, right):
        return ruleBinary[7 - (left << 2 | center << 1 | right)]

    return rule

def evolve(grid, ruleFunc, cellA, cellB, considerDistantNeighbors=True):
    for t in range(1, steps):
        for i in range(latticeSize):
            left = grid[t-1][(i-1) % latticeSize]
            center = grid[t-1][i]
            right = grid[t-1][(i+1) % latticeSize]
            newState = ruleFunc(left, center, right)

            if considerDistantNeighbors:
                if i == cellA:
                    # newState ^= grid[t-1][cellB]
                    # Uncomment one of the following loc to change the operation
                     newState &= grid[t-1][cellB] #AND
                    # newState |= grid[t-1][cellB] #OR
                    # newState = ~(grid[t-1][cellB] & newState) & 1 #NAND
                    # newState = ~(grid[t-1][cellB] | newState) & 1 #NOR
                    # newState = ~(grid[t-1][cellB] ^ newState) & 1 #XNOR

            grid[t, i] = newState

def plotSTdiagrams(ruleNumber, gridOriginal, gridChanged, cellA, cellB):
    densitiesOriginal = np.mean(gridOriginal, axis=1)
    densitiesChanged = np.mean(gridChanged, axis=1)
    differencesBaselineVsChanged = np.sum(gridOriginal != gridChanged, axis=1)
    heatmapData = np.abs(gridOriginal - gridChanged)

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(gridOriginal, cmap='binary', interpolation='nearest')
    ax1.set_title(f'Original with Rule {ruleNumber}')
    ax1.set_xlabel('Cell Index')
    ax1.set_ylabel('Time Step')
    ax1.axvline(x=cellA, color='red', linestyle='--', label='Cell A')
    ax1.axvline(x=cellB, color='blue', linestyle='--', label='Cell B')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gridChanged, cmap='binary', interpolation='nearest')
    ax2.set_title(f'Changed with Rule {ruleNumber} and Distant Neighbors')
    ax2.set_xlabel('Cell Index')
    ax2.set_ylabel('Time Step')
    ax2.axvline(x=cellA, color='red', linestyle='--', label='Cell A')
    ax2.axvline(x=cellB, color='blue', linestyle='--', label='Cell B')

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(densitiesOriginal, label='Original', marker='o')
    ax3.plot(densitiesChanged, label='Changed', marker='x')
    ax3.set_title('Density of Active Cells Over Time')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Density of Active Cells')
    ax3.legend()
    ax3.grid(True)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(differencesBaselineVsChanged, marker='o')
    ax4.set_title(f'Difference Over Time for Rule {ruleNumber}')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Number of Differing Cells (vs Baseline)')
    ax4.grid(True)

    ax5 = fig.add_subplot(gs[2:, :])
    im = ax5.imshow(heatmapData, cmap='Reds', interpolation='nearest', aspect='auto')
    ax5.set_title(f'Differing Cells Heatmap: Original vs Changed for Rule {ruleNumber}')
    ax5.set_xlabel('Cell Index')
    ax5.set_ylabel('Time Step')
    ax5.axvline(x=cellA, color='red', linestyle='--', label='Cell A')
    ax5.axvline(x=cellB, color='blue', linestyle='--', label='Cell B')
    cbar = plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    cbar.set_label('Difference')

    ax1.legend()
    ax2.legend()
    ax5.legend()

    plt.tight_layout()
    plt.show()

def plotTransitionGraph(ruleNumber):
    ruleFunc = createRule(ruleNumber)
    configurations = []
    transitions = []
    for left in [0, 1]:
        for center in [0, 1]:
            for right in [0, 1]:
                nextState = ruleFunc(left, center, right)
                configurations.append((left, center, right))
                transitions.append(nextState)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, config in enumerate(configurations):
        left, center, right = config
        nextState = transitions[i]
        ax.text(0, -i, f'({left}, {center}, {right})', fontsize=12, va='center', ha='center', color='blue')
        ax.text(1, -i, f'→ {nextState}', fontsize=12, va='center', ha='center', color='red')

    ax.set_xlim(-1, 2)
    ax.set_ylim(-8, 1)
    ax.axis('off')
    plt.title(f'Transition Graph for Rule {ruleNumber}')
    plt.show()

initialCondition = np.zeros(latticeSize, dtype=int)
initialCondition[cellA] = 1
print("length=", len(initialCondition))
print(initialCondition)

for rule in SRules:
    gridOriginal = np.zeros((steps, latticeSize), dtype=int)
    gridChanged = np.zeros((steps, latticeSize), dtype=int)

    gridOriginal[0] = initialCondition
    gridChanged[0] = initialCondition.copy()

    ruleFunc = createRule(rule)
    evolve(gridOriginal, ruleFunc, cellA, cellB, False)
    evolve(gridChanged, ruleFunc, cellA, cellB)

    plotSTdiagrams(rule, gridOriginal, gridChanged, cellA, cellB)
    plotTransitionGraph(rule)
