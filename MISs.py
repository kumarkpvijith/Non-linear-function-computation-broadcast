# https://github.com/kumarkpvijith/Non-linear-function-computation-broadcast.git

import numpy as np

n_RV = 3
F_q = 3


def F1(x1, x2, x3):
    return (y + z) % F_q


def F2(x1, x2, x3):
    return (x + z) % F_q


def F3(x1, x2, x3):
    return (x + 2 * y) % F_q


matrix = []

for x in range(0, F_q):
    for y in range(0, F_q):
        for z in range(0, F_q):
            matrix.append([x, y, z, F1(x, y, z), F2(x, y, z), F3(x, y, z)])

Out = [0, 0, 0, F1(0, 0, 0), F2(0, 0, 0), F3(0, 0, 0)]

for row in matrix:
    Out = np.vstack([Out, row])
Out = np.delete(Out, 0, 0)
# print(Out)

X_0 = np.array(np.where(Out[:, 0] == 0))
XX = np.zeros((F_q, F_q ** (n_RV - 1)))
for i in range(0, F_q):
    XX[i, :] = np.array(np.where(Out[:, 0] == i))

YY = np.zeros((F_q, F_q ** (n_RV - 1)))
for i in range(0, F_q):
    YY[i, :] = np.array(np.where(Out[:, 1] == i))

ZZ = np.zeros((F_q, F_q ** (n_RV - 1)))
for i in range(0, F_q):
    ZZ[i, :] = np.array(np.where(Out[:, 2] == i))

XX = XX.astype(int)
YY = YY.astype(int)
ZZ = ZZ.astype(int)

for a in range(0, F_q):
    X_0 = np.array(XX[a])
Adj_X = np.zeros((F_q ** n_RV, F_q ** n_RV))

for a in range(0, F_q):
    X_0 = np.array([XX[a]])
    for i in range(0, X_0.shape[1]):
        for j in range(0, X_0.shape[1]):
            if Out[X_0[0, j], 3] != Out[X_0[0, i], 3]:
                Adj_X[X_0[0, i], X_0[0, j]] += 1

Adj_Y = np.zeros((F_q ** n_RV, F_q ** n_RV))

for a in range(0, F_q):
    Y_0 = np.array([YY[a]])
    for i in range(0, Y_0.shape[1]):
        for j in range(0, Y_0.shape[1]):
            if Out[Y_0[0, j], 4] != Out[Y_0[0, i], 4]:
                Adj_Y[Y_0[0, i], Y_0[0, j]] += 1

Adj_Z = np.zeros((F_q ** n_RV, F_q ** n_RV))
for a in range(0, F_q):
    Z_0 = np.array([ZZ[a]])
    for i in range(0, Z_0.shape[1]):
        for j in range(0, Z_0.shape[1]):
            if Out[Z_0[0, j], 5] != Out[Z_0[0, i], 5]:
                Adj_Z[Z_0[0, i], Z_0[0, j]] += 1

Adj_XYZ = Adj_X + Adj_Y + Adj_Z
Adj_XYZ[Adj_XYZ > 0] = 1


def is_independent_set(matrix, vertices):
    """Check if the set of vertices is an independent set."""
    for i, v1 in enumerate(vertices):
        for v2 in vertices[i + 1:]:
            if matrix[v1][v2] == 1:
                return False
    return True


def find_independent_sets(matrix, current_set, start, all_sets):
    """Recursive function to find all maximally independent sets."""
    for vertex in range(start, len(matrix)):
        # Prune if adding this vertex makes the set not independent
        if all(matrix[vertex][v] == 0 for v in current_set):
            new_set = current_set + [vertex]
            all_sets.append(new_set)
            find_independent_sets(matrix, new_set, vertex + 1, all_sets)


def filter_maximal_sets(all_sets):
    """Filter to keep only maximally independent sets."""
    maximal_sets = []
    sorted_sets = sorted(all_sets, key=len, reverse=True)
    for s in sorted_sets:
        if not any(set(s) < set(other) for other in maximal_sets):
            maximal_sets.append(s)
    return maximal_sets


def maximal_independent_sets(matrix):
    """Find all maximally independent sets of a graph represented by an adjacency matrix."""
    all_sets = []
    find_independent_sets(matrix, [], 0, all_sets)
    return filter_maximal_sets(all_sets)


adjacency_matrix = Adj_XYZ

# Find maximally independent sets
independent_sets = maximal_independent_sets(adjacency_matrix)

print("Maximally independent sets:", independent_sets)
print("Number of Maximally independent sets:",len(independent_sets))