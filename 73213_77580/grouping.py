###############################################################################
## Computer Vision 2026-2027 - NOVA FCT
## Project Stage 1 - Grouping shots into scenes (given; Part 3, Task 7)
###############################################################################


def group_shots(similarity, threshold):
    """Group shots into scenes with union-find (connected components).

    similarity: n x n matrix (NumPy array or list of lists), similarity[a][b] = similarity between shots a and b
                (your Task 6 result, one row/column per shot, in the order of gt["shots"])
    threshold:  two shots are linked when similarity[a][b] >= threshold
    Returns a list of n integer scene labels: shots with the same label are in the same group. Linking is transitive
    (if a-b and b-c are linked, a, b and c form one group), and a shot linked to no other shot is a group on its own.

    How to call it, for one video:

        labels = grouping.group_shots(sim, threshold=0.9)          # sim: your shot x shot similarity matrix
        scenes = {str(s["id"]): label for s, label in zip(gt["shots"], labels)}   # the "scenes" field (Section 8)

    If your similarity is a distance instead (small = similar), pass its negative, e.g. group_shots(-dist, -0.1).
    """
    n = len(similarity)
    parent = list(range(n))

    def find(x):                     # root of x's tree, with path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(n):
        for b in range(a + 1, n):
            if similarity[a][b] >= threshold:
                parent[find(a)] = find(b)          # merge the two groups

    # relabel the roots as 0, 1, 2, ... in order of first appearance
    roots, labels = {}, []
    for i in range(n):
        labels.append(roots.setdefault(find(i), len(roots)))
    return labels
