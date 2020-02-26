import numpy as np


def sim(relas_a, relas_b):
    total = max(len(relas_a), len(relas_b))
    sim_count = len(relas_a.intersection(relas_b))
    return 1.0 * sim_count / total


def diff_score(relas_a, relas_b):
    return 1.0 - sim(relas_a, relas_b)


def rerank(images):
    N = len(images)
    if N == 0:
        return []
    div_scores = []
    diff_scores = [1]
    for i in range(1, N):
        diff_scores.append(diff_score(images[i], images[i - 1]))
    diff_score_idxs = np.argsort(np.array(diff_scores))[::-1]
    diff_rank = np.zeros(N)
    for i, idx in enumerate(diff_score_idxs):
        diff_rank[idx] = i + 1

    for i in range(N):
        div_scores.append(1.0 / (i + 1) + 1.0 / diff_rank[i])

    div_scores = np.array(div_scores)
    return np.argsort(div_scores)[::-1]

