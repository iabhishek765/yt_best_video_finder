"""Ranking evaluation metrics for query-grouped video candidates."""

from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score


def evaluate_ranking(
    videos,
    score_key="ranking_score",
    label_key="human_rating",
    query_key="query",
    k=5,
):
    """Evaluate ranking scores against human ratings, per query.

    Precision@k treats human ratings >= 4 as relevant.
    Queries with fewer than two valid videos are skipped.
    Returns aggregate metrics and per-query results.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    groups = defaultdict(list)

    for video in videos:
        if not isinstance(video, dict):
            raise TypeError("Every video must be a dictionary.")

        query = video.get(query_key)
        score = video.get(score_key)
        label = video.get(label_key)

        if query is None or score is None or label is None:
            continue

        try:
            score = float(score)
            label = float(label)
        except (TypeError, ValueError, OverflowError):
            continue

        if not np.isfinite(score) or not np.isfinite(label):
            continue

        groups[query].append((score, label))

    per_query = {}

    for query, rows in groups.items():
        if len(rows) < 2:
            continue

        scores = np.array([row[0] for row in rows], dtype=float)
        labels = np.array([row[1] for row in rows], dtype=float)

        ndcg = float(
            ndcg_score(
                [labels],
                [scores],
                k=min(k, len(rows)),
            )
        )

        order = np.argsort(-scores, kind="stable")
        top_labels = labels[order[:k]]
        precision = float(np.mean(top_labels >= 4.0))

        # Convert predicted order into positions, then compare against
        # human ratings. Ties in human ratings are handled by Kendall tau.
        predicted_positions = np.empty(len(rows), dtype=int)
        predicted_positions[order] = np.arange(len(rows))
        tau_result = kendalltau(-predicted_positions, labels)
        tau = (
            float(tau_result.statistic)
            if np.isfinite(tau_result.statistic)
            else None
        )

        per_query[query] = {
            "ndcg_at_k": ndcg,
            "precision_at_k": precision,
            "kendall_tau": tau,
            "video_count": len(rows),
        }

    if not per_query:
        return {
            "ndcg_at_k": None,
            "precision_at_k": None,
            "kendall_tau": None,
            "evaluated_queries": 0,
            "per_query": {},
        }

    ndcg_values = [result["ndcg_at_k"] for result in per_query.values()]
    precision_values = [
        result["precision_at_k"] for result in per_query.values()
    ]
    tau_values = [
        result["kendall_tau"]
        for result in per_query.values()
        if result["kendall_tau"] is not None
    ]

    return {
        "ndcg_at_k": float(np.mean(ndcg_values)),
        "precision_at_k": float(np.mean(precision_values)),
        "kendall_tau": (
            float(np.mean(tau_values)) if tau_values else None
        ),
        "evaluated_queries": len(per_query),
        "per_query": per_query,
    }