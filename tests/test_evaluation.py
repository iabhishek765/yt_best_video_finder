import pytest

from src.evaluation import evaluate_ranking


def test_perfect_ranking_scores():
    videos = [
        {"query": "python", "ranking_score": 3, "human_rating": 5},
        {"query": "python", "ranking_score": 2, "human_rating": 4},
        {"query": "python", "ranking_score": 1, "human_rating": 2},
    ]

    result = evaluate_ranking(videos, k=2)

    assert result["ndcg_at_k"] == pytest.approx(1.0)
    assert result["precision_at_k"] == pytest.approx(1.0)
    assert result["kendall_tau"] == pytest.approx(1.0)
    assert result["evaluated_queries"] == 1


def test_poor_ranking_has_lower_ndcg():
    videos = [
        {"query": "python", "ranking_score": 3, "human_rating": 1},
        {"query": "python", "ranking_score": 2, "human_rating": 2},
        {"query": "python", "ranking_score": 1, "human_rating": 5},
    ]

    result = evaluate_ranking(videos, k=2)

    assert result["ndcg_at_k"] < 1.0
    assert result["precision_at_k"] == pytest.approx(0.0)
    assert result["kendall_tau"] < 0.0


def test_queries_are_evaluated_separately():
    videos = [
        {"query": "python", "ranking_score": 2, "human_rating": 5},
        {"query": "python", "ranking_score": 1, "human_rating": 3},
        {"query": "sql", "ranking_score": 2, "human_rating": 4},
        {"query": "sql", "ranking_score": 1, "human_rating": 2},
    ]

    result = evaluate_ranking(videos)

    assert result["evaluated_queries"] == 2
    assert set(result["per_query"]) == {"python", "sql"}


def test_missing_ratings_are_skipped():
    videos = [
        {"query": "python", "ranking_score": 3, "human_rating": 5},
        {"query": "python", "ranking_score": 2, "human_rating": None},
        {"query": "python", "ranking_score": 1, "human_rating": 2},
    ]

    result = evaluate_ranking(videos)

    assert result["evaluated_queries"] == 1
    assert result["per_query"]["python"]["video_count"] == 2


def test_no_valid_query_returns_none_metrics():
    result = evaluate_ranking(
        [{"query": "python", "ranking_score": 1, "human_rating": 5}]
    )

    assert result["evaluated_queries"] == 0
    assert result["ndcg_at_k"] is None


def test_invalid_k_is_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_ranking([], k=0)