import math
from datetime import UTC, datetime

import pytest

from src.features import build_video_features, normalize_features_by_query

FIXED_NOW = datetime(2026, 10, 25, tzinfo=UTC)


def test_normal_metadata_produces_expected_features():
    videos = [
        {
            "video_id": "video1",
            "original_rank": 3,
            "title": "Python Tutorial",
            "published_at": "2026-10-15T00:00:00Z",
            "duration": "PT10M",
            "view_count": "1000",
            "like_count": "100",
            "comment_count": "20",
        }
    ]

    result = build_video_features(videos, as_of=FIXED_NOW)[0]

    assert result["age_days"] == pytest.approx(10)
    assert result["duration_seconds"] == 600
    assert result["log_views"] == pytest.approx(math.log1p(1000))
    assert result["log_views_per_day"] == pytest.approx(math.log1p(100))
    assert result["smoothed_like_ratio"] == pytest.approx(101 / 1100)
    assert result["comments_per_1000_views"] == pytest.approx(20)


def test_missing_metadata_does_not_crash():
    result = build_video_features(
        [{"video_id": "video2", "original_rank": 1}],
        as_of=FIXED_NOW,
    )[0]

    assert result["age_days"] is None
    assert result["duration_seconds"] is None
    assert result["log_views"] is None
    assert result["log_views_per_day"] is None
    assert result["smoothed_like_ratio"] is None
    assert result["comments_per_1000_views"] is None


def test_zero_views_does_not_divide_by_zero():
    videos = [
        {
            "video_id": "video3",
            "published_at": "2026-10-20T00:00:00Z",
            "duration": "PT0S",
            "view_count": 0,
            "like_count": 0,
            "comment_count": 0,
        }
    ]

    result = build_video_features(videos, as_of=FIXED_NOW)[0]

    assert result["log_views"] == 0
    assert result["log_views_per_day"] == pytest.approx(0)
    assert result["smoothed_like_ratio"] == pytest.approx(0.01)
    assert result["comments_per_1000_views"] is None
    assert result["duration_seconds"] == 0


def test_input_metadata_and_original_rank_are_preserved():
    original = {
        "video_id": "video4",
        "original_rank": 7,
        "title": "Learn Python",
        "view_count": 500,
        "like_count": 25,
        "comment_count": 5,
        "published_at": "2026-10-20T00:00:00Z",
        "duration": "PT5M",
    }
    original_copy = original.copy()

    result = build_video_features(
        [original],
        as_of=FIXED_NOW,
    )[0]

    assert result["original_rank"] == 7
    assert result["title"] == "Learn Python"
    assert result["video_id"] == "video4"
    assert original == original_copy
    assert result is not original


def test_invalid_counts_and_dates_are_handled_safely():
    videos = [
        {
            "video_id": "video5",
            "published_at": "not-a-date",
            "duration": "not-a-duration",
            "view_count": "unknown",
            "like_count": -10,
            "comment_count": None,
        }
    ]

    result = build_video_features(videos, as_of=FIXED_NOW)[0]

    assert result["age_days"] is None
    assert result["duration_seconds"] is None
    assert result["log_views"] is None
    assert result["log_views_per_day"] is None
    assert result["smoothed_like_ratio"] is None
    assert result["comments_per_1000_views"] is None


def test_normalization_is_per_query():
    videos = [
        {"query": "A", "original_rank": 1, "log_views": 10},
        {"query": "A", "original_rank": 2, "log_views": 20},
        {"query": "B", "original_rank": 1, "log_views": 100},
        {"query": "B", "original_rank": 2, "log_views": 300},
    ]

    result = normalize_features_by_query(
        videos,
        feature_names=["log_views"],
    )

    assert [v["norm_log_views"] for v in result] == pytest.approx([0.0, 1.0, 0.0, 1.0])
    assert [v["log_views"] for v in result] == [10, 20, 100, 300]
    assert [v["original_rank"] for v in result] == [1, 2, 1, 2]
    assert videos[0]["log_views"] == 10


def test_normalization_handles_missing_and_constant_values():
    videos = [
        {"query": "A", "score": 5},
        {"query": "A", "score": None},
        {"query": "A", "score": 5},
    ]

    result = normalize_features_by_query(
        videos,
        feature_names=["score"],
    )

    assert result[0]["norm_score"] == 0.0
    assert result[1]["norm_score"] is None
    assert result[2]["norm_score"] == 0.0
    assert [v["score"] for v in result] == [5, None, 5]
