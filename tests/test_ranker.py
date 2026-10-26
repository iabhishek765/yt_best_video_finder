import pytest

from src.ranker import rank_videos


@pytest.fixture
def sample_videos():
    return [
        {
            "video_id": "video_a",
            "original_rank": 1,
            "query": "python tutorial",
            "view_count": 1000,
            "like_count": 100,
            "comment_count": 20,
            "log_views_per_day": 2.0,
            "smoothed_like_ratio": 0.10,
            "comments_per_1000_views": 20.0,
        },
        {
            "video_id": "video_b",
            "original_rank": 2,
            "query": "python tutorial",
            "view_count": 5000,
            "like_count": 100,
            "comment_count": 10,
            "log_views_per_day": 4.0,
            "smoothed_like_ratio": 0.02,
            "comments_per_1000_views": 2.0,
        },
        {
            "video_id": "video_c",
            "original_rank": 3,
            "query": "python tutorial",
            "view_count": 500,
            "like_count": 50,
            "comment_count": 15,
            "log_views_per_day": 1.0,
            "smoothed_like_ratio": 0.09,
            "comments_per_1000_views": 30.0,
        },
    ]


def test_b0_preserves_youtube_order(sample_videos):
    ranked = rank_videos(sample_videos, baseline="B0")

    assert [v["video_id"] for v in ranked] == [
        "video_a",
        "video_b",
        "video_c",
    ]
    assert [v["rank"] for v in ranked] == [1, 2, 3]


def test_b1_sorts_by_views_descending(sample_videos):
    ranked = rank_videos(sample_videos, baseline="B1")

    assert [v["video_id"] for v in ranked] == [
        "video_b",
        "video_a",
        "video_c",
    ]


def test_b2_sorts_by_smoothed_like_ratio(sample_videos):
    ranked = rank_videos(sample_videos, baseline="B2")

    assert [v["video_id"] for v in ranked] == [
        "video_a",
        "video_c",
        "video_b",
    ]


def test_b3_returns_scores_and_ranks(sample_videos):
    ranked = rank_videos(sample_videos, baseline="B3")

    assert len(ranked) == 3
    assert [v["rank"] for v in ranked] == [1, 2, 3]
    assert all(v["ranking_score"] is not None for v in ranked)


def test_missing_views_are_ranked_last():
    videos = [
        {"video_id": "missing", "original_rank": 1, "view_count": None},
        {"video_id": "valid", "original_rank": 2, "view_count": 100},
    ]

    ranked = rank_videos(videos, baseline="B1")

    assert [v["video_id"] for v in ranked] == ["valid", "missing"]
    assert ranked[1]["ranking_score"] is None


def test_input_records_are_not_modified(sample_videos):
    original = [dict(video) for video in sample_videos]

    rank_videos(sample_videos, baseline="B3")

    assert sample_videos == original
    assert all("rank" not in video for video in sample_videos)


def test_unknown_baseline_is_rejected(sample_videos):
    with pytest.raises(ValueError, match="Unknown baseline"):
        rank_videos(sample_videos, baseline="B9")


def test_b3_rejects_unknown_weight_names(sample_videos):
    with pytest.raises(ValueError, match="Unsupported B3 weight names"):
        rank_videos(
            sample_videos,
            baseline="B3",
            weights={"unknown_feature": 1.0},
        )