import json
import sqlite3

import pytest

from src.collector import YouTubeCollector
from datetime import datetime, timedelta, timezone


def make_collector(tmp_path):
    collector = YouTubeCollector.__new__(YouTubeCollector)
    collector.db_path = tmp_path / "test_cache.db"
    collector.db_path.parent.mkdir(parents=True, exist_ok=True)
    collector._initialize_database()
    return collector


def test_empty_query_is_rejected(tmp_path):
    collector = make_collector(tmp_path)

    with pytest.raises(ValueError, match="non-empty string"):
        collector.collect_videos("   ")


@pytest.mark.parametrize("count", [0, 101, -1])
def test_invalid_result_count_is_rejected(tmp_path, count):
    collector = make_collector(tmp_path)

    with pytest.raises(ValueError, match="between 1 and 100"):
        collector.collect_videos("machine learning", count)


def test_cache_round_trip(tmp_path):
    collector = make_collector(tmp_path)
    videos = [
        {
            "video_id": "abc123",
            "original_rank": 1,
            "title": "Intro to Machine Learning",
        }
    ]

    collector._write_cache("machine learning|1", "machine learning", videos)

    cached = collector._read_cache("machine learning|1")
    assert cached == videos


def test_cache_is_stored_in_sqlite(tmp_path):
    collector = make_collector(tmp_path)
    videos = [{"video_id": "abc123", "original_rank": 1}]

    collector._write_cache("test|1", "test", videos)

    with sqlite3.connect(collector.db_path) as conn:
        row = conn.execute(
            "SELECT query, payload_json FROM query_cache WHERE query_key = ?",
            ("test|1",),
        ).fetchone()

    assert row[0] == "test"
    assert json.loads(row[1]) == videos


def test_collect_videos_reuses_cache(monkeypatch, tmp_path):
    collector = make_collector(tmp_path)

    calls = {"search": 0, "details": 0}

    def fake_search(query, max_results):
        calls["search"] += 1
        return [
            {
                "video_id": "abc123",
                "original_rank": 1,
                "search_title": "Intro to Python",
            }
        ]

    def fake_details(video_ids):
        calls["details"] += 1
        return {
            "abc123": {
                "id": "abc123",
                "snippet": {
                    "title": "Intro to Python",
                    "description": "A Python tutorial",
                    "channelId": "channel123",
                    "channelTitle": "Example Channel",
                    "publishedAt": "2025-01-01T00:00:00Z",
                },
                "statistics": {
                    "viewCount": "100",
                },
                "contentDetails": {
                    "duration": "PT10M",
                },
            }
        }

    monkeypatch.setattr(collector, "_search_video_ids", fake_search)
    monkeypatch.setattr(collector, "_get_video_details", fake_details)

    first = collector.collect_videos("python tutorial", max_results=1)
    second = collector.collect_videos("python tutorial", max_results=1)

    assert first == second
    assert len(first) == 1
    assert calls == {"search": 1, "details": 1}


def test_missing_optional_metadata_is_preserved(monkeypatch, tmp_path):
    collector = make_collector(tmp_path)

    monkeypatch.setattr(
        collector,
        "_search_video_ids",
        lambda query, max_results: [
            {
                "video_id": "abc123",
                "original_rank": 1,
                "search_title": "Intro to Python",
            }
        ],
    )

    monkeypatch.setattr(
        collector,
        "_get_video_details",
        lambda video_ids: {
            "abc123": {
                "id": "abc123",
                "snippet": {
                    "title": "Intro to Python",
                    "description": "",
                    "channelId": "channel123",
                    "channelTitle": "Example Channel",
                    "publishedAt": "2025-01-01T00:00:00Z",
                },
                "statistics": {},
                "contentDetails": {},
            }
        },
    )

    videos = collector.collect_videos("python tutorial", max_results=1)

    assert len(videos) == 1
    assert videos[0]["view_count"] is None
    assert videos[0]["like_count"] is None
    assert videos[0]["comment_count"] is None
    assert videos[0]["duration"] is None


def test_expired_cache_is_not_reused(tmp_path):
    collector = make_collector(tmp_path)

    query_key = "machine learning|1"
    query = "machine learning"
    videos = [{"video_id": "abc123"}]

    collector._write_cache(query_key, query, videos)

    expired_time = (
        datetime.now(timezone.utc) - timedelta(hours=25)
    ).isoformat()

    with sqlite3.connect(collector.db_path) as conn:
        conn.execute(
            """
            UPDATE query_cache
            SET fetched_at = ?
            WHERE query_key = ?
            """,
            (expired_time, query_key),
        )

    assert collector._read_cache(query_key) is None
