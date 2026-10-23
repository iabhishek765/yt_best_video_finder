
import json
import sqlite3
from datetime import datetime, timezone

import pytest

from src.collector import YouTubeCollector


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
