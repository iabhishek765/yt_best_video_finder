
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "raw" / "youtube_cache.db"
CACHE_HOURS = 24


class YouTubeCollector:
    """Collect YouTube search candidates and cache results in SQLite."""

    def __init__(self, api_key=None, db_path=DEFAULT_DB_PATH):
        load_dotenv(PROJECT_ROOT / ".env")

        self.api_key = api_key or os.getenv("YT_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "YT_API_KEY is missing. Add it to the project-root .env file."
            )

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.youtube = build(
            "youtube",
            "v3",
            developerKey=self.api_key,
            cache_discovery=False,
        )

        self._initialize_database()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _initialize_database(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_key TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _normalize_query(query):
        return " ".join(query.strip().lower().split())

    def _read_cache(self, query_key):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT fetched_at, payload_json
                FROM query_cache
                WHERE query_key = ?
                """,
                (query_key,),
            ).fetchone()

        if row is None:
            return None

        fetched_at = datetime.fromisoformat(row[0])
        if datetime.now(timezone.utc) - fetched_at > timedelta(
            hours=CACHE_HOURS
        ):
            return None

        return json.loads(row[1])

    def _write_cache(self, query_key, query, videos):
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(videos, ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO query_cache
                    (query_key, query, fetched_at, payload_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(query_key) DO UPDATE SET
                    query = excluded.query,
                    fetched_at = excluded.fetched_at,
                    payload_json = excluded.payload_json
                """,
                (query_key, query, now, payload),
            )

    def _search_video_ids(self, query, max_results):
        video_ids = []
        next_page_token = None

        while len(video_ids) < max_results:
            page_size = min(50, max_results - len(video_ids))

            response = self.youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=page_size,
                pageToken=next_page_token,
                order="relevance",
            ).execute()

            for item in response.get("items", []):
                video_id = item.get("id", {}).get("videoId")
                if video_id:
                    video_ids.append(
                        {
                            "video_id": video_id,
                            "original_rank": len(video_ids) + 1,
                            "search_title": item["snippet"]["title"],
                        }
                    )

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        return video_ids

    def _get_video_details(self, video_ids):
        details = {}

        # videos.list accepts up to 50 IDs per request.
        for start in range(0, len(video_ids), 50):
            batch = video_ids[start : start + 50]

            response = self.youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(batch),
                maxResults=len(batch),
            ).execute()

            for item in response.get("items", []):
                details[item["id"]] = item

        return details

    def collect_videos(self, query, max_results=50, use_cache=True):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not isinstance(max_results, int) or isinstance(max_results, bool):
            raise ValueError("max_results must be an integer from 1 to 100.")

        if not 1 <= max_results <= 100:
            raise ValueError("max_results must be between 1 and 100.")

        query = query.strip()
        query_key = self._normalize_query(query) + f"|{max_results}"

        if use_cache:
            cached = self._read_cache(query_key)
            if cached is not None:
                return cached

        candidates = self._search_video_ids(query, max_results)
        ids = [candidate["video_id"] for candidate in candidates]
        details = self._get_video_details(ids) if ids else {}

        collected_at = datetime.now(timezone.utc).isoformat()
        videos = []

        for candidate in candidates:
            video_id = candidate["video_id"]
            item = details.get(video_id)

            # A video can disappear between search and detail retrieval.
            if item is None:
                continue

            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})
            content = item.get("contentDetails", {})

            videos.append(
                {
                    "video_id": video_id,
                    "original_rank": candidate["original_rank"],
                    "query": query,
                    "collected_at": collected_at,
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "channel_id": snippet.get("channelId", ""),
                    "channel_title": snippet.get("channelTitle", ""),
                    "published_at": snippet.get("publishedAt"),
                    "duration": content.get("duration"),
                    "view_count": statistics.get("viewCount"),
                    "like_count": statistics.get("likeCount"),
                    "comment_count": statistics.get("commentCount"),
                    "video_url": (
                        f"https://www.youtube.com/watch?v={video_id}"
                    ),
                }
            )

        self._write_cache(query_key, query, videos)
        return videos
