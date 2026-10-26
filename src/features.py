"""Build metadata-based features for YouTube video ranking."""

import math
from datetime import UTC, datetime

import isodate


def _parse_datetime(value):
    """Parse a YouTube publication timestamp into a timezone-aware datetime."""
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value)

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)

        return parsed.astimezone(UTC)
    except (TypeError, ValueError):
        return None


def _parse_duration_seconds(value):
    """Convert an ISO 8601 duration such as PT10M into seconds."""
    if not value:
        return None

    try:
        duration = isodate.parse_duration(value)
        seconds = duration.total_seconds()

        if not math.isfinite(seconds) or seconds < 0:
            return None

        return int(seconds)
    except (TypeError, ValueError, OverflowError):
        return None


def _non_negative_number(value):
    """Convert a count to a non-negative number, or return None."""
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(number) or number < 0:
        return None

    return number


def build_video_features(videos, as_of=None):
    """Return copies of video records enriched with metadata features.

    Missing or invalid source values produce None for dependent features.
    The original records are not modified.
    """
    if as_of is None:
        as_of = datetime.now(UTC)
    elif as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
    else:
        as_of = as_of.astimezone(UTC)

    enriched_videos = []

    for video in videos:
        result = dict(video)

        published_at = _parse_datetime(video.get("published_at"))
        views = _non_negative_number(video.get("view_count"))
        likes = _non_negative_number(video.get("like_count"))
        comments = _non_negative_number(video.get("comment_count"))

        duration_seconds = _parse_duration_seconds(video.get("duration"))

        if published_at is not None:
            age_days = max(
                0.0,
                (as_of - published_at).total_seconds() / 86400,
            )
        else:
            age_days = None

        if views is not None:
            log_views = math.log1p(views)
        else:
            log_views = None

        if views is not None and age_days is not None:
            log_views_per_day = math.log1p(views / max(age_days, 1.0))
        else:
            log_views_per_day = None

        # A small prior stabilizes the ratio for videos with few views.
        # Missing counts stay missing rather than being treated as zero.
        if views is not None and likes is not None:
            smoothed_like_ratio = (likes + 1.0) / (views + 100.0)
        else:
            smoothed_like_ratio = None

        if views is not None and comments is not None:
            comments_per_1000_views = comments / views * 1000.0 if views > 0 else None
        else:
            comments_per_1000_views = None

        result.update(
            {
                "age_days": age_days,
                "duration_seconds": duration_seconds,
                "log_views": log_views,
                "log_views_per_day": log_views_per_day,
                "smoothed_like_ratio": smoothed_like_ratio,
                "comments_per_1000_views": comments_per_1000_views,
            }
        )

        enriched_videos.append(result)

    return enriched_videos


def normalize_features_by_query(
    videos,
    feature_names=None,
    query_key="query",
):
    """Min-max normalize numeric features within each query.

    Missing values remain None. Constant-valued features become 0.0.
    Input dictionaries are not modified.
    """
    if feature_names is None:
        feature_names = [
            "age_days",
            "duration_seconds",
            "log_views",
            "log_views_per_day",
            "smoothed_like_ratio",
            "comments_per_1000_views",
        ]

    normalized = [dict(video) for video in videos]
    query_groups = {}

    for index, video in enumerate(normalized):
        query = video.get(query_key)
        query_groups.setdefault(query, []).append(index)

    for indices in query_groups.values():
        for feature in feature_names:
            valid_values = {}

            for index in indices:
                value = _non_negative_number(normalized[index].get(feature))
                if value is not None:
                    valid_values[index] = value

            if not valid_values:
                for index in indices:
                    normalized[index][f"norm_{feature}"] = None
                continue

            minimum = min(valid_values.values())
            maximum = max(valid_values.values())
            spread = maximum - minimum

            for index in indices:
                if index not in valid_values:
                    normalized[index][f"norm_{feature}"] = None
                elif spread == 0:
                    normalized[index][f"norm_{feature}"] = 0.0
                else:
                    normalized[index][f"norm_{feature}"] = (
                        valid_values[index] - minimum
                    ) / spread

    return normalized
