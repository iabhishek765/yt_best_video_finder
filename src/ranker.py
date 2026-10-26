"""Simple ranking baselines for YouTube video candidates."""

from src.features import normalize_features_by_query


BASELINES = ("B0", "B1", "B2", "B3")

# Initial design weights for the metadata-only engagement baseline.
DEFAULT_WEIGHTS = {
    "log_views_per_day": 0.40,
    "smoothed_like_ratio": 0.35,
    "comments_per_1000_views": 0.25,
}


def _number(value):
    """Return a finite, non-negative number or None."""
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if number < 0 or number != number or number == float("inf"):
        return None
    if number == float("-inf"):
        return None

    return number


def _original_rank(video, fallback):
    """Use YouTube's original position to break ties consistently."""
    rank = _number(video.get("original_rank"))
    return rank if rank is not None else fallback


def _like_ratio(video):
    """Use the feature value, or calculate the project's smoothed ratio."""
    ratio = _number(video.get("smoothed_like_ratio"))
    if ratio is not None:
        return ratio

    views = _number(video.get("view_count"))
    likes = _number(video.get("like_count"))

    if views is None or likes is None:
        return None

    return (likes + 1.0) / (views + 100.0)


def rank_videos(videos, baseline="B0", weights=None):
    """Return copied video records sorted by a selected baseline.

    B0: original YouTube search order.
    B1: view count, descending.
    B2: smoothed like ratio, descending.
    B3: weighted metadata feature percentiles, descending.

    Each result includes `ranking_score` and `rank`.
    The input list and its dictionaries are not modified.
    """
    if baseline not in BASELINES:
        raise ValueError(
            f"Unknown baseline {baseline!r}. Choose one of {BASELINES}."
        )

    if not isinstance(videos, (list, tuple)):
        raise TypeError("videos must be a list or tuple of dictionaries.")

    for video in videos:
        if not isinstance(video, dict):
            raise TypeError("Every video must be a dictionary.")

    copied = [dict(video) for video in videos]

    if baseline == "B3":
        selected_weights = dict(DEFAULT_WEIGHTS)
        if weights is not None:
            unknown = set(weights) - set(selected_weights)
            if unknown:
                raise ValueError(
                    f"Unsupported B3 weight names: {sorted(unknown)}"
                )
            for feature, weight in weights.items():
                numeric_weight = _number(weight)
                if numeric_weight is None:
                    raise ValueError(
                        f"Weight for {feature!r} must be non-negative."
                    )
                selected_weights[feature] = numeric_weight

        if sum(selected_weights.values()) <= 0:
            raise ValueError("At least one B3 weight must be positive.")

        # Calculate the metadata features first, if they are not already present.
        for video in copied:
            if "smoothed_like_ratio" not in video:
                video["smoothed_like_ratio"] = _like_ratio(video)

        normalized = normalize_features_by_query(
            copied,
            feature_names=list(selected_weights),
        )

        for video in normalized:
            weighted_sum = 0.0
            available_weight = 0.0

            for feature, weight in selected_weights.items():
                value = _number(video.get(f"norm_{feature}"))
                if value is not None and weight > 0:
                    weighted_sum += value * weight
                    available_weight += weight

            video["ranking_score"] = (
                weighted_sum / available_weight
                if available_weight > 0
                else None
            )

        copied = normalized

    else:
        for index, video in enumerate(copied):
            if baseline == "B0":
                video["ranking_score"] = -_original_rank(video, index + 1)
            elif baseline == "B1":
                video["ranking_score"] = _number(video.get("view_count"))
            elif baseline == "B2":
                video["ranking_score"] = _like_ratio(video)

    indexed = list(enumerate(copied))

    if baseline == "B0":
        indexed.sort(
            key=lambda pair: (
                _original_rank(pair[1], pair[0] + 1),
                pair[0],
            )
        )
    else:
        indexed.sort(
            key=lambda pair: (
                pair[1].get("ranking_score") is None,
                -pair[1]["ranking_score"]
                if pair[1].get("ranking_score") is not None
                else 0.0,
                _original_rank(pair[1], pair[0] + 1),
                pair[0],
            )
        )

    results = []
    for position, (_, video) in enumerate(indexed, start=1):
        video["rank"] = position
        results.append(video)

    return results