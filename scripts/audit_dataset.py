
import json
from collections import Counter, defaultdict
from pathlib import Path

DATA_PATH = Path("data/raw/candidate_videos.json")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    with DATA_PATH.open(encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Expected the dataset to be a JSON list.")

    print(f"Total records: {len(records)}")

    query_counts = Counter(
        record.get("query_id") for record in records
    )
    print(f"Unique queries: {len(query_counts)}")
    print("\nRecords per query:")
    for query_id, count in sorted(query_counts.items()):
        print(f"  {query_id}: {count}")

    missing_ids = [
        i for i, record in enumerate(records)
        if not record.get("video_id")
    ]
    missing_ranks = [
        i for i, record in enumerate(records)
        if record.get("original_rank") is None
    ]

    print(f"\nMissing video IDs: {len(missing_ids)}")
    print(f"Missing original ranks: {len(missing_ranks)}")

    # Group records by query and video ID.
    grouped = defaultdict(list)
    for record in records:
        key = (record.get("query_id"), record.get("video_id"))
        grouped[key].append(record)

    duplicate_groups = {
        key: items
        for key, items in grouped.items()
        if key[1] and len(items) > 1
    }

    repeated_rows = sum(
        len(items) - 1 for items in duplicate_groups.values()
    )

    print(f"\nDuplicate query/video groups: {len(duplicate_groups)}")
    print(f"Extra repeated rows: {repeated_rows}")

    for (query_id, video_id), items in sorted(
        duplicate_groups.items()
    ):
        ranks = [item.get("original_rank") for item in items]
        titles = [item.get("title", "") for item in items]
        print(
            f"  Query={query_id}, video_id={video_id}, "
            f"ranks={ranks}, titles={titles}"
        )

    # Check whether IDs recur across different queries.
    video_queries = defaultdict(set)
    for record in records:
        video_id = record.get("video_id")
        if video_id:
            video_queries[video_id].add(record.get("query_id"))

    cross_query_reuse = {
        video_id: queries
        for video_id, queries in video_queries.items()
        if len(queries) > 1
    }
    print(
        "\nVideo IDs appearing across multiple queries:",
        len(cross_query_reuse),
    )

    required_fields = [
        "query_id",
        "video_id",
        "original_rank",
        "title",
        "published_at",
        "duration",
        "view_count",
        "like_count",
        "comment_count",
    ]

    print("\nMissing required-field values:")
    for field in required_fields:
        missing = sum(
            record.get(field) is None or record.get(field) == ""
            for record in records
        )
        print(f"  {field}: {missing}")

    print("\nAudit complete. No records were modified.")


if __name__ == "__main__":
    main()
