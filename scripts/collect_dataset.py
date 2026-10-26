
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.collector import YouTubeCollector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERY_FILE = PROJECT_ROOT / "data" / "labeled" / "queries.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "candidate_videos.json"


def main():
    parser = argparse.ArgumentParser(
        description="Collect a reproducible YouTube candidate dataset."
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=3,
        help="Number of queries to process (default: 3).",
    )
    parser.add_argument(
        "--videos-per-query",
        type=int,
        default=5,
        help="Maximum candidate videos per query (default: 5).",
    )
    args = parser.parse_args()

    if args.limit_queries < 1:
        parser.error("--limit-queries must be at least 1.")
    if not 1 <= args.videos_per_query <= 100:
        parser.error("--videos-per-query must be between 1 and 100.")

    queries = pd.read_csv(QUERY_FILE)

    required_columns = {
        "query_id",
        "query",
        "intent",
        "language",
    }
    missing = required_columns - set(queries.columns)
    if missing:
        raise ValueError(f"Missing query columns: {sorted(missing)}")

    queries = queries.head(args.limit_queries)
    collector = YouTubeCollector()
    records = []

    print(f"Queries selected: {len(queries)}")
    print(f"Maximum videos per query: {args.videos_per_query}")

    for index, row in queries.iterrows():
        query_id = str(row["query_id"])
        query_text = str(row["query"])

        print(
            f"\n[{index + 1}/{len(queries)}] "
            f"{query_id}: {query_text}"
        )

        try:
            videos = collector.collect_videos(
                query_text,
                max_results=args.videos_per_query,
                use_cache=True,
            )

            for video in videos:
                record = dict(video)
                record["query_id"] = query_id
                record["intent"] = str(row["intent"])
                record["language"] = str(row["language"])
                records.append(record)

            print(f"Collected: {len(videos)} videos")

        except Exception as exc:
            print(
                f"FAILED for {query_id}: "
                f"{type(exc).__name__}: {exc}"
            )

        # Save after every query so progress survives interruption.
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = OUTPUT_FILE.with_suffix(".tmp")
        temporary_file.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_file.replace(OUTPUT_FILE)

        time.sleep(0.2)

    print("\nCollection run finished.")
    print(f"Records saved: {len(records)}")
    print(f"Output: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    print("Original YouTube ranks are retained in each video record.")


if __name__ == "__main__":
    main()
