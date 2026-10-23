
import os

from dotenv import load_dotenv
from googleapiclient.discovery import build


def main():
    load_dotenv()

    api_key = os.getenv("YT_API_KEY")
    if not api_key or api_key == "paste_your_actual_key_here":
        raise RuntimeError(
            "Set your real YT_API_KEY in the project-root .env file."
        )

    youtube = build(
        "youtube",
        "v3",
        developerKey=api_key,
    )

    response = youtube.search().list(
        part="snippet",
        q="learn pandas for beginners",
        type="video",
        maxResults=1,
    ).execute()

    items = response.get("items", [])
    if not items:
        print("API connected, but the query returned no videos.")
        return

    video = items[0]
    print("API connection successful!")
    print("Title:", video["snippet"]["title"])
    print("Video ID:", video["id"]["videoId"])
    print(
        "URL:",
        f"https://www.youtube.com/watch?v={video['id']['videoId']}",
    )


if __name__ == "__main__":
    main()
