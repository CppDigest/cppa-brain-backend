"""Fetch Boost mailing list emails from API and save to JSON files."""

import json
import requests
import sys
import os
import time
from typing import Dict, Optional, List


def _filter_by_date(
    results: List[Dict], start: str, end: str
) -> tuple[List[Dict], bool]:
    """Filter results by date range. Returns (filtered, should_stop)."""
    filtered, stop = [], False
    for r in results:
        d = r.get("date")
        if start and d and d < start:
            stop = True
            break
        if end and d and d > end:
            continue
        filtered.append(r)
    return filtered, stop


def _fetch_page(url: str, page: int, max_retries: int = 3) -> Optional[Dict]:
    """Fetch a single API page with retry logic for 429 errors."""
    url_with_params = f"{url}?limit=100&offset={(page-1)*100}"
    delay_seconds_default = 10  # move to config

    for attempt in range(max_retries):
        try:
            resp = requests.get(url_with_params, timeout=30)

            if resp.status_code == 429:
                retry_after = delay_seconds_default

                # Try to get retry_after from headers
                if "Retry-After" in resp.headers:
                    retry_after = int(resp.headers["Retry-After"])
                else:
                    try:
                        body = resp.json()
                        retry_after = body.get("retry_after") or body.get("retry-after")
                        if retry_after:
                            retry_after = int(retry_after)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass

                if attempt == max_retries - 1:
                    return None
                print(
                    f"Rate limited on page {page}, waiting {retry_after}s before retry {attempt+1}/{max_retries}"
                )
                time.sleep(retry_after)
                continue

            # Check for other errors
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                # Already handled above, but catch any edge cases
                continue
            print(f"HTTP error fetching page {page}: {e}")
            return None
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching page {page}: {e}")
            return None

    return None


def fetch_api_data(
    api_url: str, start_date=None, end_date=None
) -> Optional[List[Dict]]:
    """Fetch data from Boost API with pagination."""
    results, url, page = [], api_url, 1
    while url:
        data = _fetch_page(url, page)
        if not data:
            return None
        if "results" not in data:
            return [data]
        filtered, stop = _filter_by_date(data.get("results", []), start_date, end_date)
        results.extend(filtered)
        if stop:
            break
        url = data.get("next")
        if url:
            page += 1
    return results if results else None


def _format_email(item: Dict, source_url: str) -> Dict:
    """Format a single email item."""
    p = item.get("parent")
    t = item.get("thread")
    s = item.get("sender")
    return {
        "msg_id": item.get("message_id_hash"),
        "parent_id": p.split("/")[-2] if p else "",
        "thread_id": t.split("/")[-2] if t else "",
        "subject": item.get("subject"),
        "content": item.get("content"),
        "list_name": source_url.split("/")[-3],
        "sent_at": item.get("date"),
        "sender_address": s.get("address", "").replace(" (a) ", "@") if s else "",
        "sender_mailman_id": s.get("mailman_id", "") if s else "",
        "sender_name": item.get("sender_name"),
    }


BOOST_URLS = [
    "https://lists.boost.org/archives/api/list/boost-announce@lists.boost.org/emails/",
    "https://lists.boost.org/archives/api/list/boost-users@lists.boost.org/emails/",
    "https://lists.boost.org/archives/api/list/boost@lists.boost.org/emails/",
]


def _fetch_contents(url_list: List[Dict]) -> List[Dict]:
    """Fetch email content for each URL."""
    contents = []
    for item in url_list:
        url = item.get("url")
        if not url:
            continue
        data = fetch_api_data(url)
        if data:
            contents.extend(data)
        else:
            print(f"No data found for {url}")
    return contents


def _save_file(data: List[Dict], filename: str) -> bool:
    """Save formatted email data to JSON file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Successfully updated: {filename} ({len(data)} emails)")
        return True
    except (IOError, OSError) as e:
        print(f"Error writing {filename}: {e}")
        return False


def fetch_and_save_boost_emails(start_date: str, end_date: str) -> bool:
    """Fetch emails from Boost mailing list archives and save to JSON files."""
    base_dir = f"data/message_by_thread/json/{start_date}_{end_date}"
    os.makedirs(base_dir, exist_ok=True)

    all_results, success = [], 0
    for api_url in BOOST_URLS:
        url_list = fetch_api_data(api_url, start_date, end_date)
        if not url_list or not isinstance(url_list, list):
            print(f"No email data found for {api_url}")
            continue
        contents = _fetch_contents(url_list)
        formatted = [_format_email(item, api_url) for item in contents]
        all_results.extend(formatted)
        filename = f"{base_dir}/{api_url.split('/')[-3]}.json"
        if _save_file(formatted, filename):
            success += 1
        else:
            return False
    print(f"\nProcessed {len(all_results)} emails across {success} lists")
    return success > 0


def main():
    """Main execution function."""
    start_date = "2025-09-01"
    end_date = "2026-01-17"
    print("=" * 50)
    print(
        f"Boost Mailing List Email Fetcher(start_date={start_date}, end_date={end_date})"
    )
    print("=" * 50)

    success = fetch_and_save_boost_emails(start_date, end_date)

    print("\nUpdate completed successfully!" if success else "\nUpdate failed!")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
