"""Read and save lost-and-found JSON data."""
import json
from pathlib import Path


def load_items(filename):
    with open(filename, encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError("The database must contain an items list.")
    seen = set()
    for item in data["items"]:
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in ("id", "item", "color", "location", "date", "status")
        ):
            raise ValueError("Each item must contain all required text fields.")
        if item["id"] in seen:
            raise ValueError("Item IDs must be unique.")
        seen.add(item["id"])
    return data["items"]


def get_unclaimed_items(items):
    return [item for item in items if item["status"] == "unclaimed"]


def save_result(result, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)
        file.write("\n")
