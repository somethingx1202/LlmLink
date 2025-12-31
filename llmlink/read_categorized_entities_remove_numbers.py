from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Union


def remove_numbers_from_entities(json_path: Path, output_path: Path | None = None) -> None:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    updated = _clean_entity_lists(data)

    # target = output_path or json_path.with_name("categorized_entities_number_removed.json")
    target = output_path or json_path.with_name("categorized_entities_perfect_prediction_duplicate_removed_number_removed.json")
    with target.open("w", encoding="utf-8") as file:
        json.dump(updated, file, indent=2, ensure_ascii=False)


def _clean_entity_lists(data: dict) -> dict:
    for entity_type, entries in data.items():
        if isinstance(entries, Iterable):
            cleaned_entries: List[Union[list, str]] = []
            for entry in entries:
                if isinstance(entry, list):
                    filtered_entry = [item for item in entry if not isinstance(item, (int, float))]
                    cleaned_entries.append(filtered_entry)
                else:
                    cleaned_entries.append(entry)
            data[entity_type] = cleaned_entries
    return data


if __name__ == "__main__":
    # json_file = Path(__file__).with_name("categorized_entities_duplicate_removed.json")
    json_file = Path(__file__).with_name("categorized_entities_perfect_prediction_duplicate_removed.json")
    remove_numbers_from_entities(json_file)