from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def keep_first_occurrences(entries: Iterable[List]) -> List[List]:
    seen = set()
    unique_entries: List[List] = []
    for name, index in entries:
        if name in seen:
            continue
        seen.add(name)
        unique_entries.append([name, index])
    return unique_entries


def renumber_entries(deduped: Dict[str, List[List]]) -> Dict[str, List[List]]:
    ordered: List[tuple[int, int, str, str]] = []
    seq = 0
    for category, entries in deduped.items():
        for name, original_index in entries:
            ordered.append((original_index, seq, category, name))
            seq += 1

    ordered.sort(key=lambda item: (item[0], item[1]))

    index_map = {
        (category, name): new_index
        for new_index, (_, _, category, name) in enumerate(ordered)
    }

    return {
        category: [
            [name, index_map[(category, name)]]
            for name, _ in entries
        ]
        for category, entries in deduped.items()
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    # input_path = base_dir / "categorized_entities.json"
    # output_path = base_dir / "categorized_entities_duplicate_removed.json"
    input_path = base_dir / "categorized_entities_perfect_prediction.json"
    output_path = base_dir / "categorized_entities_perfect_prediction_duplicate_removed.json"

    with input_path.open(encoding="utf-8") as infile:
        categorized_entities = json.load(infile)

    deduped = {
        category: keep_first_occurrences(entries)
        for category, entries in categorized_entities.items()
    }

    renumbered = renumber_entries(deduped)

    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(renumbered, outfile, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()