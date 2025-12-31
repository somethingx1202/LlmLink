import argparse
import json
from collections import deque
from typing import Deque, Dict, List, Tuple

Entity = Tuple[str, str, int, int, str]
CategoryPayload = Dict[str, List[List[object]]]

CATEGORY_CHOICES = {
    "p": "PROPER_NOUN",
    "proper_noun": "PROPER_NOUN",
    "n": "NOUN_PHRASE",
    "noun_phrase": "NOUN_PHRASE",
    "r": "PRONOUNS",
    "pronouns": "PRONOUNS",
}


def read_brat_annotations(path: str) -> List[Entity]:
    annotations: List[Entity] = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(f"Unexpected annotation format: {line}")
            entity_id = parts[0]
            type_and_offsets = parts[1]
            entity_name = parts[2]

            tokens = type_and_offsets.split()
            if len(tokens) < 3:
                raise ValueError(f"Missing offsets for entity {entity_id}")

            entity_type = tokens[0]
            offsets_segment = " ".join(tokens[1:])
            first_segment = offsets_segment.split(";")[0].strip()
            offset_tokens = first_segment.split()
            if len(offset_tokens) != 2:
                raise ValueError(f"Invalid offset segment '{first_segment}' in {entity_id}")

            start, end = map(int, offset_tokens)
            annotations.append((entity_id, entity_type, start, end, entity_name))
    return annotations


def parse_occurrence_index(entity_id: str) -> int:
    digits = "".join(ch for ch in entity_id if ch.isdigit())
    if not digits:
        raise ValueError(f"No numeric index found in entity ID '{entity_id}'")
    return int(digits)


def prompt_category(entity: Entity) -> str:
    entity_id, entity_type, start, end, name = entity
    prompt = (
        f"Entity '{name}' (ID: {entity_id}, type: {entity_type}, span: {start}-{end}) "
        "[p] PROPER_NOUN, [n] NOUN_PHRASE, [r] PRONOUNS: "
    )
    while True:
        response = input(prompt).strip().lower()
        category = CATEGORY_CHOICES.get(response)
        if category:
            return category
        print("Please enter p, n, or r.")


def categorize_entities(entities: List[Entity]) -> CategoryPayload:
    queue: Deque[Entity] = deque(entities)
    categorized: CategoryPayload = {
        "PROPER_NOUN": [],
        "NOUN_PHRASE": [],
        "PRONOUNS": [],
    }
    while queue:
        entity = queue.popleft()
        category = prompt_category(entity)
        occurrence_index = parse_occurrence_index(entity[0])
        categorized[category].append([entity[4], occurrence_index])
    return categorized


def save_categories(categories: CategoryPayload, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(categories, file, ensure_ascii=False, indent=2)


def process_annotations(input_path: str, output_path: str) -> None:
    entities = read_brat_annotations(input_path)
    categorized = categorize_entities(entities)
    save_categories(categorized, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate BRAT entities into categories.")
    parser.add_argument("--input", required=True, help="Path to the BRAT .ann file.")
    parser.add_argument("--output", required=True, help="Destination JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_annotations(args.input, args.output)


if __name__ == "__main__":
    main()
    # Usage example:
    # python read_conll_ner.py --input ../litbank/entities/brat/11_alices_adventures_in_wonderland_brat.ann --output categorized_entities.json