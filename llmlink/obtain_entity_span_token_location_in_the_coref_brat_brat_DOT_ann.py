from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

ENTITY_LABEL_MAPPING = OrderedDict(
    [
        ("NOM_PER", "NOUN_PHRASE"),
        ("NOM_FAC", "NOUN_PHRASE"),
        ("NOM_LOC", "NOUN_PHRASE"),
        ("PROP_PER", "PROPER_NOUN"),
        ("PROP_GPE", "PROPER_NOUN"),
        ("PRON_PER", "PRONOUNS"),
        ("PRON_FAC", "PRONOUNS"),
    ]
)

VALID_ENTITY_PREFIXES = tuple(ENTITY_LABEL_MAPPING.keys())


@dataclass
class EntityAnnotation:
    tid: str
    label: str
    start: int
    end: int
    text: str
    token_index: int | None = None
    normalized_label: str | None = None

    def to_entity_line(self) -> str:
        return f"{self.tid}\t{self.label}\t{self.token_index}\t{self.text}"

    def to_ordered_mapping(self) -> str:
        if self.token_index is None:
            raise ValueError(f"Missing token index for entity {self.tid}")
        if self.normalized_label is None:
            raise ValueError(f"Missing normalized label for entity {self.tid}")
        escaped_text = escape_for_output(self.text)
        label_value = ENTITY_LABEL_MAPPING[self.normalized_label]
        return f'{{{{["{escaped_text}", {self.token_index}]: "{label_value}"}}}}'


def normalize_entity_label(label: str) -> str | None:
    for prefix in VALID_ENTITY_PREFIXES:
        if label.startswith(prefix):
            return prefix
    return None


def extract_first_span(offsets_text: str) -> Tuple[int, int] | None:
    first_span = offsets_text.split(";", 1)[0].strip()
    parts = first_span.split()
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def is_supported_entity_label(label: str) -> bool:
    return label.startswith(VALID_ENTITY_PREFIXES)


def parse_ann_file(ann_path: Path) -> List[EntityAnnotation]:
    entities: List[EntityAnnotation] = []
    with ann_path.open("r", encoding="utf-8") as ann_file:
        for raw_line in ann_file:
            line = raw_line.strip()
            if not line or not line.startswith("T"):
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            span_meta = parts[1]
            meta_tokens = span_meta.split()
            if len(meta_tokens) < 3:
                continue

            label = meta_tokens[0]
            if not is_supported_entity_label(label):
                continue

            normalized_label = normalize_entity_label(label)
            if normalized_label is None:
                continue

            offsets_text = span_meta[len(label):].strip()
            span = extract_first_span(offsets_text)
            if span is None:
                continue
            start, end = span

            text = parts[2]
            entities.append(
                EntityAnnotation(
                    tid=parts[0],
                    label=label,
                    start=start,
                    end=end,
                    text=text,
                    normalized_label=normalized_label,
                )
            )
    return entities


def tokenize_by_space_with_offsets(text: str) -> Tuple[List[str], List[int]]:
    tokens: List[str] = []
    starts: List[int] = []

    for match in re.finditer(r"\S+", text):
        tokens.append(match.group())
        starts.append(match.start())

    return tokens, starts


def resolve_token_indices(
    entities: Iterable[EntityAnnotation],
    tokens: List[str],
    starts: List[int],
) -> None:
    for entity in entities:
        token_idx = locate_token_index(entity.start, tokens, starts)
        entity.token_index = token_idx


def locate_token_index(start_offset: int, tokens: List[str], starts: List[int]) -> int:
    for idx, (token, token_start) in enumerate(zip(tokens, starts)):
        if not token:
            continue
        token_end = token_start + len(token)
        if token_start <= start_offset < token_end:
            return idx
    raise ValueError(f"Unable to locate token index for offset {start_offset}")


def write_entity_output(entities: Iterable[EntityAnnotation], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for entity in entities:
            output_file.write(f"{entity.to_entity_line()}\n")


def escape_for_output(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def write_json_output(entities: Iterable[EntityAnnotation], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_entries = [entity.to_ordered_mapping() for entity in entities]
    payload = "[" + ", ".join(ordered_entries) + "]"
    with output_path.open("w", encoding="utf-8") as json_file:
        json_file.write(payload)


def sort_entities_by_token_index(entities: List[EntityAnnotation]) -> List[EntityAnnotation]:
    return sorted(
        entities,
        key=lambda entity: (
            entity.token_index if entity.token_index is not None else -1,
            entity.start,
            entity.tid,
        ),
    )


def default_input_paths() -> Tuple[Path, Path]:
    project_root = Path(__file__).resolve().parent.parent
    base_name = "11_alices_adventures_in_wonderland_brat"
    brat_dir = project_root / "litbank" / "coref" / "brat"
    txt_path = brat_dir / f"{base_name}.txt"
    ann_path = brat_dir / f"{base_name}.ann"
    return txt_path, ann_path


def derive_default_outputs(txt_path: Path) -> Tuple[Path, Path]:
    base_name = txt_path.stem
    output_dir = Path.cwd()
    entity_output = output_dir / f"{base_name}_coref_entity_only.ann"
    json_output = output_dir / f"{base_name}_coref_json_entity_only.ann"
    return entity_output, json_output


def main() -> None:
    default_txt, default_ann = default_input_paths()

    parser = argparse.ArgumentParser(
        description="Extract coreference entity annotations with token indices."
    )
    parser.add_argument("--txt", type=Path, default=default_txt, help="Path to the .txt file.")
    parser.add_argument("--ann", type=Path, default=default_ann, help="Path to the .ann file.")
    parser.add_argument(
        "--entity-output",
        type=Path,
        default=None,
        help="Output path for the entity-only .ann file.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Output path for the JSON entity file.",
    )

    args = parser.parse_args()

    txt_path: Path = args.txt
    ann_path: Path = args.ann

    if not txt_path.exists():
        raise FileNotFoundError(f"Text file not found: {txt_path}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    entity_output, json_output = derive_default_outputs(txt_path)
    if args.entity_output is not None:
        entity_output = args.entity_output
    if args.json_output is not None:
        json_output = args.json_output

    text_content = txt_path.read_text(encoding="utf-8")
    tokens, token_starts = tokenize_by_space_with_offsets(text_content)

    entities = parse_ann_file(ann_path)
    resolve_token_indices(entities, tokens, token_starts)

    sorted_entities = sort_entities_by_token_index(entities)

    write_entity_output(sorted_entities, entity_output)
    write_json_output(sorted_entities, json_output)


if __name__ == "__main__":
    main()