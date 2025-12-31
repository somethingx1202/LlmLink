from __future__ import annotations

import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from obtain_entity_span_token_location_in_the_coref_brat_brat_DOT_ann import (
    EntityAnnotation,
    escape_for_output,
    parse_ann_file,
    resolve_token_indices,
    sort_entities_by_token_index,
    tokenize_by_space_with_offsets,
)


SpanTuple = Tuple[str, int, int, int, int, str]


def read_brat_coref_to_ordereddict(
    path: str,
) -> "OrderedDict[str, List[SpanTuple]]":
    """
    Read a BRAT .ann coreference file (and its paired .txt file) into an OrderedDict.

    Parameters
    ----------
    path:
        Path to the .ann file or its stem (with/without extension).

    Returns
    -------
    OrderedDict[str, List[SpanTuple]]
        Mapping from the first-seen coreference key (e.g., ``Alice-1``) to a list of
        mention tuples ``(text, token_start, token_end, char_start, char_end)``. Token
        indices are 0-based and ``token_end`` is exclusive.
    """
    ann_path = _resolve_ann_path(path)
    txt_path = ann_path.with_suffix(".txt")

    text = txt_path.read_text(encoding="utf-8")
    tokens = _tokenize(text)

    type_lookup = _build_span_type_lookup(ann_path)
    entity_order = OrderedDict()
    with ann_path.open(encoding="utf-8") as ann_file:
        for line in ann_file:
            line = line.strip()
            if not line or not line.startswith("T"):
                continue

            try:
                ann_id, raw_meta, mention_text = line.split("\t", maxsplit=2)
            except ValueError:
                continue  # malformed annotation line

            label, spans = _parse_label_and_spans(raw_meta)
            if not spans:
                continue

            # Only keep entries that carry an explicit coreference label.
            if "-" not in label:
                continue

            token_start, token_end = _aggregate_token_span(spans, tokens)
            char_start = spans[0][0]
            char_end = spans[-1][1]
            entity_type = (
                type_lookup.get((char_start, char_end, mention_text))
                or type_lookup.get((char_start, char_end, None))
                or "UNKNOWN"
            )
            mention_tuple: SpanTuple = (
                mention_text,
                token_start,
                token_end,
                char_start,
                char_end,
                entity_type,
            )

            if label not in entity_order:
                entity_order[label] = []

            if mention_tuple not in entity_order[label]:
                entity_order[label].append(mention_tuple)

    return entity_order


def _resolve_ann_path(path: str | os.PathLike[str]) -> Path:
    ann_path = Path(path)
    if ann_path.suffix != ".ann":
        ann_path = ann_path.with_suffix(".ann")
    if not ann_path.exists():
        raise FileNotFoundError(f"BRAT annotation file not found: {ann_path}")
    return ann_path


def _tokenize(text: str) -> List[Tuple[int, int]]:
    """
    Tokenize text on whitespace, returning a list of (char_start, char_end) spans.
    """
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def _parse_label_and_spans(raw_meta: str) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Parse the label and char spans from the BRAT metadata column.
    """
    parts = raw_meta.split(maxsplit=1)
    if not parts:
        return "", []
    label = parts[0]
    if len(parts) == 1:
        return label, []

    spans_str = parts[1]
    span_ranges: List[Tuple[int, int]] = []
    for segment in spans_str.split(";"):
        seg = segment.strip()
        if not seg:
            continue
        start_end = seg.split()
        if len(start_end) != 2:
            continue
        start, end = int(start_end[0]), int(start_end[1])
        span_ranges.append((start, end))

    span_ranges.sort(key=lambda item: item[0])
    return label, span_ranges


def _aggregate_token_span(
    spans: Sequence[Tuple[int, int]],
    tokens: Sequence[Tuple[int, int]],
) -> Tuple[int, int]:
    """
    Convert possibly multi-span char offsets into an aggregated token span.
    """
    token_starts: List[int] = []
    token_ends: List[int] = []

    for char_start, char_end in spans:
        start_idx, end_idx = _char_span_to_token_span(char_start, char_end, tokens)
        token_starts.append(start_idx)
        token_ends.append(end_idx)

    return min(token_starts), max(token_ends)


def _char_span_to_token_span(
    char_start: int,
    char_end: int,
    tokens: Sequence[Tuple[int, int]],
) -> Tuple[int, int]:
    """
    Find the token indices covering the given char span.
    """
    start_idx = None
    end_idx = None

    for idx, (tok_start, tok_end) in enumerate(tokens):
        if tok_end <= char_start:
            continue
        if start_idx is None:
            start_idx = idx
        if tok_start < char_end:
            end_idx = idx
        if tok_start >= char_end and end_idx is not None:
            break

    if start_idx is None or end_idx is None:
        raise ValueError(
            f"Unable to align character span [{char_start}, {char_end}) with tokens."
        )

    return start_idx, end_idx + 1


def _build_span_type_lookup(ann_path: Path) -> dict[Tuple[int, int, str | None], str]:
    lookup: dict[Tuple[int, int, str | None], str] = {}
    with ann_path.open(encoding="utf-8") as ann_file:
        for line in ann_file:
            line = line.strip()
            if not line or not line.startswith("T"):
                continue
            try:
                _, raw_meta, mention_text = line.split("\t", maxsplit=2)
            except ValueError:
                continue
            label, spans = _parse_label_and_spans(raw_meta)
            if not spans or "-" in label:
                continue
            char_start = spans[0][0]
            char_end = spans[-1][1]
            lookup[(char_start, char_end, mention_text)] = label
            lookup[(char_start, char_end, None)] = label
    return lookup


def _entity_type_priority(entity_type: str | None) -> int:
    if entity_type and entity_type.startswith("PROP_"):
        return 0
    if entity_type and entity_type.startswith("NOM_"):
        return 1
    return 2


def _select_canonical_mention(
    mentions: Sequence[SpanTuple],
) -> Tuple[str, str | None]:
    best_idx = 0
    best_priority = _entity_type_priority(mentions[0][5])
    for idx, mention in enumerate(mentions):
        priority = _entity_type_priority(mention[5])
        if priority < best_priority or (priority == best_priority and idx < best_idx):
            best_idx = idx
            best_priority = priority
    best_mention = mentions[best_idx]
    return best_mention[0], best_mention[5]


def _index_coref_spans(
    coref_clusters: "OrderedDict[str, List[SpanTuple]]",
) -> Tuple[
    dict[Tuple[int, int], str],
    dict[str, str],
    dict[str, int],
    dict[str, str],
]:
    span_to_label: dict[Tuple[int, int], str] = {}
    label_to_canonical: dict[str, str] = {}
    label_to_size: dict[str, int] = {}
    label_to_type: dict[str, str] = {}

    for label, mentions in coref_clusters.items():
        if not mentions:
            continue
        canonical_text, canonical_type = _select_canonical_mention(mentions)
        label_to_canonical[label] = canonical_text
        label_to_size[label] = len(mentions)
        label_to_type[label] = canonical_type if canonical_type is not None else mentions[0][5]
        for mention in mentions:
            span_to_label[(mention[3], mention[4])] = label

    return span_to_label, label_to_canonical, label_to_size, label_to_type


def build_coref_resolution_map(
    coref_clusters: "OrderedDict[str, List[SpanTuple]]",
    entities: Sequence[EntityAnnotation],
) -> "OrderedDict[Tuple[str, int], str | None]":
    (
        span_to_label,
        label_to_canonical,
        label_to_size,
        label_to_type,
    ) = _index_coref_spans(coref_clusters)
    resolution = OrderedDict()

    for entity in entities:
        if entity.token_index is None:
            continue
        label = span_to_label.get((entity.start, entity.end))
        if label is None:
            resolved = None
        else:
            cluster_size = label_to_size.get(label, 0)
            canonical_text = label_to_canonical.get(label)
            canonical_type = label_to_type.get(label)
            if canonical_text is None:
                resolved = None
            elif cluster_size <= 1 and canonical_type and canonical_type.startswith("PROP_"):
                resolved = canonical_text
            elif cluster_size <= 1:
                resolved = None
            else:
                resolved = canonical_text
        resolution[(entity.text, entity.token_index)] = resolved

    return resolution


def prepare_sorted_entities(txt_path: Path, ann_path: Path) -> List[EntityAnnotation]:
    text_content = txt_path.read_text(encoding="utf-8")
    tokens, starts = tokenize_by_space_with_offsets(text_content)
    entities = parse_ann_file(ann_path)
    resolve_token_indices(entities, tokens, starts)
    return sort_entities_by_token_index(entities)


def format_coref_resolution_map(
    resolution_map: "OrderedDict[Tuple[str, int], str | None]"
) -> str:
    lines = ["{"]
    for (text, token_idx), resolved in resolution_map.items():
        escaped_key = escape_for_output(text)
        value = "null" if resolved is None else f'"{escape_for_output(resolved)}"'
        lines.append(f'    ["{escaped_key}", {token_idx}]: {value},')
    if len(lines) > 1:
        lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)


def write_coref_resolution_json(txt_path: Path, ann_path: Path, output_path: Path) -> None:
    coref_clusters = read_brat_coref_to_ordereddict(ann_path)
    sorted_entities = prepare_sorted_entities(txt_path, ann_path)
    resolution_map = build_coref_resolution_map(coref_clusters, sorted_entities)
    payload = format_coref_resolution_map(resolution_map)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    ann_file_path = Path(
        # "../litbank/coref/brat/11_alices_adventures_in_wonderland_brat.ann"
        "../litbank/coref/brat/2489_moby_dick_brat.ann"
    ).resolve()
    txt_file_path = ann_file_path.with_suffix(".txt")

    coref_data = read_brat_coref_to_ordereddict(ann_file_path)
    sorted_entities = prepare_sorted_entities(txt_file_path, ann_file_path)
    mapping = build_coref_resolution_map(coref_data, sorted_entities)
    formatted_coref_resolution_map = format_coref_resolution_map(mapping)
    print(formatted_coref_resolution_map)
