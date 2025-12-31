from collections import OrderedDict
from typing import Dict, List, Tuple


def read_conll_coref_to_ordereddict(
    path: str,
) -> "OrderedDict[str, List[Tuple[str, int, int, int, int]]]":
    """
    Parse a LitBank-style .conll coreference file into an OrderedDict.

    Returns:
      OrderedDict where:
        - key: str, the first-appearing mention text for a coreference cluster.
               If that key mention contains nested single-token mentions of other
               clusters (e.g., "her sister" where "her" is a separate cluster),
               those tokens are replaced with the other clusters' first-mention
               keys (e.g., "Alice sister").
        - value: List of tuples (text_str, token_start_pos, token_end_pos, text_start_pos, text_end_pos)
          where:
            - token_start_pos: token index within the sentence (inclusive)
            - token_end_pos: token index within the sentence (exclusive)
            - text_start_pos: character offset in the sentence string (inclusive)
            - text_end_pos: character offset in the sentence string (exclusive)

    Notes:
      - Sentences are reconstructed by joining tokens with single spaces (including punctuation),
        so character offsets are calculated for that representation (e.g., "I like the dog , it ...").
      - Coref annotations are expected in the last column; multiple annotations per token are '|' delimited.
      - Mentions do not cross sentence boundaries in LitBank; token indices are sentence-local.
      - The order of items in the OrderedDict follows the first occurrence of each cluster in the document.
      - Only the cluster key string is normalized with replacements; mention tuples retain original text and offsets.
    """
    # Per-cluster storage
    clusters: Dict[str, Dict] = {}  # cluster_id -> { 'order': int, 'key': str, 'mentions': list }
    cluster_order: List[str] = []   # first-seen order of cluster ids

    # Current sentence buffers
    sent_tokens: List[str] = []
    sent_char_offsets: List[int] = []  # char start per token in the reconstructed sentence
    cur_char: int = 0                   # running char position in reconstructed sentence
    open_spans: Dict[str, int] = {}     # cluster_id -> token_start_idx (within current sentence)

    # Track single-token mentions by token index within the current sentence
    # token_idx -> List[cluster_id]
    token_single_mentions: Dict[int, List[str]] = {}

    first_seen_counter = 0

    def flush_sentence():
        # On sentence boundary: ensure no dangling opens; clear sentence buffers
        nonlocal sent_tokens, sent_char_offsets, cur_char, open_spans, token_single_mentions
        open_spans.clear()
        sent_tokens = []
        sent_char_offsets = []
        cur_char = 0
        token_single_mentions = {}

    def add_mention(cluster_id: str, start_tok: int, end_tok_inclusive: int):
        """Create and store a mention for a cluster from current sentence buffers."""
        nonlocal first_seen_counter

        # Ensure cluster book-keeping exists
        if cluster_id not in clusters:
            # First mention for this cluster determines key text and order
            # Build key text, but replace nested single-token mentions with their cluster keys (if known)
            tok_slice = sent_tokens[start_tok : end_tok_inclusive + 1]
            replaced_tokens = list(tok_slice)
            for local_i, tok_idx in enumerate(range(start_tok, end_tok_inclusive + 1)):
                inner_ids = token_single_mentions.get(tok_idx, [])
                # If multiple cluster ids mark the same token, prefer the earliest-seen one.
                inner_ids_sorted = sorted(
                    (cid for cid in inner_ids if cid in clusters and cid != cluster_id),
                    key=lambda cid: clusters[cid]["order"]
                )
                if inner_ids_sorted:
                    replaced_tokens[local_i] = clusters[inner_ids_sorted[0]]["key"]

            mention_text_for_key = " ".join(replaced_tokens)

            clusters[cluster_id] = {
                "order": first_seen_counter,
                "key": mention_text_for_key,
                "mentions": [],
            }
            cluster_order.append(cluster_id)
            first_seen_counter += 1

        # Compute spans (token_end exclusive; char_end exclusive)
        token_start = start_tok
        token_end_excl = end_tok_inclusive + 1
        char_start = sent_char_offsets[start_tok]
        char_end_excl = sent_char_offsets[end_tok_inclusive] + len(sent_tokens[end_tok_inclusive])
        text = " ".join(sent_tokens[token_start:token_end_excl])

        clusters[cluster_id]["mentions"].append(
            (text, token_start, token_end_excl, char_start, char_end_excl)
        )

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # Document markers
            if not line or line.startswith("#end"):
                # Sentence boundary or end of document
                flush_sentence()
                if line.startswith("#end"):
                    # Finished the document
                    break
                continue
            if line.startswith("#begin"):
                # New document marker; reset sentence buffers
                flush_sentence()
                continue

            # Parse token row
            parts = line.split("\t")
            if len(parts) < 4:
                # Unexpected format; skip
                continue

            token = parts[3]
            coref_cell = parts[-1].strip() if parts else ""

            # Append token and record its char start
            sent_tokens.append(token)
            sent_char_offsets.append(cur_char)
            # Advance char position for next token (space-separated sentence reconstruction)
            cur_char += len(token) + 1

            tok_idx = len(sent_tokens) - 1

            if not coref_cell or coref_cell == "_":
                continue

            # Process one or more annotations on this token
            for piece in coref_cell.split("|"):
                piece = piece.strip()
                if not piece or piece == "_":
                    continue

                # Single-token mention: "(ID)"
                if piece.startswith("(") and piece.endswith(")"):
                    cid = piece[1:-1]
                    if cid:
                        # Record single-token mention for replacement normalization of future keys
                        token_single_mentions.setdefault(tok_idx, []).append(cid)
                        add_mention(cid, tok_idx, tok_idx)
                    continue

                # Span opener: "(ID"
                if piece.startswith("(") and not piece.endswith(")"):
                    cid = piece[1:]
                    if cid:
                        open_spans[cid] = tok_idx
                    continue

                # Span closer: "ID)"
                if piece.endswith(")") and not piece.startswith("("):
                    cid = piece[:-1]
                    if cid:
                        start_tok = open_spans.pop(cid, tok_idx)  # fall back to current token if unmatched
                        end_tok = tok_idx
                        add_mention(cid, start_tok, end_tok)
                    continue

                # Anything else is unexpected; ignore silently

    # Build the requested OrderedDict using first-mention text as the key
    result: "OrderedDict[str, List[Tuple[str, int, int, int, int]]]" = OrderedDict()
    for cid in sorted(cluster_order, key=lambda x: clusters[x]["order"]):
        key_text = clusters[cid]["key"]
        # If duplicate keys occur across clusters, later ones will overwrite.
        # If that is undesirable, disambiguate here.
        result[key_text] = clusters[cid]["mentions"]

    return result

# Example usage:
coref_data = read_conll_coref_to_ordereddict("/mnt/nvme1n1p1/Data/LlmLink/litbank/coref/conll/11_alices_adventures_in_wonderland_brat.conll")
for key, mentions in coref_data.items():
    print(f"Cluster Key: {key}")
    for mention in mentions:
        print(f"  Mention: {mention}")