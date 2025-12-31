# LlmLink — LitBank NER & Coreference Utilities

## Overview
This repository contains small utilities to:
- Call an LLM for LitBank-style NER extraction (`litbank_link_ner.py`, main entry).
- Read and transform LitBank BRAT/CoNLL annotations for NER and coreference.
- Post-process predicted entities (deduplicate, renumber, remove numeric suffixes).

Data folders (read-only):
- `litbank/`: LitBank texts and gold annotations (original, brat, conll, tsv).
- `narrativeqa/`: Scripts and CSVs for NarrativeQA (not primary focus here).

Code folder:
- `llmlink/`: Python sources, including `litbank_link_ner.py` and helpers.
- `llmlink/llm/openai_wrapper.py`: OpenAI client wrapper with caching.

## Requirements
- Python 3.10+ (recommended)
- Install dependencies (no requirements file provided; minimally):
  ```bash
  pip install openai tenacity python-dotenv numpy
  ```
  If you run into missing modules, install them as prompted (e.g., `pipmaster`).

## Environment
Create a `.env` (or set env vars) for API access:
```
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1   # optional, custom endpoint if any
LLMLINK_VERSION=dev                         # optional, for User-Agent
```
Responses are cached under `/mnt/nvme1n1p1/Data/openai_inference_cache`.

## Quickstart (main entry)
`litbank_link_ner.py` provides the NER pipeline and a self-test in `__main__`.
Run the sample (uses LitBank Moby Dick chunk selection, then NER, then evaluates):
```bash
cd /mnt/nvme1n1p1/Data/LlmLink/llmlink
python litbank_link_ner.py
```
What it does:
1) Splits a LitBank novel into chunks close to a target chapter length.
2) Calls the first available model from `client.models.list()` for NER.
3) Normalizes and positions entities in text.
4) Evaluates against a chosen BRAT gold `.ann` file and writes:
   - `temp.txt` (combined input chunk),
   - `res_NER_acc.txt` (prediction, mismatches, ground truth, accuracy).

To reuse the class:
```python
from litbank_link_ner import LitBankLinkNER
link = LitBankLinkNER(api_key="...", base_url="..., e.g., https://api.openai.com/v1")
result = link.NER_recognize("Your text here")
print(result)
```

## Key Scripts
- `litbank_link_ner.py`: Main NER workflow; chunking helpers; evaluation.
- `read_conll_ner.py`: CLI to categorize BRAT entities interactively into PROPER_NOUN / NOUN_PHRASE / PRONOUNS.
- `read_conll_coref.py`: Parse LitBank CoNLL coref into `OrderedDict` keyed by first mention.
- `read_conll_coref_brat.py`: Parse BRAT coref `.ann` + `.txt`; build resolution map; emit JSON mapping.
- `obtain_entity_span_token_location_in_the_coref_brat_brat_DOT_ann.py`: Extract entities with token indices; normalize labels; write `.ann` and JSON.
- `read_categorized_entities_remove_duplicates_numbers.py`: Deduplicate entity lists and renumber sequentially.
- `read_categorized_entities_remove_numbers.py`: Remove numeric items from categorized-entity JSON.
- `chapter_identifier_extract.py`: Dump the first 12 lines of each LitBank original text (chapter header helper).
- `individual_test_count_ith_word.py`: Simple word index demo.

## Data Layout (LitBank)
- `litbank/original/`: Raw novels.
- `litbank/entities/brat/`: BRAT NER annotations (`*.ann`, `*.txt`).
- `litbank/coref/brat/` and `litbank/coref/conll/`: Coreference annotations.
- `litbank/entities/tsv`, `litbank/coref/tsv`, `litbank/coref/conll`: Converted formats.

## Notes
- Model choice: `litbank_link_ner.py` picks the first model returned by `client.models.list()`; set your endpoint so a suitable chat model is first.
- Streaming: `openai_wrapper.openai_complete` supports streaming; caching is disabled for streamed calls.
- Character/token offsets: Coref/NER readers assume whitespace tokenization consistent with LitBank formatting; do not modify source `.txt` files.
- File paths in scripts are relative to repo root unless overridden by CLI args.

## Troubleshooting
- Missing modules: install as indicated (e.g., `pip install pipmaster`).
- Invalid JSON from model: ensure the model prompt is followed; adjust `temperature/top_p` in `litbank_link_ner.py` if outputs are unstable.
- Span alignment errors: verify you are using the matching `.txt` and `.ann` pair from the same basename.