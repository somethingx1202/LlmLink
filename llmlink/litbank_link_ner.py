from llm.openai_wrapper import create_openai_client, openai_complete
import logging
from collections.abc import Iterable
import json
from pathlib import Path
import re
import time
from typing import Any


class LitBankLinkNER:
    DEFAULT_CHUNK_SIZE = 16384

    def __init__(self, api_key: str, base_url: str):
        self.client = create_openai_client(
            api_key=api_key,
            base_url=base_url,
        )

    def check_available_models(self) -> list[str]:
        models = self.client.models.list()
        return [model.id for model in models.data]

    def get_response(self, prompt: str, model_name: str, max_tokens: int | None = None) -> str | Iterable[str]:
        request_params: dict[str, Any] = {}
        
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        request_params["temperature"] = 0.6
        # request_params["temperature"] = 0.8
        # Those are set in the llamacpp
        request_params["top_p"] = 0.95
        # request_params["top_p"] = 1.2
        # request_params["frequency_penalty"] = 0.4
        request_params["frequency_penalty"] = 0.1
        # request_params["repeat_penalty"] = 1.1
        request_params["presence_penalty"] = 0.001
        # request_params["min_p"] = 0.0
        # request_params["top_k"] = 20
        # request_params["num_ctx"] = 32768

        if max_tokens is None:
            return openai_complete(
                prompt,
                client=self.client,
                model_name=model_name,
                is_stream=True,
                request_params=request_params,
            )

        repeat_count = 0
        while True:
            response = openai_complete(
                prompt,
                client=self.client,
                model_name=model_name,
                is_stream=True,
                request_params=request_params,
            )
            text_response, chunk_count = self._response_to_text(response)
            if self._estimate_token_count(text_response, chunk_count) < max_tokens:
                return text_response

            repeat_count += 1
            if repeat_count > 5:
                raise RuntimeError("Model response exceeded max_tokens after 5 retries.")
            time.sleep(16)  # wait before retrying to avoid rapid repeated requests

    @staticmethod
    def _response_to_text(response: str | Iterable[str]) -> tuple[str, int | None]:
        if response is None:
            return "", 0
        if isinstance(response, bytes):
            return response.decode("utf-8", errors="replace"), None
        if isinstance(response, str):
            return response, None
        if isinstance(response, Iterable):
            print("--- Streamed Response Start ---")
            parts: list[str] = []
            for chunk in response:
                if isinstance(chunk, bytes):
                    decoded = chunk.decode("utf-8", errors="replace")
                    parts.append(decoded)
                    print(decoded, end="", flush=True)
                else:
                    text_chunk = str(chunk)
                    parts.append(text_chunk)
                    print(text_chunk, end="", flush=True)
            print()  # newline after stream
            print("--- Streamed Response End ---")
            return "".join(parts), len(parts)
        return str(response), None

    @staticmethod
    def _estimate_token_count(text: str, chunk_count: int | None = None) -> int:
        if chunk_count is not None:
            return chunk_count
        if not text:
            return 0
        return len(re.findall(r"\S+", text))

    @staticmethod
    def _split_into_chunks(text: str, default_chunk_size: int | None = None) -> list[str]:
        if not text:
            return []
        if default_chunk_size is None:
            default_chunk_size = LitBankLinkNER.DEFAULT_CHUNK_SIZE

        lines = text.splitlines()
        divider_indices: list[int] = [
            idx for idx, line in enumerate(lines) if LitBankLinkNER._is_divider_line(line)
        ]

        if divider_indices:
            starts: list[int] = []
            first_divider = divider_indices[0]
            if any(line.strip() for line in lines[:first_divider]):
                starts.append(0)
            starts.extend(divider_indices)
            starts = sorted(set(starts))
            starts.append(len(lines))

            chunks: list[str] = []
            i = 0
            while i < len(starts) - 1:
                start = starts[i]
                j = i + 1
                while j < len(starts):
                    segment_lines = lines[start:starts[j]]
                    if LitBankLinkNER._has_body_content(segment_lines):
                        break
                    j += 1
                end = starts[j] if j < len(starts) else len(lines)
                chunk_lines = lines[start:end]
                chunk_text = "\n".join(chunk_lines).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                i = j if j > i else i + 1

            if chunks:
                return chunks

        fallback = text[:default_chunk_size].strip()
        return [fallback] if fallback else []

    @staticmethod
    def extract_first_chunk(text: str, default_chunk_size: int | None = None) -> str:
        chunks = LitBankLinkNER._split_into_chunks(text, default_chunk_size)
        return chunks[0] if chunks else ""

    @staticmethod
    def extract_chunks_from(text: str, chunk_index: int, default_chunk_size: int | None = None) -> list[str]:
        if chunk_index < 0:
            chunk_index = 0
        chunks = LitBankLinkNER._split_into_chunks(text, default_chunk_size)
        if chunk_index >= len(chunks):
            return []
        return chunks[chunk_index:]

    @staticmethod
    def _has_body_content(lines: list[str]) -> bool:
        for line in lines:
            if line.strip() and not LitBankLinkNER._is_divider_line(line):
                return True
        return False

    @staticmethod
    def _is_divider_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        if re.match(
            r"^(chapter|book|part|volume|letter|adventure|section|phase|prologue|epilogue|preface)\b",
            stripped,
            flags=re.IGNORECASE,
        ):
            return True

        if re.match(r"^(?:[IVXLCDM]+|[0-9]+)(?:\s*[-–]\s*(?:[IVXLCDM]+|[0-9]+))?(?:[.:])?$", stripped):
            return True

        if (
            len(stripped) <= 80
            and any(char.isalpha() for char in stripped)
            and re.fullmatch(r"[A-Z0-9][A-Z0-9\s',\-.:;_/()]*", stripped)
        ):
            return True

        return False

    NER_SYSTEM_PROMPT = "You are an expert NER assistant."
    NER_USER_PROMPT_TEMPLATE = '''
# INSTRUCTIONS
Given a chapter, you have to identify ONLY the following: 
PROPER_NOUN - The full or partial name of an individual person, place, or organization, e.g. Tom Sawyer, London, and Oxfam. 
NOUN_PHRASE - A grouping of words that includes a noun, and functions within a sentence as a subject, object, or another role, e.g., the boy, the city, and the organization. This ONLY includes those that refer to persons, places or organizations. 
PRONOUNS - A word that is used instead of a noun or noun phrase. This includes common forms (I, me, my, myself, you, your, yourself, she, her, herself, he, him, his, himself, it, its, we, our, they, them, their), historical forms (thou, thee, thine, ye) and forms originating in transcriptions of speech ('em, 'ee, yeh, yer). 
This ONLY includes those that refer to persons, places or organizations. You will output in JSON format, without any other text. If an entity type is not present, output an empty list for it. Like this: {{\"PROPER_NOUN\": [], \"NOUN_PHRASE\": [], \"PRONOUNS\": []}}. Each entity should only appear once in the list, even if it occurs multiple times in the text.
# EXAMPLES 
## EXAMPLE 1: 
### EXAMPLE INPUT DATA: Dear Livesey--As I do not know whether you are at the hall or still in London, I send this in double to both places. 
### EXAMPLE OUTPUT DATA: {{"PROPER_NOUN": ["Livesey", "London"], "NOUN_PHRASE": ["the hall", "both places"], "PRONOUNS": ["I", "you", "this"]}}
## EXAMPLE 2:
### EXAMPLE INPUT DATA: CHAPTER I. Down the Rabbit-Hole

Alice was beginning to get very tired of sitting by her sister on the
bank, and of having nothing to do: once or twice she had peeped into thebook her sister was reading, but it had no pictures or conversations in
it, ‘and what is the use of a book,’ thought Alice ‘without pictures or
conversations?’

So she was considering in her own mind (as well as she could, for the
hot day made her feel very sleepy and stupid), whether the pleasure
of making a daisy-chain would be worth the trouble of getting up and
picking the daisies, when suddenly a White Rabbit with pink eyes ran
close by her.

There was nothing so VERY remarkable in that; nor did Alice think it so
VERY much out of the way to hear the Rabbit say to itself, ‘Oh dear!
Oh dear! I shall be late!’ (when she thought it over afterwards, it
occurred to her that she ought to have wondered at this, but at the time
it all seemed quite natural); but when the Rabbit actually TOOK A WATCH
OUT OF ITS WAISTCOAT-POCKET, and looked at it, and then hurried on,
Alice started to her feet, for it flashed across her mind that she had
never before seen a rabbit with either a waistcoat-pocket, or a watch
to take out of it, and burning with curiosity, she ran across the field
after it, and fortunately was just in time to see it pop down a large
rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how
in the world she was to get out again.

The rabbit-hole went straight on like a tunnel for some way, and then
dipped suddenly down, so suddenly that Alice had not a moment to think
about stopping herself before she found herself falling down a very deep
well.

Either the well was very deep, or she fell very slowly, for she had
plenty of time as she went down to look about her and to wonder what was
going to happen next. First, she tried to look down and make out what
she was coming to, but it was too dark to see anything; then she
looked at the sides of the well, and noticed that they were filled with
cupboards and book-shelves; here and there she saw maps and pictures
hung upon pegs. She took down a jar from one of the shelves as
she passed; it was labelled ‘ORANGE MARMALADE’, but to her great
disappointment it was empty: she did not like to drop the jar for fear
of killing somebody, so managed to put it into one of the cupboards as
she fell past it.

‘Well!’ thought Alice to herself, ‘after such a fall as this, I shall
think nothing of tumbling down stairs! How brave they’ll all think me at
home! Why, I wouldn’t say anything about it, even if I fell off the top
of the house!’ (Which was very likely true.)

Down, down, down. Would the fall NEVER come to an end! ‘I wonder how
many miles I’ve fallen by this time?’ she said aloud. ‘I must be getting
somewhere near the centre of the earth. Let me see: that would be four
thousand miles down, I think--’ (for, you see, Alice had learnt several
things of this sort in her lessons in the schoolroom, and though this
was not a VERY good opportunity for showing off her knowledge, as there
was no one to listen to her, still it was good practice to say it over)
‘--yes, that’s about the right distance--but then I wonder what Latitude
or Longitude I’ve got to?’ (Alice had no idea what Latitude was, or
Longitude either, but thought they were nice grand words to say.)

Presently she began again. ‘I wonder if I shall fall right THROUGH the
earth! How funny it’ll seem to come out among the people that walk with
their heads downward! The Antipathies, I think--’ (she was rather glad
there WAS no one listening, this time, as it didn’t sound at all the
right word) ‘--but I shall have to ask them what the name of the country
is, you know. Please, Ma’am, is this New Zealand or Australia?’ (and
she tried to curtsey as she spoke--fancy CURTSEYING as you’re falling
through the air! Do you think you could manage it?) ‘And what an
ignorant little girl she’ll think me for asking! No, it’ll never do to
ask: perhaps I shall see it written up somewhere.’

Down, down, down. There was nothing else to do, so Alice soon began
talking again. ‘Dinah’ll miss me very much to-night, I should think!’
(Dinah was the cat.) ‘I hope they’ll remember her saucer of milk at
tea-time. Dinah my dear! I wish you were down here with me! There are no
mice in the air, I’m afraid, but you might catch a bat, and that’s very
like a mouse, you know. But do cats eat bats, I wonder?’ And here Alice
began to get rather sleepy, and went on saying to herself, in a dreamy
sort of way, ‘Do cats eat bats? Do cats eat bats?’ and sometimes, ‘Do
bats eat cats?’ for, you see, as she couldn’t answer either question,
it didn’t much matter which way she put it. She felt that she was dozing
off, and had just begun to dream that she was walking hand in hand with
Dinah, and saying to her very earnestly, ‘Now, Dinah, tell me the truth:
did you ever eat a bat?’ when suddenly, thump! thump! down she came upon
a heap of sticks and dry leaves, and the fall was over.

Alice was not a bit hurt, and she jumped up on to her feet in a moment:
she looked up, but it was all dark overhead; before her was another
long passage, and the White Rabbit was still in sight, hurrying down it.
There was not a moment to be lost: away went Alice like the wind, and
was just in time to hear it say, as it turned a corner, ‘Oh my ears
and whiskers, how late it’s getting!’ She was close behind it when she
turned the corner, but the Rabbit was no longer to be seen: she found
herself in a long, low hall, which was lit up by a row of lamps hanging
from the roof.

There were doors all round the hall, but they were all locked; and when
Alice had been all the way down one side and up the other, trying every
door, she walked sadly down the middle, wondering how she was ever to
get out again.

Suddenly she came upon a little three-legged table, all made of solid
glass; there was nothing on it except a tiny golden key, and Alice’s
first thought was that it might belong to one of the doors of the hall;
but, alas! either the locks were too large, or the key was too small,
but at any rate it would not open any of them. However, on the second
time round, she came upon a low curtain she had not noticed before, and
behind it was a little door about fifteen inches high: she tried the
little golden key in the lock, and to her great delight it fitted!

Alice opened the door and found that it led into a small passage, not
much larger than a rat-hole: she knelt down and looked along the passage
into the loveliest garden you ever saw. How she longed to get out of
that dark hall, and wander about among those beds of bright flowers and
those cool fountains, but she could not even get her head through the
doorway; ‘and even if my head would go through,’ thought poor Alice, ‘it
would be of very little use without my shoulders. Oh, how I wish I could
shut up like a telescope! I think I could, if I only knew how to begin.’
For, you see, so many out-of-the-way things had happened lately,
that Alice had begun to think that very few things indeed were really
impossible.

There seemed to be no use in waiting by the little door, so she went
back to the table, half hoping she might find another key on it, or at
any rate a book of rules for shutting people up like telescopes: this
time she found a little bottle on it, [‘which certainly was not here
before,’ said Alice,) and round the neck of the bottle was a paper
label, with the words ‘DRINK ME’ beautifully printed on it in large
letters.

It was all very well to say ‘Drink me,’ but the wise little Alice was
not going to do THAT in a hurry. ‘No, I’ll look first,’ she said, ‘and
see whether it’s marked “poison” or not’; for she had read several nice
little histories about children who had got burnt, and eaten up by wild
beasts and other unpleasant things, all because they WOULD not remember
the simple rules their friends had taught them: such as, that a red-hot
poker will burn you if you hold it too long; and that if you cut your
finger VERY deeply with a knife, it usually bleeds; and she had never
forgotten that, if you drink much from a bottle marked ‘poison,’ it is
almost certain to disagree with you, sooner or later.

However, this bottle was NOT marked ‘poison,’ so Alice ventured to taste
it, and finding it very nice, (it had, in fact, a sort of mixed flavour
of cherry-tart, custard, pine-apple, roast turkey, toffee, and hot
buttered toast,) she very soon finished it off.

  *    *    *    *    *    *    *

    *    *    *    *    *    *

  *    *    *    *    *    *    *

‘What a curious feeling!’ said Alice; ‘I must be shutting up like a
telescope.’

And so it was indeed: she was now only ten inches high, and her face
brightened up at the thought that she was now the right size for going
through the little door into that lovely garden. First, however, she
waited for a few minutes to see if she was going to shrink any further:
she felt a little nervous about this; ‘for it might end, you know,’ said
Alice to herself, ‘in my going out altogether, like a candle. I wonder
what I should be like then?’ And she tried to fancy what the flame of a
candle is like after the candle is blown out, for she could not remember
ever having seen such a thing.

After a while, finding that nothing more happened, she decided on going
into the garden at once; but, alas for poor Alice! when she got to the
door, she found she had forgotten the little golden key, and when she
went back to the table for it, she found she could not possibly reach
it: she could see it quite plainly through the glass, and she tried her
best to climb up one of the legs of the table, but it was too slippery;
and when she had tired herself out with trying, the poor little thing
sat down and cried.

‘Come, there’s no use in crying like that!’ said Alice to herself,
rather sharply; ‘I advise you to leave off this minute!’ She generally
gave herself very good advice, (though she very seldom followed it),
and sometimes she scolded herself so severely as to bring tears into
her eyes; and once she remembered trying to box her own ears for having
cheated herself in a game of croquet she was playing against herself,
for this curious child was very fond of pretending to be two people.
‘But it’s no use now,’ thought poor Alice, ‘to pretend to be two people!
Why, there’s hardly enough of me left to make ONE respectable person!’

Soon her eye fell on a little glass box that was lying under the table:
she opened it, and found in it a very small cake, on which the words
‘EAT ME’ were beautifully marked in currants. ‘Well, I’ll eat it,’ said
Alice, ‘and if it makes me grow larger, I can reach the key; and if it
makes me grow smaller, I can creep under the door; so either way I’ll
get into the garden, and I don’t care which happens!’

She ate a little bit, and said anxiously to herself, ‘Which way? Which
way?’, holding her hand on the top of her head to feel which way it was
growing, and she was quite surprised to find that she remained the same
size: to be sure, this generally happens when one eats cake, but Alice
had got so much into the way of expecting nothing but out-of-the-way
things to happen, that it seemed quite dull and stupid for life to go on
in the common way.

So she set to work, and very soon finished off the cake.

  *    *    *    *    *    *    *

    *    *    *    *    *    *

  *    *    *    *    *    *    *
### EXAMPLE OUTPUT DATA: {{
  "PROPER_NOUN": [
    [
      "Alice"
    ],
    [
      "the White Rabbit"
    ],
    [
      "Dinah"
    ],
    [
      "New Zealand"
    ],
    [
      "Australia"
    ],
    [
      "the house"
    ],
    [
      "the schoolroom"
    ],
    [
      "the earth"
    ],
    [
      "the centre of the earth"
    ],
    [
      "the Antipathies"
    ],
    [
      "the hall"
    ],
    [
      "the garden"
    ],
    [
      "the candle"
    ],
    [
      "the top of the house"
    ],
    [
      "the world"
    ],
    [
      "the country"
    ],
    [
      "Ma’am"
    ],
    [
      "the rabbit-hole"
    ],
    [
      "the well"
    ],
    [
      "the field"
    ],
    [
      "the hedge"
    ],
    [
      "the passage"
    ],
    [
      "the lamp"
    ],
    [
      "the door"
    ],
    [
      "the key"
    ],
    [
      "the bottle"
    ],
    [
      "the cake"
    ],
    [
      "the table"
    ]
  ],
  "NOUN_PHRASE": [
    [
      "her sister"
    ],
    [
      "the bank"
    ],
    [
      "the book"
    ],
    [
      "the daisy-chain"
    ],
    [
      "the daisies"
    ],
    [
      "the field"
    ],
    [
      "a large rabbit-hole under the hedge"
    ],
    [
      "a tunnel"
    ],
    [
      "a very deep well"
    ],
    [
      "the well"
    ],
    [
      "the sides of the well"
    ],
    [
      "cupboards and book-shelves"
    ],
    [
      "maps and pictures hung upon pegs"
    ],
    [
      "a jar"
    ],
    [
      "ORANGE MARMALADE"
    ],
    [
      "the house"
    ],
    [
      "the top of the house"
    ],
    [
      "the people that walk with their heads downward"
    ],
    [
      "the country"
    ],
    [
      "the hall"
    ],
    [
      "a long, low hall"
    ],
    [
      "a small passage"
    ],
    [
      "a rat-hole"
    ],
    [
      "a little door"
    ],
    [
      "the passage into the garden"
    ],
    [
      "the loveliest garden you ever saw"
    ],
    [
      "beds of bright flowers"
    ],
    [
      "cool fountains"
    ],
    [
      "the doorway"
    ],
    [
      "the dark hall"
    ],
    [
      "the little golden key"
    ],
    [
      "the bottle marked 'poison'"
    ],
    [
      "the cake marked 'EAT ME'"
    ],
    [
      "the glass box"
    ],
    [
      "the top of her head"
    ],
    [
      "the corner"
    ],
    [
      "a row of lamps hanging from the roof"
    ],
    [
      "the world"
    ],
    [
      "the candle"
    ],
    [
      "the flame of a candle"
    ],
    [
      "the rules their friends had taught them"
    ],
    [
      "a red-hot poker"
    ],
    [
      "a knife"
    ],
    [
      "a bottle marked 'poison'"
    ],
    [
      "the game of croquet"
    ],
    [
      "a child pretending to be two people"
    ]
  ],
  "PRONOUNS": [
    [
      "she"
    ],
    [
      "her"
    ],
    [
      "it"
    ],
    [
      "him"
    ],
    [
      "them"
    ],
    [
      "they"
    ],
    [
      "one"
    ],
    [
      "you"
    ],
    [
      "me"
    ],
    [
      "myself"
    ],
    [
      "herself"
    ],
    [
      "himself"
    ],
    [
      "itself"
    ],
    [
      "there"
    ],
    [
      "this"
    ],
    [
      "that"
    ],
    [
      "what"
    ],
    [
      "how"
    ],
    [
      "when"
    ],
    [
      "where"
    ],
    [
      "which"
    ],
    [
      "who"
    ],
    [
      "whom"
    ],
    [
      "therefore"
    ],
    [
      "thus"
    ],
    [
      "so"
    ],
    [
      "such"
    ],
    [
      "a"
    ],
    [
      "the"
    ],
    [
      "his"
    ],
    [
      "its"
    ],
    [
      "their"
    ],
    [
      "their own"
    ],
    [
      "somebody"
    ],
    [
      "anybody"
    ],
    [
      "nobody"
    ],
    [
      "everything"
    ],
    [
      "nothing"
    ],
    [
      "something"
    ]
  ]
}}
#INPUT DATA:
{input_text}
#OUTPUT:
'''

    @staticmethod
    def _find_next_occurrence(source_text: str, target: str, start: int) -> int | None:
        if start < 0:
            start = 0
        direct_index = source_text.find(target, start)
        if direct_index != -1:
            return direct_index
        pattern = re.compile(re.escape(target), flags=re.IGNORECASE)
        match = pattern.search(source_text, pos=start)
        if match:
            return match.start()
        return None

    @staticmethod
    def _append_occurrence_indices(
        source_text: str, entities: dict[str, list[str] | None]
    ) -> dict[str, list[list[str | int | None]] | None]:
        results: dict[str, list[list[str | int | None]] | None] = {}
        if not source_text:
            for key, values in entities.items():
                if not values:
                    results[key] = None
                else:
                    formatted = [
                        [
                            value if isinstance(value, str) else str(value),
                            None,
                        ]
                        for value in values
                    ]
                    results[key] = formatted
            return results

        offsets: dict[str, int] = {}
        records: list[dict[str, Any]] = []
        grouped: dict[str, list[dict[str, Any]]] = {}

        for key, values in entities.items():
            if not values:
                results[key] = None
                continue

            grouped[key] = []
            for seq_index, raw_value in enumerate(values):
                original_value = raw_value if isinstance(raw_value, str) else str(raw_value)
                record: dict[str, Any] = {
                    "key": key,
                    "seq_index": seq_index,
                    "raw_value": original_value,
                    "position": None,
                    "order": None,
                    "global_seq": len(records),
                }

                if isinstance(raw_value, str):
                    value = raw_value.strip()
                    if value:
                        offset_key = value.lower()
                        start_pos = offsets.get(offset_key, 0)
                        index = LitBankLinkNER._find_next_occurrence(source_text, value, start_pos)
                        if index is not None:
                            offsets[offset_key] = index + len(value)
                            record["position"] = index

                records.append(record)
                grouped[key].append(record)

        matched_records = sorted(
            (record for record in records if record["position"] is not None),
            key=lambda record: (record["position"], record["global_seq"]),
        )

        for order, record in enumerate(matched_records):
            record["order"] = order

        for key, group in grouped.items():
            results[key] = [[record["raw_value"], record["order"]] for record in group]

        return results

    def NER_recognize(self, text: str) -> dict[str, list[list[str | int | None]] | None]:
        if not text or not text.strip():
            return {"PROPER_NOUN": None, "NOUN_PHRASE": None, "PRONOUNS": None}

        available_models = self.check_available_models()
        if not available_models:
            raise RuntimeError("No models available for NER recognition.")
        model_name = available_models[0]

        # This is similar to context_base = dict(input_text=input_text)
        # self.NER_USER_PROMPT_TEMPLATE.format(**context_base)
        prompt = (
            f"{self.NER_SYSTEM_PROMPT}\n\n"
            f"{self.NER_USER_PROMPT_TEMPLATE.format(input_text=text.strip())}"
        )

        response = self.get_response(prompt, model_name, max_tokens=2048)
        text_response, _ = self._response_to_text(response)
        cleaned_response = text_response.strip()

        start = cleaned_response.find("{")
        end = cleaned_response.rfind("}")
        if start != -1 and end != -1:
            cleaned_response = cleaned_response[start : end + 1]

        try:
            parsed = json.loads(cleaned_response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON returned by NER model: {text_response}") from exc

        normalized: dict[str, list[str] | None] = {}
        for key in ("PROPER_NOUN", "NOUN_PHRASE", "PRONOUNS"):
            value = parsed.get(key)
            if isinstance(value, list):
                normalized[key] = value
            elif value in (None, "null"):
                normalized[key] = None
            elif isinstance(value, str) and value.strip():
                normalized[key] = [value.strip()]
            else:
                normalized[key] = None

        return self._append_occurrence_indices(text, normalized)


def _tokenize_for_jaccard(text: str) -> set[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return set(tokens)


def _jaccard_score(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


def _load_ground_truth_entities(ann_path: Path) -> list[str]:
    entities: list[str] = []
    if not ann_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ann_path}")
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or not line.startswith("T"):
            continue
        parts = line.split("\t", 2)
        if len(parts) == 3:
            entity_name = parts[2].strip()
            if entity_name:
                entities.append(entity_name)
    return entities


def _collect_predicted_entities(predictions: dict[str, list[list[str | int | None]] | None]) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()
    for values in predictions.values():
        if not values:
            continue
        for entry in values:
            if not entry:
                continue
            name = entry[0]
            if isinstance(name, str):
                normalized = name.strip()
                if normalized and normalized.lower() not in seen:
                    collected.append(normalized)
                    seen.add(normalized.lower())
    return collected


def evaluate_ner_accuracy(
    predictions: dict[str, list[list[str | int | None]] | None],
    ground_truth_ann: Path,
    jaccard_threshold: float = 0.5,
) -> tuple[float, list[str], list[str], list[str]]:
    ground_truth_entities = _load_ground_truth_entities(ann_path=ground_truth_ann)
    if not ground_truth_entities:
        return 0.0, [], _collect_predicted_entities(predictions), ground_truth_entities
    predicted_entities = _collect_predicted_entities(predictions)

    predicted_token_sets = [
        (_tokenize_for_jaccard(name), name, idx) for idx, name in enumerate(predicted_entities)
    ]
    correct = 0
    unmatched_ground_truth: list[str] = []
    matched_predicted_indices: set[int] = set()

    for gold_name in ground_truth_entities:
        gold_tokens = _tokenize_for_jaccard(gold_name)
        gold_lower = gold_name.lower()
        matched_index: int | None = None
        for predicted_tokens, predicted_name, pred_idx in predicted_token_sets:
            predicted_lower = predicted_name.lower()
            if (
                _jaccard_score(gold_tokens, predicted_tokens) > jaccard_threshold
                or gold_lower in predicted_lower
                or predicted_lower in gold_lower
            ):
                matched_index = pred_idx
                break
        if matched_index is not None:
            correct += 1
            matched_predicted_indices.add(matched_index)
        else:
            unmatched_ground_truth.append(gold_name)

    unmatched_predictions = [
        predicted_entities[idx] for idx in range(len(predicted_entities)) if idx not in matched_predicted_indices
    ]
    accuracy = correct / len(ground_truth_entities)
    return accuracy, unmatched_ground_truth, unmatched_predictions, ground_truth_entities


if __name__ == "__main__":

    # ------ Test LitBankLinkNER NER recognition and Calculate accuracy------
    litbank_link = LitBankLinkNER(
        api_key="",
        base_url="",
    )
    test_text_path = (Path(__file__).resolve().parent / "../litbank/original/2489_moby_dick.txt").resolve()
    # test_text_path = (Path(__file__).resolve().parent / "../litbank/original/120_treasure_island.txt").resolve()
    # test_text_path = (Path(__file__).resolve().parent / "../litbank/original/11_alices_adventures_in_wonderland.txt").resolve()
    test_text = test_text_path.read_text(encoding="utf-8")
    chunks = litbank_link.extract_chunks_from(test_text, chunk_index=0)
    print("Chunk count:", len(chunks))

    # chunk_test_path = (Path(__file__).resolve().parent / "../litbank/entities/brat/11_alices_adventures_in_wonderland_brat.txt").resolve()
    # chunk_test_path = (Path(__file__).resolve().parent / "../litbank/entities/brat/120_treasure_island_brat.txt").resolve()
    chunk_test_path = (Path(__file__).resolve().parent / "../litbank/entities/brat/2489_moby_dick_brat.txt").resolve()
    chunk_test_text = chunk_test_path.read_text(encoding="utf-8")

    target_length = len(chunk_test_text)
    max_chunks_to_consider = min(5, len(chunks))
    best_combined_text = ""
    best_chunk_count = 0
    smallest_diff = float("inf")

    cumulative_text = ""
    for idx in range(max_chunks_to_consider):
        cumulative_text += chunks[idx]
        diff = abs(len(cumulative_text) - target_length)
        if diff < smallest_diff:
            smallest_diff = diff
            best_chunk_count = idx + 1
            best_combined_text = cumulative_text

    remaining_diff = target_length - len(best_combined_text)
    if remaining_diff > 64 and best_chunk_count < len(chunks):
        best_combined_text += chunks[best_chunk_count]
        best_chunk_count += 1
        smallest_diff = abs(len(best_combined_text) - target_length)

    output_path = Path("./temp.txt")
    output_path.write_text(best_combined_text, encoding="utf-8")
    print(f"Target chunk length: {target_length}")
    print(f"Using first {best_chunk_count} chunk(s) with combined length {len(best_combined_text)} (diff {smallest_diff})")
    
    ner_result = litbank_link.NER_recognize(best_combined_text)
    print("NER Recognition Result:")
    print(json.dumps(ner_result, ensure_ascii=False, indent=2))

    # ground_truth_path = (Path(__file__).resolve().parent / "../litbank/entities/brat/11_alices_adventures_in_wonderland_brat.ann").resolve()
    # ground_truth_path = (Path(__file__).resolve().parent / "../litbank/entities/brat/120_treasure_island_brat.ann").resolve()
    ground_truth_path = (Path(__file__).resolve().parent / "../litbank/entities/brat/2489_moby_dick_brat.ann").resolve()
    accuracy, unmatched_ground_truth, unmatched_predictions, ground_truth_entities = evaluate_ner_accuracy(
        ner_result, ground_truth_path
    )
    mismatched_summary = {
        "unmatched_ground_truth": unmatched_ground_truth,
        "unmatched_predictions": unmatched_predictions,
    }
    result_output_path = Path("./res_NER_acc.txt")
    result_output_path.write_text(
        "#This is prediction result\n"
        f"{json.dumps(ner_result, ensure_ascii=False, indent=2)}\n"
        "#This is Mismatched entities\n"
        f"{json.dumps(mismatched_summary, ensure_ascii=False, indent=2)}\n"
        "#This is groundtruth entities\n"
        f"{json.dumps(ground_truth_entities, ensure_ascii=False, indent=2)}\n"
        "#This is accuracy\n"
        f"{accuracy:.4f}\n",
        encoding="utf-8",
    )
    print(f"NER accuracy: {accuracy:.4f}")
    