"""
Grader for the GPT tokenizer unit. No pytest.

From the notebook:
    from test_tokenizer import grade
    grade(text_to_ids, ids_to_text, get_stats, merge, BasicTokenizer, upto=3)
    grade(text_to_ids, ids_to_text, get_stats, merge, BasicTokenizer,
          RegexTokenizer, compression_ratio, skip=(7,))   # skip a stretch milestone

Milestones run in order and the grader stops at the first failure.

The expected merge sequences and token ids below were produced by a reference
BPE trained on TRAIN_TEXT with the tie-break rule stated in the notebook
docstrings (first-seen pair wins). They are constants so that the reference
implementation itself does not live in this file.
"""

import os
import re
import sys
import traceback

# Filled in by grade(...) so the notebook can hand over its own functions.
text_to_ids = ids_to_text = get_stats = merge = None
BasicTokenizer = RegexTokenizer = compression_ratio = None

HERE = os.path.dirname(os.path.abspath(__file__))
SHAKESPEARE = os.path.join(HERE, "..", "data", "tinyshakespeare.txt")


# ----------------------------------------------------------------------------
# harness
# ----------------------------------------------------------------------------


class Fail(Exception):
    pass


_passed = []  # checks passed so far within the current milestone


def check(cond, msg):
    if not cond:
        raise Fail(msg)
    _passed.append(msg)


def _where(exc):
    """'BasicTokenizer.train' for the innermost frame that raised exc."""
    tb = exc.__traceback__
    while tb.tb_next:
        tb = tb.tb_next
    code = tb.tb_frame.f_code
    return getattr(code, "co_qualname", code.co_name)


def _progress():
    if _passed:
        print(
            f"    {len(_passed)} check(s) in this milestone passed before that. "
            f"Most recent one guarded against: {_passed[-1]!r}"
        )


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _show(b):
    """Render a bytes object for a failure message."""
    try:
        return repr(b.decode("utf-8"))
    except Exception:
        return repr(b)


# ----------------------------------------------------------------------------
# fixed texts and reference outputs
# ----------------------------------------------------------------------------

TRAIN_TEXT = (
    "The tokenizer is a completely separate stage from the language model. "
    "It has its own training set, its own training algorithm, and once trained "
    "it performs the translation between strings and token ids in both directions. "
    "The language model never sees the raw text; it sees the token ids. "
    "The tokenizer is the reason the model struggles with spelling, with reversing "
    "a string, and with arithmetic on long numbers."
)
UNSEEN_TEXT = (
    "The model sees tokens, not letters; the tokenizer decides where a string splits."
)
VOCAB_SIZE = 288  # 32 merges

# (pair, new id) in training order, BasicTokenizer on TRAIN_TEXT
BASIC_MERGES = [
    ((32, 116), 256),
    ((105, 110), 257),
    ((104, 101), 258),
    ((32, 115), 259),
    ((105, 116), 260),
    ((115, 32), 261),
    ((101, 114), 262),
    ((114, 97), 263),
    ((256, 258), 264),
    ((257, 103), 265),
    ((111, 110), 266),
    ((101, 110), 267),
    ((101, 108), 268),
    ((97, 110), 269),
    ((260, 104), 270),
    ((256, 111), 271),
    ((271, 107), 272),
    ((272, 267), 273),
    ((32, 105), 274),
    ((101, 116), 275),
    ((259, 116), 276),
    ((264, 32), 277),
    ((256, 263), 278),
    ((44, 32), 279),
    ((84, 258), 280),
    ((259, 101), 281),
    ((97, 103), 282),
    ((282, 101), 283),
    ((283, 32), 284),
    ((109, 111), 285),
    ((285, 100), 286),
    ((286, 268), 287),
]
BASIC_UNSEEN_IDS = [
    280,
    32,
    287,
    281,
    101,
    115,
    273,
    115,
    279,
    110,
    111,
    116,
    32,
    108,
    275,
    116,
    262,
    115,
    59,
    264,
    273,
    105,
    122,
    262,
    32,
    100,
    101,
    99,
    105,
    100,
    101,
    261,
    119,
    258,
    114,
    101,
    32,
    97,
    276,
    114,
    265,
    259,
    112,
    108,
    260,
    115,
    46,
]

# RegexTokenizer (with the notebook's SPLIT_PATTERN) on TRAIN_TEXT
REGEX_MERGES = [
    ((32, 116), 256),
    ((105, 110), 257),
    ((104, 101), 258),
    ((32, 115), 259),
    ((105, 116), 260),
    ((32, 97), 261),
    ((101, 114), 262),
    ((114, 97), 263),
    ((256, 258), 264),
    ((257, 103), 265),
    ((111, 110), 266),
    ((101, 110), 267),
    ((101, 108), 268),
    ((260, 104), 269),
    ((256, 111), 270),
    ((270, 107), 271),
    ((271, 267), 272),
    ((32, 105), 273),
    ((101, 116), 274),
    ((259, 116), 275),
    ((32, 260), 276),
    ((256, 263), 277),
    ((84, 258), 278),
    ((259, 101), 279),
    ((97, 103), 280),
    ((280, 101), 281),
    ((32, 108), 282),
    ((32, 109), 283),
    ((283, 111), 284),
    ((284, 100), 285),
    ((285, 268), 286),
    ((277, 257), 287),
]
REGEX_UNSEEN_IDS = [
    278,
    286,
    279,
    101,
    115,
    272,
    115,
    44,
    32,
    110,
    111,
    116,
    282,
    274,
    116,
    262,
    115,
    59,
    264,
    272,
    105,
    122,
    262,
    32,
    100,
    101,
    99,
    105,
    100,
    101,
    115,
    32,
    119,
    258,
    114,
    101,
    261,
    275,
    114,
    265,
    259,
    112,
    108,
    260,
    115,
    46,
]
SPECIAL_TOKENS = {"<|endoftext|>": 300}
SPECIAL_TEXT = "hello<|endoftext|>world<|endoftext|>"
SPECIAL_IDS = [258, 108, 108, 111, 300, 119, 111, 114, 108, 100, 300]


def _shakespeare(n):
    if os.path.exists(SHAKESPEARE):
        with open(SHAKESPEARE, encoding="utf-8") as f:
            return f.read()[:n]
    return (TRAIN_TEXT * (n // len(TRAIN_TEXT) + 1))[:n]


# ----------------------------------------------------------------------------
# milestone 1: text <-> bytes <-> ints
# ----------------------------------------------------------------------------


def m1_bytes_roundtrip():
    ids = text_to_ids("hello")
    check(
        isinstance(ids, list),
        f"text_to_ids should return a list, got {type(ids).__name__}",
    )
    check(
        ids == [104, 101, 108, 108, 111],
        f"text_to_ids('hello') gave {ids}, expected the five byte values of h e l l o",
    )
    check(all(isinstance(i, int) for i in ids), "elements should be plain ints")

    for s, n in [("a", 1), ("é", 2), ("안", 3), ("😀", 4)]:
        got = len(text_to_ids(s))
        check(
            got == n,
            f"text_to_ids({s!r}) has {got} elements, expected {n}. One character "
            "is not one byte; think about which encoding makes ASCII cost one byte "
            "and everything else more.",
        )
    ids = text_to_ids("안녕 😀")
    check(
        all(0 <= i < 256 for i in ids),
        f"text_to_ids('안녕 😀') contains values outside 0..255: {ids}. "
        "Those look like code points, not bytes.",
    )

    for s in ["hello world", "안녕하세요 👋 héllo", "", "\n\t tabs and\nnewlines"]:
        back = ids_to_text(text_to_ids(s))
        check(back == s, f"round trip broke: {s!r} -> {text_to_ids(s)} -> {back!r}")

    # a byte sequence that is not valid UTF-8 must not crash the decoder
    try:
        out = ids_to_text([128])
    except UnicodeDecodeError:
        raise Fail(
            "ids_to_text([128]) raised UnicodeDecodeError. A language model can emit "
            "any byte sequence, including ones that are not valid UTF-8; the decoder "
            "has to survive that."
        )
    check(
        out == "�",
        f"ids_to_text([128]) gave {out!r}; an invalid byte should come back as the "
        "replacement character U+FFFD.",
    )


# ----------------------------------------------------------------------------
# milestone 2: count consecutive pairs
# ----------------------------------------------------------------------------


def m2_get_stats():
    st = get_stats([1, 2, 3, 1, 2])
    check(
        isinstance(st, dict), f"get_stats should return a dict, got {type(st).__name__}"
    )
    check(
        st == {(1, 2): 2, (2, 3): 1, (3, 1): 1},
        f"get_stats([1,2,3,1,2]) gave {st}, expected {{(1, 2): 2, (2, 3): 1, (3, 1): 1}}. "
        "Keys are (left, right) tuples of consecutive ids.",
    )
    st = get_stats([7, 7, 7])
    check(
        st == {(7, 7): 2},
        f"get_stats([7,7,7]) gave {st}, expected {{(7, 7): 2}}: three 7s in a row "
        "contain two consecutive pairs.",
    )
    check(get_stats([]) == {}, f"get_stats([]) should be {{}}, got {get_stats([])}")
    check(get_stats([5]) == {}, f"get_stats([5]) should be {{}}, got {get_stats([5])}")

    # accumulate into an existing dict (needed later for regex chunks)
    acc = {(1, 2): 5}
    out = get_stats([1, 2, 9], acc)
    check(
        out == {(1, 2): 6, (2, 9): 1},
        f"get_stats([1,2,9], counts={{(1,2): 5}}) gave {out}, expected "
        "{(1, 2): 6, (2, 9): 1}. When a counts dict is passed in, counts should be "
        "added to it, not started from zero.",
    )
    check(
        out is acc,
        "get_stats(ids, counts) should return the same dict object it was given.",
    )

    # the most common pair in real text should be findable with max(stats, key=stats.get)
    st = get_stats(text_to_ids("aaab aaab aaab"))
    top = max(st, key=st.get)
    check(
        top == (97, 97) and st[top] == 6,
        f"on 'aaab aaab aaab' the top pair is {top} with count {st[top]}; expected "
        "(97, 97) with count 6.",
    )


# ----------------------------------------------------------------------------
# milestone 3: merge one pair everywhere
# ----------------------------------------------------------------------------


def m3_merge():
    out = merge([5, 6, 6, 7, 9, 1], (6, 7), 99)
    check(
        isinstance(out, list), f"merge should return a list, got {type(out).__name__}"
    )
    check(
        out == [5, 6, 99, 9, 1],
        f"merge([5,6,6,7,9,1], (6,7), 99) gave {out}, expected [5, 6, 99, 9, 1].",
    )

    out = merge([1, 2, 3], (1, 2), 99)
    check(
        out == [99, 3],
        f"merge([1,2,3], (1,2), 99) gave {out}, expected [99, 3]. "
        + ("The element after the merged pair went missing." if out == [99] else ""),
    )

    out = merge([3, 1, 2], (1, 2), 99)
    check(
        out == [3, 99],
        f"merge([3,1,2], (1,2), 99) gave {out}, expected [3, 99]. The pair sits at "
        "the very end of the list; is the loop looking that far?",
    )

    out = merge([1, 2], (1, 2), 99)
    check(out == [99], f"merge([1,2], (1,2), 99) gave {out}, expected [99].")

    out = merge([1, 2, 1, 2, 1, 2], (1, 2), 99)
    check(
        out == [99, 99, 99],
        f"merge([1,2,1,2,1,2], (1,2), 99) gave {out}, expected [99, 99, 99]. Every "
        "occurrence should be replaced, and after a replacement the scan must skip "
        "past both consumed elements.",
    )

    out = merge([7, 7, 7], (7, 7), 99)
    check(
        out == [99, 7],
        f"merge([7,7,7], (7,7), 99) gave {out}, expected [99, 7]: overlapping pairs "
        "are merged left to right without reuse.",
    )

    out = merge([2, 1, 2, 1], (1, 2), 99)
    check(
        out == [2, 99, 1],
        f"merge([2,1,2,1], (1,2), 99) gave {out}, expected [2, 99, 1].",
    )

    src = [4, 5, 6]
    out = merge(src, (8, 9), 99)
    check(
        out == [4, 5, 6],
        f"merge with a pair that never occurs gave {out}, expected [4, 5, 6].",
    )
    check(src == [4, 5, 6], "merge mutated its input list; it should build a new one.")

    check(merge([], (1, 2), 99) == [], "merge([]) should return [].")
    check(merge([1], (1, 2), 99) == [1], "merge([1], (1,2), 99) should return [1].")


# ----------------------------------------------------------------------------
# milestone 4: the BPE training loop
# ----------------------------------------------------------------------------


def _check_merge_order(tok_merges, ref_merges, chunks_fn, what):
    """Replay training step by step using the learner's own get_stats/merge
    (already validated) to diagnose where the merge order diverges."""
    check(
        isinstance(tok_merges, dict),
        f"{what}.merges should be a dict of (int, int) -> int, got {type(tok_merges).__name__}",
    )
    check(
        len(tok_merges) == len(ref_merges),
        f"{what}.merges has {len(tok_merges)} entries after training to vocab_size "
        f"{VOCAB_SIZE}; expected {len(ref_merges)}. How many merges does it take to "
        "get from 256 byte tokens to vocab_size tokens?",
    )
    for k, v in tok_merges.items():
        check(
            isinstance(k, tuple) and len(k) == 2 and isinstance(v, int),
            f"{what}.merges entry {k!r}: {v!r} is not (int, int) -> int",
        )
    ids_learner = sorted(tok_merges.values())
    check(
        ids_learner == list(range(256, 256 + len(ref_merges))),
        f"{what}.merges assigns ids {ids_learner[:3]}..{ids_learner[-1]}; expected "
        "256, 257, ... consecutively. New token ids should start right after the "
        "256 byte tokens.",
    )
    learner_seq = sorted(tok_merges.items(), key=lambda kv: kv[1])

    chunks = chunks_fn()
    for step, ((ref_pair, ref_idx), (l_pair, l_idx)) in enumerate(
        zip(ref_merges, learner_seq)
    ):
        stats = {}
        for c in chunks:
            get_stats(c, stats)
        if l_pair != ref_pair:
            l_count = stats.get(l_pair, 0)
            r_count = stats[ref_pair]
            if l_count == r_count:
                raise Fail(
                    f"merge #{step + 1} (token {l_idx}): you merged {l_pair} "
                    f"{_show(_bytes_of(l_pair, chunks_fn, learner_seq[:step]))} but the "
                    f"reference merged {ref_pair} "
                    f"{_show(_bytes_of(ref_pair, chunks_fn, learner_seq[:step]))}. Both "
                    f"occur {r_count} times: this is a tie, and your tie-break picks a "
                    "different winner than the contract in the docstring."
                )
            if l_count == 0:
                raise Fail(
                    f"merge #{step + 1} (token {l_idx}): you merged {l_pair} "
                    f"{_show(_bytes_of(l_pair, chunks_fn, learner_seq[:step]))}, but at "
                    "that point this pair does not occur inside any chunk at all; the "
                    f"most common pair is {ref_pair} with {r_count} occurrences. The "
                    "statistics are being counted on something other than the chunks."
                )
            raise Fail(
                f"merge #{step + 1} (token {l_idx}): you merged {l_pair}, which occurs "
                f"{l_count} times at that point, but the most common pair is {ref_pair} "
                f"with {r_count} occurrences. Merges are happening in a different order "
                "than most-common-first; check what the stats are being computed on."
            )
        chunks = [merge(c, ref_pair, ref_idx) for c in chunks]


def _bytes_of(pair, chunks_fn, prior_merges):
    """Bytes of a pair given the merges done so far (for messages only)."""
    vocab = {i: bytes([i]) for i in range(256)}
    for (p0, p1), idx in prior_merges:
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab.get(pair[0], b"?") + vocab.get(pair[1], b"?")


def _check_vocab(tok, ref_merges, what):
    check(
        isinstance(tok.vocab, dict),
        f"{what}.vocab should be a dict of int -> bytes, got {type(tok.vocab).__name__}",
    )
    check(
        len(tok.vocab) == 256 + len(ref_merges),
        f"{what}.vocab has {len(tok.vocab)} entries, expected {256 + len(ref_merges)} "
        "(the 256 byte tokens plus one per merge).",
    )
    for i in range(256):
        check(
            tok.vocab.get(i) == bytes([i]),
            f"{what}.vocab[{i}] is {tok.vocab.get(i)!r}, expected {bytes([i])!r}. The "
            "first 256 entries are the raw bytes.",
        )
    vocab = {i: bytes([i]) for i in range(256)}
    for (p0, p1), idx in ref_merges:
        vocab[idx] = vocab[p0] + vocab[p1]
        check(
            isinstance(tok.vocab.get(idx), bytes),
            f"{what}.vocab[{idx}] is {tok.vocab.get(idx)!r}; every entry should be a "
            "bytes object.",
        )
        check(
            tok.vocab[idx] == vocab[idx],
            f"{what}.vocab[{idx}] is {_show(tok.vocab[idx])} but that token was made by "
            f"merging {p0} and {p1}, so it should spell {_show(vocab[idx])}. The vocab "
            "entry for a merged token is built from the entries of its two parts, in "
            "order.",
        )


def m4_train():
    tok = BasicTokenizer()
    tok.train(TRAIN_TEXT, VOCAB_SIZE)
    _check_merge_order(
        tok.merges, BASIC_MERGES, lambda: [text_to_ids(TRAIN_TEXT)], "BasicTokenizer"
    )
    _check_vocab(tok, BASIC_MERGES, "BasicTokenizer")

    # trivial cases
    t2 = BasicTokenizer()
    t2.train("abc", 256)
    check(
        t2.merges == {},
        f"train(text, vocab_size=256) should do zero merges, but merges = {t2.merges}",
    )
    t3 = BasicTokenizer()
    t3.train("ab", 300)
    check(
        len(t3.merges) <= 1,
        f"train('ab', 300) produced {len(t3.merges)} merges. After the single pair is "
        "merged there is nothing left to count; the loop should stop rather than "
        "invent merges.",
    )

    # it compresses: a longer training run on real text
    text = _shakespeare(4000)
    t4 = BasicTokenizer()
    t4.train(text, 320)
    check(
        len(t4.merges) == 64,
        f"expected 64 merges on the Shakespeare slice, got {len(t4.merges)}",
    )
    ids = text_to_ids(text)
    for pair, idx in sorted(t4.merges.items(), key=lambda kv: kv[1]):
        ids = merge(ids, pair, idx)
    ratio = len(text.encode("utf-8")) / len(ids)
    check(
        ratio > 1.5,
        f"after 64 merges on 4000 chars of Shakespeare the text shrinks by only "
        f"{ratio:.3f}x; the reference gets about 1.64x. The merges being chosen are "
        "not the most common pairs.",
    )


# ----------------------------------------------------------------------------
# milestone 5: encode / decode
# ----------------------------------------------------------------------------


def _check_encode_decode(tok, ref_ids, what):
    # decode first: it only needs vocab
    out = tok.decode([104, 105])
    check(out == "hi", f"{what}.decode([104, 105]) gave {out!r}, expected 'hi'.")
    out = tok.decode(list(tok.merges.values())[:1])
    check(
        isinstance(out, str),
        f"{what}.decode should return a str, got {type(out).__name__}",
    )
    try:
        out = tok.decode([128])
    except UnicodeDecodeError:
        raise Fail(
            f"{what}.decode([128]) raised UnicodeDecodeError. Same lesson as milestone 1: "
            "the decoder must survive byte sequences that are not valid UTF-8."
        )
    check(
        out == "�",
        f"{what}.decode([128]) gave {out!r}, expected the replacement character.",
    )

    ids = tok.encode("")
    check(ids == [], f"{what}.encode('') gave {ids!r}, expected [].")
    ids = tok.encode("x")
    check(ids == [120], f"{what}.encode('x') gave {ids}, expected [120].")

    for s in ["hello world", "안녕 😀 héllo", "the the the", TRAIN_TEXT[:80]]:
        ids = tok.encode(s)
        check(isinstance(ids, list), f"{what}.encode should return a list")
        check(
            all(isinstance(i, int) and i in tok.vocab for i in ids),
            f"{what}.encode({s!r}) produced ids that are not in the vocab: "
            f"{[i for i in ids if i not in tok.vocab][:5]}",
        )
        back = tok.decode(ids)
        check(
            back == s,
            f"{what} round trip broke: {s!r} -> {ids} -> {back!r}",
        )

    ids = tok.encode(UNSEEN_TEXT)
    check(
        len(ids) < len(UNSEEN_TEXT.encode("utf-8")),
        f"{what}.encode on unseen text produced {len(ids)} ids for "
        f"{len(UNSEEN_TEXT.encode('utf-8'))} bytes. No merge was applied at all.",
    )
    if ids != ref_ids:
        for pos, (a, b) in enumerate(zip(ids, ref_ids)):
            if a != b:
                raise Fail(
                    f"{what}.encode(UNSEEN_TEXT): at position {pos}, token {a} "
                    f"{_show(tok.vocab.get(a, b'?'))} appears where the reference has "
                    f"{b} {_show(tok.vocab.get(b, b'?'))} (you produced {len(ids)} ids, "
                    f"reference {len(ref_ids)}). The text round-trips, so the merges "
                    "are being applied in a different order than they were learned."
                )
        raise Fail(
            f"{what}.encode(UNSEEN_TEXT) produced {len(ids)} ids but the reference "
            f"produced {len(ref_ids)}; the prefixes agree, the lengths do not."
        )


def m5_encode_decode():
    tok = BasicTokenizer()
    tok.train(TRAIN_TEXT, VOCAB_SIZE)
    _check_encode_decode(tok, BASIC_UNSEEN_IDS, "BasicTokenizer")

    # encode must reproduce the training tokenization
    ids = text_to_ids(TRAIN_TEXT)
    for pair, idx in sorted(tok.merges.items(), key=lambda kv: kv[1]):
        ids = merge(ids, pair, idx)
    got = tok.encode(TRAIN_TEXT)
    check(
        got == ids,
        f"encoding the training text gives {len(got)} ids but training itself ended "
        f"with {len(ids)}. encode should reproduce exactly the sequence training "
        "arrived at.",
    )


# ----------------------------------------------------------------------------
# milestone 6 (stretch): regex pre-tokenization
# ----------------------------------------------------------------------------


def m6_regex():
    tok = RegexTokenizer()
    check(
        hasattr(tok, "pattern") and isinstance(tok.pattern, str),
        "RegexTokenizer should keep its split pattern as a string in .pattern",
    )
    chunks = re.findall(tok.pattern, "Hello world's 123 done!!  ok")
    check(
        chunks == ["Hello", " world", "'s", " 123", " done", "!!", " ", " ok"],
        f're.findall(pattern, "Hello world\'s 123 done!!  ok") gave {chunks}; the '
        "given SPLIT_PATTERN should split that into "
        "['Hello', ' world', \"'s\", ' 123', ' done', '!!', ' ', ' ok']. "
        "Is the given pattern being used unchanged?",
    )

    tok.train(TRAIN_TEXT, VOCAB_SIZE)
    _check_merge_order(
        tok.merges,
        REGEX_MERGES,
        lambda: [text_to_ids(c) for c in re.findall(tok.pattern, TRAIN_TEXT)],
        "RegexTokenizer",
    )
    _check_vocab(tok, REGEX_MERGES, "RegexTokenizer")

    # no learned token may straddle a letter/space boundary
    for idx, b in tok.vocab.items():
        if idx < 256:
            continue
        s = b.decode("utf-8", errors="replace")
        check(
            not re.search(r"\S\s+\S", s),
            f"RegexTokenizer.vocab[{idx}] = {s!r} spans a space between two words. "
            "Pairs are being counted across chunk boundaries.",
        )

    _check_encode_decode(tok, REGEX_UNSEEN_IDS, "RegexTokenizer")

    # whitespace runs: the pattern hands the last space to the word that follows,
    # so encode must see the same boundaries train saw
    tok3 = RegexTokenizer()
    tok3.train("    a    b    c    d    e    f", 262)
    ids = tok3.encode("  x")
    check(
        ids == [32, 32, 120],
        f"after training on runs of four spaces, RegexTokenizer.encode('  x') gave "
        f"{ids} = {[_show(tok3.vocab.get(i, b'?')) for i in ids]}, expected "
        "[32, 32, 120] = [' ', ' ', 'x']. Training never saw those two spaces as one "
        "chunk (the pattern gives the last space to the word after it), yet encode "
        "merged them. encode is not cutting the text where train cut it.",
    )

    # the classic: a token learned as ' the' must not be glued to the following word
    tok2 = RegexTokenizer()
    tok2.train("the cat the cat the cat the cat", 260)
    ids = tok2.encode("the cat")
    for i in ids:
        s = tok2.vocab[i].decode("utf-8", errors="replace")
        check(
            not re.search(r"\S\s+\S", s),
            f"RegexTokenizer.encode('the cat') produced token {i} = {s!r}, which "
            "spans a space. encode is merging across chunks even though training "
            "did not.",
        )


# ----------------------------------------------------------------------------
# milestone 7 (stretch): special tokens + compression ratio
# ----------------------------------------------------------------------------


def m7_special_and_ratio():
    tok = RegexTokenizer()
    tok.train(TRAIN_TEXT, VOCAB_SIZE)

    # without special tokens, encode_special is plain encode
    check(
        tok.encode_special("hello world") == tok.encode("hello world"),
        "with no special tokens registered, encode_special should equal encode",
    )

    tok.special_tokens = dict(SPECIAL_TOKENS)
    ids = tok.encode_special("<|endoftext|>")
    check(
        ids == [300],
        f"encode_special('<|endoftext|>') gave {ids}, expected [300]. The special "
        "string should become one id, never be fed through BPE.",
    )
    ids = tok.encode_special(SPECIAL_TEXT)
    check(
        ids == SPECIAL_IDS,
        f"encode_special({SPECIAL_TEXT!r}) gave {ids}, expected {SPECIAL_IDS}. The "
        "ordinary text between special tokens should be encoded exactly as encode "
        "would, and the special tokens should sit in between in order.",
    )
    ids = tok.encode_special("<|endoftext|><|endoftext|>x")
    check(
        ids == [300, 300, 120],
        f"encode_special('<|endoftext|><|endoftext|>x') gave {ids}, expected "
        "[300, 300, 120]. Adjacent special tokens, and an empty chunk between them, "
        "must be handled.",
    )
    ids = tok.encode_special("plain text only")
    check(
        ids == tok.encode("plain text only"),
        "encode_special on text with no special tokens should match encode exactly",
    )
    ids = tok.encode("<|endoftext|>")
    check(
        300 not in ids,
        "plain encode('<|endoftext|>') produced the special id 300. Plain encode "
        "should treat that string as ordinary characters; only encode_special "
        "recognises specials.",
    )

    # compression ratio
    r = compression_ratio("hello", [1, 2, 3, 4, 5])
    check(close(r, 1.0), f"compression_ratio('hello', 5 ids) gave {r}, expected 1.0")
    r = compression_ratio("hello", [1])
    check(close(r, 5.0), f"compression_ratio('hello', 1 id) gave {r}, expected 5.0")
    r = compression_ratio("😀", [1, 2])
    check(
        close(r, 2.0),
        f"compression_ratio('😀', 2 ids) gave {r}, expected 2.0: the ratio is over "
        "bytes, not characters.",
    )

    text = _shakespeare(4000)
    b = BasicTokenizer()
    b.train(text, 320)
    r = compression_ratio(text, b.encode(text))
    check(
        r > 1.5,
        f"BasicTokenizer at vocab 320 compresses the Shakespeare slice {r:.3f}x; "
        "expected over 1.5x.",
    )


# ----------------------------------------------------------------------------

MILESTONES = [
    (1, "text -> utf-8 bytes -> ints, and back", m1_bytes_roundtrip),
    (2, "get_stats: count consecutive pairs", m2_get_stats),
    (3, "merge: replace a pair everywhere", m3_merge),
    (4, "train: the BPE loop, merges and vocab", m4_train),
    (5, "encode / decode round trip, in merge order", m5_encode_decode),
    (6, "STRETCH: regex pre-tokenization", m6_regex),
    (7, "STRETCH: special tokens + compression ratio", m7_special_and_ratio),
]


def grade(
    text_to_ids_fn,
    ids_to_text_fn,
    get_stats_fn=None,
    merge_fn=None,
    basic_cls=None,
    regex_cls=None,
    compression_ratio_fn=None,
    upto=99,
    skip=(),
):
    """Run milestones in order, stopping at the first failure.

    From a notebook:
        grade(text_to_ids, ids_to_text, upto=1)
        grade(text_to_ids, ids_to_text, get_stats, merge, BasicTokenizer, upto=5)
        grade(text_to_ids, ids_to_text, get_stats, merge, BasicTokenizer,
              RegexTokenizer, compression_ratio)
        grade(text_to_ids, ids_to_text, get_stats, merge, BasicTokenizer,
              RegexTokenizer, compression_ratio, skip=(7,))   # skip a stretch milestone
    Returns 0 on all-pass, 1 otherwise.
    """
    global text_to_ids, ids_to_text, get_stats, merge
    global BasicTokenizer, RegexTokenizer, compression_ratio
    text_to_ids, ids_to_text = text_to_ids_fn, ids_to_text_fn
    get_stats, merge = get_stats_fn, merge_fn
    BasicTokenizer, RegexTokenizer = basic_cls, regex_cls
    compression_ratio = compression_ratio_fn
    for num, title, fn in MILESTONES:
        if num > upto:
            break
        if num in skip:
            print(f"[skip] milestone {num}: {title}")
            continue
        _passed.clear()
        try:
            fn()
        except Fail as e:
            print(f"\n[FAIL] milestone {num}: {title}\n")
            print(f"    {e}")
            _progress()
            print()
            return 1
        except NotImplementedError as e:
            print(f"\n[TODO] milestone {num}: {title}")
            print(
                f"    {_where(e)} raised NotImplementedError -- this is the next thing to write."
            )
            _progress()
            print()
            return 1
        except TypeError as e:
            if "NoneType" in str(e):
                print(f"\n[TODO] milestone {num}: {title}")
                print(
                    "    function or class not passed to grade() yet -- hand it over "
                    "once it exists.\n"
                )
                return 1
            traceback.print_exc()
            return 1
        except Exception:
            print(f"\n[ERROR] milestone {num}: {title}\n")
            traceback.print_exc()
            return 1

        print(f"[ok]   milestone {num}: {title}")
    print("\nall milestones passed.")
    return 0


def main():
    """CLI route: grade tokenizer.py in this folder."""
    import tokenizer as t

    upto = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    return grade(
        t.text_to_ids,
        t.ids_to_text,
        getattr(t, "get_stats", None),
        getattr(t, "merge", None),
        getattr(t, "BasicTokenizer", None),
        getattr(t, "RegexTokenizer", None),
        getattr(t, "compression_ratio", None),
        upto=upto,
    )


if __name__ == "__main__":
    sys.exit(main())
