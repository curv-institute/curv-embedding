"""
Synthetic data generator for curv-embedding evaluation.

Generates test documents with known properties (boundaries, anchors) for
evaluating chunking and retrieval algorithms.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field


@dataclass
class SyntheticDocument:
    """A synthetic document with known semantic boundaries and anchors.

    Attributes:
        doc_id: Unique identifier for the document.
        content: Raw document content as bytes.
        domain: Content domain (text/code/json/logs).
        boundary_offsets: Known semantic boundary byte offsets.
        planted_anchors: Tuples of (start, end, anchor_id) for planted anchors.
    """

    doc_id: str
    content: bytes
    domain: str
    boundary_offsets: list[int] = field(default_factory=list)
    planted_anchors: list[tuple[int, int, str]] = field(default_factory=list)


# Markov-like word generation data for realistic text
_WORD_STARTS = "bcdfghjklmnpqrstvwxyz"
_VOWELS = "aeiou"
_WORD_ENDS = "bcdfgklmnprstxz"

# Common programming keywords and identifiers
_CODE_KEYWORDS = [
    "def", "class", "return", "if", "else", "elif", "for", "while",
    "import", "from", "try", "except", "finally", "with", "as",
    "yield", "async", "await", "lambda", "pass", "break", "continue",
]
_CODE_TYPES = ["int", "str", "float", "bool", "list", "dict", "None", "Any"]
_CODE_NAMES = [
    "data", "result", "value", "item", "count", "index", "name", "config",
    "handler", "manager", "processor", "parser", "builder", "factory",
    "service", "client", "server", "request", "response", "context",
]

# Log levels and components
_LOG_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
_LOG_COMPONENTS = [
    "auth", "api", "db", "cache", "queue", "worker", "scheduler",
    "http", "grpc", "storage", "metrics", "config", "router",
]
_LOG_ACTIONS = [
    "started", "completed", "failed", "retrying", "connected",
    "disconnected", "received", "sent", "processed", "cached",
    "validated", "rejected", "timeout", "initialized", "shutdown",
]


def _generate_word(rng: random.Random, min_len: int = 2, max_len: int = 10) -> str:
    """Generate a pronounceable pseudo-word."""
    length = rng.randint(min_len, max_len)
    word = []
    use_vowel = rng.random() < 0.3  # Sometimes start with vowel

    for i in range(length):
        if use_vowel:
            word.append(rng.choice(_VOWELS))
        else:
            if i == length - 1:
                word.append(rng.choice(_WORD_ENDS))
            else:
                word.append(rng.choice(_WORD_STARTS))
        use_vowel = not use_vowel

    return "".join(word)


def _generate_sentence(rng: random.Random, min_words: int = 5, max_words: int = 20) -> str:
    """Generate a pseudo-sentence with capitalization and punctuation."""
    num_words = rng.randint(min_words, max_words)
    words = [_generate_word(rng) for _ in range(num_words)]
    words[0] = words[0].capitalize()

    # Occasionally insert commas
    for i in range(1, len(words) - 1):
        if rng.random() < 0.15:
            words[i] = words[i] + ","

    # End punctuation
    end_punct = rng.choice([".", ".", ".", "!", "?"])
    return " ".join(words) + end_punct


def _generate_paragraph(rng: random.Random, min_sentences: int = 3, max_sentences: int = 8) -> str:
    """Generate a paragraph of pseudo-sentences."""
    num_sentences = rng.randint(min_sentences, max_sentences)
    return " ".join(_generate_sentence(rng) for _ in range(num_sentences))


def _generate_identifier(rng: random.Random) -> str:
    """Generate a code-like identifier."""
    style = rng.choice(["snake", "camel", "single"])
    if style == "single":
        return rng.choice(_CODE_NAMES)
    elif style == "snake":
        parts = [rng.choice(_CODE_NAMES) for _ in range(rng.randint(2, 3))]
        return "_".join(parts)
    else:  # camel
        parts = [rng.choice(_CODE_NAMES) for _ in range(rng.randint(2, 3))]
        return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _generate_function(rng: random.Random, indent: int = 0) -> str:
    """Generate a function-like code structure."""
    prefix = " " * indent
    name = _generate_identifier(rng)
    num_params = rng.randint(0, 4)
    params = [_generate_identifier(rng) for _ in range(num_params)]

    # Add type hints sometimes
    typed_params = []
    for p in params:
        if rng.random() < 0.5:
            typed_params.append(f"{p}: {rng.choice(_CODE_TYPES)}")
        else:
            typed_params.append(p)

    return_type = rng.choice(_CODE_TYPES) if rng.random() < 0.5 else None
    signature = f"{prefix}def {name}({', '.join(typed_params)})"
    if return_type:
        signature += f" -> {return_type}"
    signature += ":\n"

    # Docstring sometimes
    lines = [signature]
    if rng.random() < 0.6:
        lines.append(f'{prefix}    """')
        lines.append(f"{prefix}    {_generate_sentence(rng, 3, 10)}")
        lines.append(f'{prefix}    """')

    # Body
    body_lines = rng.randint(2, 8)
    for _ in range(body_lines):
        line_type = rng.choice(["assign", "call", "if", "return", "comment"])
        if line_type == "assign":
            var = _generate_identifier(rng)
            val = rng.choice([str(rng.randint(0, 100)), f'"{_generate_word(rng)}"', "None", "[]", "{}"])
            lines.append(f"{prefix}    {var} = {val}")
        elif line_type == "call":
            func = _generate_identifier(rng)
            args = ", ".join(_generate_identifier(rng) for _ in range(rng.randint(0, 3)))
            lines.append(f"{prefix}    {func}({args})")
        elif line_type == "if":
            cond = f"{_generate_identifier(rng)} {rng.choice(['>', '<', '==', '!=', 'is'])} {rng.choice(['None', '0', 'True'])}"
            lines.append(f"{prefix}    if {cond}:")
            lines.append(f"{prefix}        pass")
        elif line_type == "return":
            lines.append(f"{prefix}    return {_generate_identifier(rng)}")
            break
        else:  # comment
            lines.append(f"{prefix}    # {_generate_sentence(rng, 3, 8)}")

    if not lines[-1].strip().startswith("return"):
        lines.append(f"{prefix}    return None")

    return "\n".join(lines) + "\n"


def _generate_class(rng: random.Random) -> str:
    """Generate a class-like code structure."""
    name = "".join(rng.choice(_CODE_NAMES).capitalize() for _ in range(rng.randint(1, 2)))
    lines = [f"class {name}:"]

    # Docstring
    if rng.random() < 0.7:
        lines.append('    """')
        lines.append(f"    {_generate_sentence(rng, 3, 10)}")
        lines.append('    """')

    # Methods
    num_methods = rng.randint(1, 4)
    for i in range(num_methods):
        if i == 0:
            # __init__ method
            lines.append("")
            lines.append("    def __init__(self):")
            lines.append(f"        self.{_generate_identifier(rng)} = None")
        else:
            lines.append("")
            lines.append(_generate_function(rng, indent=4))

    return "\n".join(lines) + "\n"


def _generate_json_object(rng: random.Random, depth: int = 0) -> str:
    """Generate a JSON-like object."""
    if depth > 2:
        # Leaf values only at max depth
        val_type = rng.choice(["string", "number", "bool", "null"])
        if val_type == "string":
            return f'"{_generate_word(rng)}"'
        elif val_type == "number":
            return str(rng.randint(0, 10000) if rng.random() < 0.7 else round(rng.random() * 1000, 2))
        elif val_type == "bool":
            return rng.choice(["true", "false"])
        else:
            return "null"

    num_fields = rng.randint(2, 6)
    fields = []
    indent = "  " * (depth + 1)
    close_indent = "  " * depth

    for _ in range(num_fields):
        key = _generate_identifier(rng)
        val_type = rng.choice(["string", "number", "bool", "null", "array", "object"])

        if val_type == "string":
            value = f'"{_generate_word(rng)}"'
        elif val_type == "number":
            value = str(rng.randint(0, 10000) if rng.random() < 0.7 else round(rng.random() * 1000, 2))
        elif val_type == "bool":
            value = rng.choice(["true", "false"])
        elif val_type == "null":
            value = "null"
        elif val_type == "array":
            arr_len = rng.randint(1, 4)
            arr_vals = [_generate_json_object(rng, depth + 2) for _ in range(arr_len)]
            value = "[" + ", ".join(arr_vals) + "]"
        else:  # object
            value = _generate_json_object(rng, depth + 1)

        fields.append(f'{indent}"{key}": {value}')

    return "{\n" + ",\n".join(fields) + f"\n{close_indent}}}"


def _generate_log_line(rng: random.Random, timestamp_base: int) -> tuple[str, int]:
    """Generate a log line with timestamp. Returns (line, next_timestamp)."""
    # Increment timestamp by 1-5000ms
    timestamp = timestamp_base + rng.randint(1, 5000)

    # Format: ISO-like timestamp
    hours = (timestamp // 3600000) % 24
    minutes = (timestamp // 60000) % 60
    seconds = (timestamp // 1000) % 60
    millis = timestamp % 1000
    ts_str = f"2024-01-15T{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}Z"

    level = rng.choice(_LOG_LEVELS)
    component = rng.choice(_LOG_COMPONENTS)
    action = rng.choice(_LOG_ACTIONS)

    # Generate message
    msg_style = rng.choice(["simple", "key_value", "detailed"])
    if msg_style == "simple":
        message = f"{action} {_generate_word(rng)}"
    elif msg_style == "key_value":
        pairs = [f"{_generate_word(rng, 3, 6)}={rng.randint(0, 1000)}" for _ in range(rng.randint(1, 4))]
        message = f"{action} " + " ".join(pairs)
    else:
        message = f"{action}: {_generate_sentence(rng, 3, 8)}"

    line = f"[{ts_str}] [{level:5}] [{component:10}] {message}\n"
    return line, timestamp


def generate_text_document(seed: int, size_bytes: int) -> SyntheticDocument:
    """Generate a prose-like text document with paragraph breaks.

    Args:
        seed: Random seed for reproducibility.
        size_bytes: Target size in bytes.

    Returns:
        SyntheticDocument with paragraph boundaries marked.
    """
    rng = random.Random(seed)
    doc_id = f"text_{seed:08x}"

    paragraphs = []
    boundaries = []
    anchors = []
    current_size = 0
    anchor_counter = 0

    while current_size < size_bytes:
        para = _generate_paragraph(rng)

        # Plant anchor with 20% probability
        if rng.random() < 0.2 and len(para) > 50:
            anchor_start = current_size + rng.randint(10, min(50, len(para) - 20))
            anchor_end = anchor_start + rng.randint(20, min(100, len(para) - 10))
            anchor_id = f"{doc_id}_anchor_{anchor_counter:04d}"
            anchors.append((anchor_start, anchor_end, anchor_id))
            anchor_counter += 1

        paragraphs.append(para)
        current_size += len(para.encode("utf-8"))

        # Mark paragraph boundary (before the newlines)
        boundaries.append(current_size)
        current_size += 2  # For "\n\n"

    content = "\n\n".join(paragraphs)

    # Trim if over size
    content_bytes = content.encode("utf-8")
    if len(content_bytes) > size_bytes:
        content_bytes = content_bytes[:size_bytes]
        # Adjust boundaries to be within content
        boundaries = [b for b in boundaries if b < size_bytes]
        anchors = [(s, e, aid) for s, e, aid in anchors if e < size_bytes]

    return SyntheticDocument(
        doc_id=doc_id,
        content=content_bytes,
        domain="text",
        boundary_offsets=boundaries,
        planted_anchors=anchors,
    )


def generate_code_document(seed: int, size_bytes: int) -> SyntheticDocument:
    """Generate a function-like code document.

    Args:
        seed: Random seed for reproducibility.
        size_bytes: Target size in bytes.

    Returns:
        SyntheticDocument with function/class boundaries marked.
    """
    rng = random.Random(seed)
    doc_id = f"code_{seed:08x}"

    blocks = []
    boundaries = []
    anchors = []
    current_size = 0
    anchor_counter = 0

    # Add imports at the start
    imports = [
        "from __future__ import annotations\n",
        "import os\n",
        "import sys\n",
        f"from typing import {', '.join(rng.sample(_CODE_TYPES, 3))}\n",
        "\n",
    ]
    header = "".join(imports)
    blocks.append(header)
    current_size = len(header.encode("utf-8"))
    boundaries.append(current_size)

    while current_size < size_bytes:
        # Alternate between functions and classes
        if rng.random() < 0.3:
            block = _generate_class(rng)
        else:
            block = _generate_function(rng)

        block += "\n"  # Extra newline between blocks
        block_bytes = block.encode("utf-8")

        # Plant anchor in function/class body
        if rng.random() < 0.25 and len(block_bytes) > 100:
            # Find a line in the middle
            lines = block.split("\n")
            if len(lines) > 4:
                anchor_line_idx = rng.randint(2, len(lines) - 2)
                line_start = sum(len(l) + 1 for l in lines[:anchor_line_idx])
                anchor_start = current_size + line_start
                anchor_end = anchor_start + len(lines[anchor_line_idx])
                anchor_id = f"{doc_id}_anchor_{anchor_counter:04d}"
                anchors.append((anchor_start, anchor_end, anchor_id))
                anchor_counter += 1

        blocks.append(block)
        current_size += len(block_bytes)
        boundaries.append(current_size)

    content = "".join(blocks)
    content_bytes = content.encode("utf-8")

    if len(content_bytes) > size_bytes:
        content_bytes = content_bytes[:size_bytes]
        boundaries = [b for b in boundaries if b < size_bytes]
        anchors = [(s, e, aid) for s, e, aid in anchors if e < size_bytes]

    return SyntheticDocument(
        doc_id=doc_id,
        content=content_bytes,
        domain="code",
        boundary_offsets=boundaries,
        planted_anchors=anchors,
    )


def generate_json_document(seed: int, size_bytes: int) -> SyntheticDocument:
    """Generate a JSON array of objects.

    Args:
        seed: Random seed for reproducibility.
        size_bytes: Target size in bytes.

    Returns:
        SyntheticDocument with object boundaries marked.
    """
    rng = random.Random(seed)
    doc_id = f"json_{seed:08x}"

    objects = []
    boundaries = []
    anchors = []
    current_size = 1  # Opening bracket
    anchor_counter = 0

    while current_size < size_bytes - 10:  # Leave room for closing
        obj = _generate_json_object(rng)
        obj_bytes = obj.encode("utf-8")

        # Plant anchor in object
        if rng.random() < 0.2 and len(obj_bytes) > 50:
            # Pick a position inside the object
            anchor_offset = rng.randint(10, len(obj_bytes) - 20)
            anchor_start = current_size + anchor_offset
            anchor_end = anchor_start + rng.randint(15, min(50, len(obj_bytes) - anchor_offset - 5))
            anchor_id = f"{doc_id}_anchor_{anchor_counter:04d}"
            anchors.append((anchor_start, anchor_end, anchor_id))
            anchor_counter += 1

        objects.append(obj)
        current_size += len(obj_bytes)
        boundaries.append(current_size)
        current_size += 2  # For ",\n"

    # Build final JSON array
    content = "[\n" + ",\n".join(objects) + "\n]"
    content_bytes = content.encode("utf-8")

    if len(content_bytes) > size_bytes:
        content_bytes = content_bytes[:size_bytes]
        # Ensure valid JSON by finding last complete object
        # This is a simplification - real impl would be more careful
        boundaries = [b for b in boundaries if b < size_bytes]
        anchors = [(s, e, aid) for s, e, aid in anchors if e < size_bytes]

    # Adjust boundaries for the opening "[\n"
    boundaries = [b + 2 for b in boundaries]

    return SyntheticDocument(
        doc_id=doc_id,
        content=content_bytes,
        domain="json",
        boundary_offsets=boundaries,
        planted_anchors=anchors,
    )


def generate_log_document(seed: int, size_bytes: int) -> SyntheticDocument:
    """Generate a log file with timestamped entries.

    Args:
        seed: Random seed for reproducibility.
        size_bytes: Target size in bytes.

    Returns:
        SyntheticDocument with log entry boundaries marked.
    """
    rng = random.Random(seed)
    doc_id = f"logs_{seed:08x}"

    lines = []
    boundaries = []
    anchors = []
    current_size = 0
    timestamp = rng.randint(0, 86400000)  # Random start within a day (in ms)
    anchor_counter = 0

    while current_size < size_bytes:
        line, timestamp = _generate_log_line(rng, timestamp)
        line_bytes = line.encode("utf-8")

        # Plant anchor on interesting log lines (ERROR/FATAL)
        if rng.random() < 0.15:
            anchor_start = current_size + 30  # After timestamp
            anchor_end = current_size + len(line_bytes) - 1
            anchor_id = f"{doc_id}_anchor_{anchor_counter:04d}"
            anchors.append((anchor_start, anchor_end, anchor_id))
            anchor_counter += 1

        lines.append(line)
        current_size += len(line_bytes)
        boundaries.append(current_size)

    content = "".join(lines)
    content_bytes = content.encode("utf-8")

    if len(content_bytes) > size_bytes:
        content_bytes = content_bytes[:size_bytes]
        boundaries = [b for b in boundaries if b < size_bytes]
        anchors = [(s, e, aid) for s, e, aid in anchors if e < size_bytes]

    return SyntheticDocument(
        doc_id=doc_id,
        content=content_bytes,
        domain="logs",
        boundary_offsets=boundaries,
        planted_anchors=anchors,
    )


def generate_corpus(
    seed: int,
    num_docs: int,
    domains: list[str],
    size_range: tuple[int, int],
) -> list[SyntheticDocument]:
    """Generate a corpus of synthetic documents.

    Args:
        seed: Random seed for reproducibility.
        num_docs: Number of documents to generate.
        domains: List of domains to sample from (text/code/json/logs).
        size_range: (min_bytes, max_bytes) tuple for document sizes.

    Returns:
        List of SyntheticDocument instances.
    """
    rng = random.Random(seed)
    min_size, max_size = size_range

    generators = {
        "text": generate_text_document,
        "code": generate_code_document,
        "json": generate_json_document,
        "logs": generate_log_document,
    }

    # Validate domains
    valid_domains = [d for d in domains if d in generators]
    if not valid_domains:
        raise ValueError(f"No valid domains provided. Must be from: {list(generators.keys())}")

    documents = []
    for i in range(num_docs):
        # Pick domain and size
        domain = rng.choice(valid_domains)
        doc_size = rng.randint(min_size, max_size)

        # Generate unique seed for this document
        doc_seed = seed + i * 1000 + hash(domain) % 1000

        doc = generators[domain](doc_seed, doc_size)
        documents.append(doc)

    return documents
