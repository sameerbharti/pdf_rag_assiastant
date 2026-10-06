import re


# ------------------------------------------------------------
# REGEX PATTERNS
# ------------------------------------------------------------

NUMBERED_STEP_PATTERN = re.compile(
    r"^\d{1,2}\.\s+"
)

SECTION_HEADING_PATTERN = re.compile(
    r"Step\s+\d+\s*[—–-]\s+[^.\n]+?(?=\s+(?:\d{1,2}\.\s+)|$)",
    re.IGNORECASE
)


# ------------------------------------------------------------
# TEXT NORMALIZATION
# ------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """
    Normalize PDF-extracted text while preserving
    meaningful line boundaries.
    """

    text = re.sub(
        r"[ \t]+\n",
        "\n",
        text
    )

    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text
    )

    lines = []

    for line in text.splitlines():

        line = re.sub(
            r"[ \t]+",
            " ",
            line
        ).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


# ------------------------------------------------------------
# SECTION-AWARE PREPROCESSING
# ------------------------------------------------------------

def _split_embedded_sections(text: str) -> str:
    """
    Ensure section headings such as:

        Step 1 — ...
        Step 2 — ...
        Step 3 — ...

    start on their own line, even when PDF extraction
    placed them in the middle of another line.
    """

    # Insert a newline before every Step heading that is
    # not already at the beginning of a line.
    text = re.sub(
        r"(?<!^)(?<!\n)\s+(Step\s+\d+\s*[—–-]\s+)",
        r"\n\1",
        text,
        flags=re.IGNORECASE
    )

    return text


# ------------------------------------------------------------
# UNIT SPLITTING
# ------------------------------------------------------------

def _split_into_units(text: str) -> list[str]:
    """
    Split document text into logical units.

    Numbered instructions are kept as individual units.
    Section headings are kept as their own units.
    """

    text = _normalize_text(text)

    if not text:
        return []

    # Fix PDF extraction where a new Step heading appears
    # in the middle of another extracted line.
    text = _split_embedded_sections(text)

    lines = text.splitlines()

    units = []
    current = []

    for line in lines:

        # ----------------------------------------------------
        # Section heading
        # ----------------------------------------------------

        if SECTION_HEADING_PATTERN.match(line):

            if current:
                units.append(
                    " ".join(current).strip()
                )
                current = []

            units.append(
                line.strip()
            )

            continue

        # ----------------------------------------------------
        # Numbered instruction
        # ----------------------------------------------------

        if NUMBERED_STEP_PATTERN.match(line):

            if current:
                units.append(
                    " ".join(current).strip()
                )
                current = []

            current.append(line)

        else:
            current.append(line)

    # --------------------------------------------------------
    # Remaining content
    # --------------------------------------------------------

    if current:
        units.append(
            " ".join(current).strip()
        )

    return [
        unit
        for unit in units
        if unit
    ]


# ------------------------------------------------------------
# CHUNKING
# ------------------------------------------------------------

def chunk_text(
    pages: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> list[dict]:
    """
    Split PDF pages into semantically meaningful chunks.

    Features:
        - Keeps numbered instructions intact.
        - Detects section headings.
        - Prevents different Step sections from being
          combined into the same chunk.
        - Preserves page metadata.
        - Preserves source metadata.
        - Uses limited contextual overlap.
        - Avoids carrying numbered instructions across
          chunk boundaries.
    """

    if not pages:
        return []

    chunks = []
    chunk_id = 0

    for page in pages:

        text = page["text"]
        page_number = page["page"]
        source = page["source"]

        units = _split_into_units(text)

        current_units = []
        current_length = 0
        current_section = None

        for unit in units:

            # ------------------------------------------------
            # Detect section heading
            # ------------------------------------------------

            is_section = bool(
                SECTION_HEADING_PATTERN.match(unit)
            )

            if is_section:

                new_section = unit

                # A new section always starts a new chunk.
                if current_units:

                    chunk_value = "\n".join(
                        current_units
                    ).strip()

                    chunks.append(
                        {
                            "text": chunk_value,
                            "page": page_number,
                            "source": source,
                            "chunk_id": chunk_id
                        }
                    )

                    chunk_id += 1

                    current_units = []
                    current_length = 0

                current_section = new_section

            # ------------------------------------------------
            # Normal size-based boundary
            # ------------------------------------------------

            unit_length = len(unit)

            if (
                current_units
                and current_length + unit_length > chunk_size
            ):

                chunk_value = "\n".join(
                    current_units
                ).strip()

                chunks.append(
                    {
                        "text": chunk_value,
                        "page": page_number,
                        "source": source,
                        "chunk_id": chunk_id
                    }
                )

                chunk_id += 1

                # ------------------------------------------------
                # Overlap
                #
                # Do not carry numbered instructions or section
                # headings into the next chunk.
                # ------------------------------------------------

                overlap_units = []
                overlap_length = 0

                for previous_unit in reversed(
                    current_units
                ):

                    is_numbered_step = bool(
                        NUMBERED_STEP_PATTERN.match(
                            previous_unit
                        )
                    )

                    is_previous_section = bool(
                        SECTION_HEADING_PATTERN.match(
                            previous_unit
                        )
                    )

                    if (
                        is_numbered_step
                        or is_previous_section
                    ):
                        continue

                    if (
                        overlap_length
                        + len(previous_unit)
                        > chunk_overlap
                    ):
                        break

                    overlap_units.insert(
                        0,
                        previous_unit
                    )

                    overlap_length += len(
                        previous_unit
                    )

                current_units = overlap_units

                current_length = sum(
                    len(item)
                    for item in current_units
                )

            # ------------------------------------------------
            # Add current unit
            # ------------------------------------------------

            current_units.append(unit)

            current_length += unit_length

        # ----------------------------------------------------
        # Final chunk
        # ----------------------------------------------------

        if current_units:

            chunk_value = "\n".join(
                current_units
            ).strip()

            chunks.append(
                {
                    "text": chunk_value,
                    "page": page_number,
                    "source": source,
                    "chunk_id": chunk_id
                }
            )

            chunk_id += 1

    return chunks