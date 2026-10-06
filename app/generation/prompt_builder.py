def build_context(results: list[dict]) -> str:
    """
    Convert retrieved chunks into structured context
    for the generation model.
    """

    if not results:
        return ""

    context_parts = []

    for i, result in enumerate(results, start=1):

        source = result["source"]

        # Keep only the filename for cleaner citations
        filename = source.replace("\\", "/").split("/")[-1]

        context_parts.append(
            f"SOURCE {i}\n"
            f"File: {filename}\n"
            f"Page: {result['page']}\n"
            f"Chunk: {result['chunk_id']}\n\n"
            f"{result['text']}"
        )

    return "\n\n" + "\n\n".join(context_parts)