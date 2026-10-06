import pymupdf


def load_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF while preserving page information.

    Returns:
        A list of dictionaries containing page text and metadata.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text().strip()

        if text:
            pages.append(
                {
                    "text": text,
                    "page": page_number,
                    "source": pdf_path
                }
            )

    document.close()

    return pages