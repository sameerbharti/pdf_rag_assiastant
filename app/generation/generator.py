import os

from dotenv import load_dotenv
from google import genai

from app.generation.extractive_fallback import (
    extract_relevant_sentences,
    extract_responsibilities
)


load_dotenv()


MODEL_NAME = "gemini-3.8-flash"


class Generator:
    def __init__(
        self,
        model_name: str = MODEL_NAME
    ):
        self.model_name = model_name

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set in the environment."
            )

        print(
            f"Initializing Gemini generation model: "
            f"{model_name}"
        )

        self.client = genai.Client(
            api_key=api_key
        )

        print(
            "Gemini generation client initialized."
        )

    def _generate_llm_answer(
        self,
        query: str,
        context: str
    ) -> str:

        prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context.

Rules:

- Every factual statement in the answer must be directly
  supported by the document context.

- Do not use outside knowledge.

- Do not invent, assume, infer, or add information that is
  not explicitly stated in the context.

- Do not add general advice, typical timelines,
  troubleshooting suggestions, or recommendations.

- If the context contains a numbered procedure that directly
  answers the question, include ALL numbered steps belonging
  to that procedure in their original numerical order.

- Do not skip a procedure step merely because that individual
  step has little lexical overlap with the user's question.

- For example, if steps 7, 8 and 9 together describe how to
  perform the requested task, include steps 7, 8 and 9.

- Do not include numbered steps from a different procedure
  or section.

- Preserve the exact order of procedural steps.

- Preserve important instructions, names, URLs, and
  document-specific details.

- Keep the answer concise and focused strictly on the user's question.

- Identify the specific information category requested by the user.
  For example:
  - "responsibilities" means return responsibilities only.
  - "skills" means return skills only.
  - "qualifications" means return qualifications only.
  - "requirements" means return requirements only.
  - "steps" or "how do I" means return the relevant procedure steps only.

- Do not include related information from other categories unless
  the user explicitly asks for it.

- When the context contains the exact heading "Your key responsibilities"
  and the user asks about responsibilities, use only the text beginning
  at "Your key responsibilities" and the responsibility items that
  follow it. Do not include the preceding background or description
  text from the same chunk.

- For responsibility questions, return each responsibility as a
  separate bullet point and preserve the wording and order from
  the document.

- If the question asks for a list, present the answer as a clear
  bullet list rather than reproducing unrelated surrounding text.

- If the context does not contain enough information to
  answer the question, say:

  "I could not find enough information in the provided
  documents to answer this question."

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        if not response.text:
            return ""

        return response.text.strip()

    def generate(
        self,
        query: str,
        context: str,
        results: list[dict] | None = None
    ) -> str:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not context.strip():
            return (
                "I could not find relevant information "
                "in the provided documents."
            )

        # --------------------------------------------------
        # Deterministic extraction for responsibility queries
        # --------------------------------------------------

        if results and "responsib" in query.lower():

            responsibility_answer = extract_responsibilities(
                query=query,
                results=results
            )

            if responsibility_answer:
                return responsibility_answer

        # --------------------------------------------------
        # Gemini generation
        # --------------------------------------------------

        try:

            answer = self._generate_llm_answer(
                query=query,
                context=context
            )

        except Exception as e:

            print(
                f"Gemini generation failed: {e}"
            )

            if results:

                fallback_answer = (
                    extract_relevant_sentences(
                        query=query,
                        results=results,
                        max_sentences=5
                    )
                )

                if fallback_answer:
                    return fallback_answer

            return (
                "I could not generate an answer from "
                "the provided documents."
            )

        # --------------------------------------------------
        # Detect obviously incomplete generation
        # --------------------------------------------------

        answer_words = len(
            answer.split()
        )

        if results and answer_words < 8:

            fallback_answer = (
                extract_relevant_sentences(
                    query=query,
                    results=results,
                    max_sentences=5
                )
            )

            if fallback_answer:
                return fallback_answer

        return answer