import openai
MODEL             = "gpt-4o"
MAX_TOKENS        = 2048
TEMPERATURE       = 0.2
MAX_CONTEXT_CHARS = 12_000


_SYSTEM_PROMPT = """You are an expert software engineering assistant.
You answer questions about a codebase using only the context provided.
When referencing code, always specify the file path and function/class name.
If the context does not contain enough information to answer confidently,
say so rather than guessing. Never fabricate function names, file paths,
or behaviours that are not present in the context."""


class LLMService:

    def __init__(self):
        self.client = openai.OpenAI()

    def generate_response(
        self,
        query:    str,
        context:  list[dict],
    ) -> str:
        context_block = self._build_context_block(context)
        prompt        = self._build_prompt(query, context_block)

        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content.strip()

        except openai.RateLimitError as e:
            raise RuntimeError("OpenAI rate limit reached — retry after a moment.") from e
        except openai.APIStatusError as e:
            raise RuntimeError(f"OpenAI API error {e.status_code}: {e.message}") from e

    def generate_response_streamed(
        self,
        query:   str,
        context: list[dict],
    ):
        context_block = self._build_context_block(context)
        prompt        = self._build_prompt(query, context_block)

        try:
            with self.client.chat.completions.stream(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except openai.RateLimitError as e:
            raise RuntimeError("OpenAI rate limit reached — retry after a moment.") from e
        except openai.APIStatusError as e:
            raise RuntimeError(f"OpenAI API error {e.status_code}: {e.message}") from e


    def _build_context_block(self, context: list[dict]) -> str:
        if not context:
            return "No relevant context was found in the repository."

        sections: list[str] = []
        total_chars = 0

        for i, result in enumerate(context, start=1):
            meta     = result.get("metadata", {})
            score    = result.get("score", 0.0)
            text     = result.get("text", "")

            source = meta.get("file_path", "unknown file")
            symbol = meta.get("name")
            lang   = meta.get("language", "")
            if symbol:
                source = f"{source} → {symbol}"

            header  = f"[{i}] {source} (similarity: {score:.2f})"
            fenced  = f"```{lang}\n{text}\n```"
            section = f"{header}\n{fenced}"

            if total_chars + len(section) > MAX_CONTEXT_CHARS:
                break

            sections.append(section)
            total_chars += len(section)

        return "\n\n".join(sections)

    @staticmethod
    def _build_prompt(query: str, context_block: str) -> str:
        return (
            f"## Retrieved context\n\n"
            f"{context_block}\n\n"
            f"## Question\n\n"
            f"{query}"
        )