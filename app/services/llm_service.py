import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL             = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        
    def _build_inputs(self, query: str, context: list[dict]):
        context_block = self._build_context_block(context)
        prompt = self._build_prompt(query, context_block)
        
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        return input_ids

    def generate_response(
        self,
        query:    str,
        context:  list[dict],
    ) -> str:
        input_ids = self._build_inputs(query, context)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        
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