import os

from groq import Groq


class LlamaClient:
    def __init__(self) -> None:
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.model_id = "llama-3.3-70b-versatile"
        self.system_instruction = (
            "You are the IMEX Digital Assistant, a professional logistics expert. "
            "You are helpful, concise, and represent the IMEX brand with a touch "
            "of modern efficiency."
        )

    def get_llama_response(self, prompt: str) -> str:
        if not self.api_key or self.client is None:
            return (
                "The IMEX assistant is temporarily unavailable because the AI key is "
                "not configured."
            )

        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            )
            return (completion.choices[0].message.content or "").strip() or (
                "Thanks for your message. How can I help you with IMEX today?"
            )
        except Exception as e:
            error_text = str(e).lower()
            if "rate limit" in error_text or "429" in error_text:
                return (
                    "The IMEX assistant is receiving high traffic right now. "
                    "Please try again in a moment."
                )
            return (
                "The IMEX assistant is currently unavailable. Please try again shortly."
            )
