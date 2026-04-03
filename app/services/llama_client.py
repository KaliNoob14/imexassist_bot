import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from groq import Groq


class LlamaClient:
    def __init__(self) -> None:
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.model_id = "llama-3.3-70b-versatile"
        self.drills_path = Path("data/imex_drills.json")
        self.drills = self._load_drills()
        self.system_instruction = (
            "You are the IMEX Digital Engine.\n"
            "You ONLY answer based on the provided Drills (Examples) and IMEX core identity.\n"
            "IMEX core identity:\n"
            "- Freight forwarding and sourcing company.\n"
            "- Services include groupage, customs clearance, logistics support, and transport assistance.\n"
            "Strict rules:\n"
            "1) Do NOT use external website data, assumptions, or general world knowledge.\n"
            "2) Prioritize the Expert response style and format shown in the drills.\n"
            "3) If the question is not covered by drills or core identity, reply politely in the user's language (Malagasy/French):\n"
            "\"Miala tsiny, mbola tsy manana ny valiny marina momba izany aho amin'izao fotoana izao. "
            "Afaka manampy anao amin'ny [Topic 1] na [Topic 2] ve aho?\"\n"
            "4) Keep responses concise and actionable."
        )

    def _load_drills(self) -> List[Dict[str, Any]]:
        try:
            if not self.drills_path.exists():
                return []
            with self.drills_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
            return []
        except Exception as exc:
            print(f"Drills load error: {type(exc).__name__}: {exc}")
            return []

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r"[a-zA-Z0-9_]{3,}", text.lower()))

    def _drill_text(self, drill: Dict[str, Any]) -> str:
        parts = [str(drill.get("description", ""))]
        for msg in drill.get("messages", []) if isinstance(drill.get("messages"), list) else []:
            if isinstance(msg, dict):
                parts.append(str(msg.get("content", "")))
        return " ".join(parts).strip()

    def _select_relevant_drills(self, prompt: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.drills:
            return []
        prompt_tokens = self._tokenize(prompt)
        if not prompt_tokens:
            return self.drills[:top_k]

        scored: List[tuple[int, Dict[str, Any]]] = []
        for drill in self.drills:
            drill_tokens = self._tokenize(self._drill_text(drill))
            score = len(prompt_tokens & drill_tokens)
            scored.append((score, drill))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [item[1] for item in scored[:top_k] if item[0] > 0]
        return selected if selected else self.drills[:top_k]

    def _build_augmented_prompt(self, prompt: str) -> str:
        relevant_drills = self._select_relevant_drills(prompt, top_k=3)
        if not relevant_drills:
            return f"Drills: none available.\n\nUser request: {prompt}"

        drill_blocks: List[str] = []
        for idx, drill in enumerate(relevant_drills, start=1):
            language = drill.get("language", "")
            description = drill.get("description", "")
            messages = drill.get("messages", [])
            expert_examples = []
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        expert_examples.append(str(msg.get("content", "")).strip())
            expert_text = "\n".join(
                f"- Expert example: {example}" for example in expert_examples[:2] if example
            )
            drill_blocks.append(
                f"Drill {idx}\nLanguage: {language}\nDescription: {description}\n{expert_text}".strip()
            )

        drills_context = "\n\n".join(drill_blocks)
        return f"Relevant Drills (Examples):\n{drills_context}\n\nUser request: {prompt}"

    def get_llama_response(self, prompt: str) -> str:
        if not self.api_key or self.client is None:
            return (
                "The IMEX assistant is temporarily unavailable because the AI key is "
                "not configured."
            )

        try:
            augmented_prompt = self._build_augmented_prompt(prompt)
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": augmented_prompt},
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
