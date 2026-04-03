import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from groq import Groq

_FALLBACK_PHRASE = (
    "Miala tsiny, mbola tsy manana ny valiny marina momba izany aho amin'izao fotoana izao. "
    "Afaka manampy anao amin'ny [Topic 1] na [Topic 2] ve aho?"
)


class LlamaClient:
    def __init__(self) -> None:
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.model_id = "llama-3.3-70b-versatile"
        self.drills_path = Path("data/imex_drills.json")
        self.drills = self._load_drills()

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

    def _build_system_instruction(self, has_few_shot: bool) -> str:
        similarity_rule = (
            "If the current user message is not 80% similar in intent and topic to the scenarios shown in the "
            "few-shot user/assistant turns that follow this instruction, you MUST reply with only the fallback phrase—verbatim, "
            "in the user's language context. Do not hallucinate. Do not invent facts, numbers, or contacts.\n\n"
            if has_few_shot
            else (
                "No few-shot examples are loaded. You MUST reply with only the fallback phrase below—verbatim. "
                "Do not hallucinate.\n\n"
            )
        )
        reasoning = (
            "Reasoning (think through this internally; do not print these steps in your reply): "
            "First, identify which drill scenario best matches the latest user message. "
            "If none matches well enough, trigger the fallback. "
            "If one matches, answer in the same tone as the assistant in that drill and preserve "
            "specific logistics data (contacts, durations, locations, services) exactly as implied by the examples.\n\n"
            if has_few_shot
            else ""
        )
        knowledge_scope = (
            "Do not use external websites, assumptions, or general world knowledge beyond what the examples imply.\n\n"
            if has_few_shot
            else "Do not use external websites, assumptions, or general world knowledge.\n\n"
        )
        return (
            "You are the IMEX Digital Engine. Speak ONLY using these examples.\n\n"
            f"{reasoning}"
            f"{similarity_rule}"
            f"{knowledge_scope}"
            f"Fallback phrase (use verbatim when no adequate drill match, or when no examples are available):\n{_FALLBACK_PHRASE}\n\n"
            "Keep user-facing replies concise and actionable. Output only the reply the user should read."
        )

    def _drill_messages_for_api(self, drill: Dict[str, Any]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        raw = drill.get("messages", [])
        if not isinstance(raw, list):
            return out
        for msg in raw:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            content = str(msg.get("content", "")).strip()
            if content:
                out.append({"role": role, "content": content})
        return out

    def _build_chat_messages(self, user_message: str, top_k: int = 3) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        selected = self._select_relevant_drills(user_message, top_k=top_k)
        few_shot: List[Dict[str, str]] = []
        for drill in selected:
            few_shot.extend(self._drill_messages_for_api(drill))
        has_few_shot = len(few_shot) > 0
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._build_system_instruction(has_few_shot)},
            *few_shot,
            {"role": "user", "content": user_message},
        ]
        return messages, selected

    def get_llama_response(self, prompt: str) -> str:
        if not self.api_key or self.client is None:
            return (
                "The IMEX assistant is temporarily unavailable because the AI key is "
                "not configured."
            )

        try:
            chat_messages, selected_drills = self._build_chat_messages(prompt)
            few_shot_turns = max(0, len(chat_messages) - 2)
            drill_summaries = [
                {
                    "id": str(d.get("id", "?")),
                    "language": str(d.get("language", "")),
                    "description": (str(d.get("description", ""))[:160] + "…")
                    if len(str(d.get("description", ""))) > 160
                    else str(d.get("description", "")),
                }
                for d in selected_drills
            ]
            print(
                "[IMEX Brain] drills selected for Groq:",
                drill_summaries,
                "| few_shot_turns:",
                few_shot_turns,
                "| total_api_messages:",
                len(chat_messages),
            )
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=chat_messages,
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
