import os

import chromadb
from groq import Groq


class LlamaClient:
    def __init__(self) -> None:
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.model_id = "llama-3.3-70b-versatile"
        self.system_instruction = (
            "You are the Official Digital Assistant for Groupe IMEX MCE MBT.\n\n"
            "Core Knowledge Base:\n"
            "- Identity: We are a global freight forwarding and sourcing company present on 4 continents: "
            "Madagascar, France, China, Canada, Indonesia, and Thailand.\n"
            "- Sourcing: We help clients find the best suppliers and products globally.\n"
            "- Groupage: Our specialty. We consolidate shipments to reduce costs - plus vous groupez, plus vous economisez.\n"
            "- Transit & Logistics: We handle all customs clearance (dedouanement) and administrative paperwork.\n"
            "- Travel: We offer Visa & Billetterie services for business and personal travel.\n"
            "- Domestic: We ensure delivery (Livraison) across all of Madagascar.\n\n"
            "Contact Details:\n"
            "- Antananarivo: Ambohimanambola (near Hintsy Hotel). Phones: +261 34 05 828 71 / +261 32 62 269 37.\n"
            "Constraints and Tone:\n"
            "- Style: Professional, welcoming, and Agile.\n"
            "- Always use the catchphrase: Nous sommes la pour vous faciliter la vie.\n"
            "- Fallback for specific container quotes: Politely ask for the user's email and Port de depart so a human expert can follow up.\n"
            "- Multilingual: Detect Malagasy, French, or English, and respond in the same language."
        )
        self.collection_name = "imex_messages"
        self.chroma_client = chromadb.PersistentClient(
            path=os.environ.get("CHROMA_PATH", "data/chroma")
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )
        self.website_collection = self.chroma_client.get_or_create_collection(
            name="imex_knowledge"
        )

    def _is_factual_service_question(self, prompt: str) -> bool:
        text = prompt.lower()
        keywords = [
            "service",
            "services",
            "freight",
            "groupage",
            "customs",
            "dedouanement",
            "visa",
            "billetterie",
            "delivery",
            "livraison",
            "sourcing",
            "quote",
            "price",
            "rate",
        ]
        return any(keyword in text for keyword in keywords)

    def _build_augmented_prompt(self, prompt: str) -> str:
        try:
            context_lines = []

            if self._is_factual_service_question(prompt):
                website_results = self.website_collection.query(
                    query_texts=[prompt],
                    n_results=2,
                    where={"source": "website"},
                )
                website_docs = website_results.get("documents", [[]])
                top_docs = (
                    website_docs[0]
                    if website_docs and isinstance(website_docs[0], list)
                    else []
                )
                for doc in top_docs:
                    if isinstance(doc, str) and doc.strip():
                        context_lines.append(
                            f"Context: In a similar past case, the agent said: {doc.strip()}"
                        )

            history_results = self.collection.query(query_texts=[prompt], n_results=2)
            metadatas = history_results.get("metadatas", [[]])
            top_items = metadatas[0] if metadatas and isinstance(metadatas[0], list) else []
            for item in top_items:
                if isinstance(item, dict):
                    response = item.get("answer")
                    if isinstance(response, str) and response.strip():
                        context_lines.append(
                            f"Context: In a similar past case, the agent said: {response.strip()}"
                        )

            if not context_lines:
                return prompt

            return f"{'\n'.join(context_lines)}\n\nUser request: {prompt}"
        except Exception:
            return prompt

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
