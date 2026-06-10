"""
LocalQAHandler — self-contained @-mention answer handler.

Used by POST /api/v1/agents/{agent_id}/answer.

This handler does NOT call aegean-consensus. It builds a role-specific
system prompt from the public agent registry and asks the configured LLM
(Praka aggregator by default) directly. Keeps the @-mention path simple
and decoupled from the consensus engine.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QAResponse:
    agent_id: str
    question: str
    answer: str
    confidence: float = 0.7
    room_id: Optional[str] = None
    model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "room_id": self.room_id,
            "model": self.model,
            "metadata": self.metadata,
        }


_FALLBACK_QA_SYSTEM = (
    "You are {name} ({id}), a specialist in a World Cup 2026 prediction panel.\n"
    "Role: {description}\n\n"
    "Rules:\n"
    " - {lang_rule}\n"
    " - Answer in 2-4 sentences.\n"
    " - Stay strictly within your specialty.\n"
    " - Do NOT invent statistics. If you don't have the number, say so.\n"
)


def _build_system_prompt(agent: Dict[str, Any], lang: str = "en") -> str:
    """
    Render the QA system prompt from prompts/templates.yaml, then append
    the product-tuned global directive (if set via the admin endpoint).
    """
    from aegeanbench.sports.prompts.loader import get_template
    from aegeanbench.sports.prompts.runtime_store import get_current_prompt
    from aegeanbench.sports.lang import lang_directive
    template = get_template("qa.system_en", default=_FALLBACK_QA_SYSTEM)
    base = template.format(
        name=agent.get("name", ""),
        id=agent.get("id", ""),
        description=agent.get("description", ""),
        lang_rule=lang_directive(lang),
    )
    addendum = get_current_prompt().strip()
    if addendum:
        base = base.rstrip() + "\n\n## GLOBAL DIRECTIVE (product-tuned)\n" + addendum
    return base


def _build_user_prompt(
    question: str,
    match_context: Optional[str],
    recent_messages: Optional[List[Dict[str, Any]]],
    user_name: Optional[str],
) -> str:
    parts: List[str] = []
    if match_context:
        parts.append(f"Match context:\n{match_context}\n")
    if recent_messages:
        rendered = "\n".join(
            f"  {m.get('user_name', 'user')}: {m.get('text', '')}"
            for m in recent_messages[-6:]
        )
        parts.append(f"Recent chat:\n{rendered}\n")
    who = f"{user_name}: " if user_name else ""
    parts.append(f"Question from {who}{question}")
    return "\n".join(parts)


class LocalQAHandler:
    """
    Async @-mention QA. Calls an OpenAI-compatible endpoint (Praka by
    default) using OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL.

    If no API key is configured we return a deterministic fallback so the
    endpoint stays responsive in dev/test environments.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://praka.ai/v1")).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.timeout = timeout

    def _lookup_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        from aegeanbench.sports.reporter.agent_registry import SPORTS_AGENTS_PUBLIC
        for a in SPORTS_AGENTS_PUBLIC:
            if a["id"] == agent_id:
                return a
        return None

    async def answer(
        self,
        agent_id: str,
        question: str,
        match_context: Optional[str] = None,
        recent_messages: Optional[List[Dict[str, Any]]] = None,
        user_name: Optional[str] = None,
        room_id: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> QAResponse:
        agent = self._lookup_agent(agent_id)
        if agent is None:
            return QAResponse(
                agent_id=agent_id, question=question, room_id=room_id,
                answer=f"No agent named '{agent_id}' is available.",
                confidence=0.0, metadata={"error": "unknown_agent"},
            )

        # Auto-detect language from the question itself when the caller
        # didn't pin it explicitly. Recent chat messages serve as a weak
        # secondary signal.
        if not lang:
            from aegeanbench.sports.lang import detect_from_signals
            secondary = [
                m.get("text", "") for m in (recent_messages or [])
            ]
            lang = detect_from_signals(question, secondary)

        system = _build_system_prompt(agent, lang=lang)
        user = _build_user_prompt(question, match_context, recent_messages, user_name)

        if not self.api_key:
            return QAResponse(
                agent_id=agent_id, question=question, room_id=room_id,
                answer=(
                    f"({agent['name']} fallback) Your question was: \"{question}\". "
                    f"LLM is not configured — set OPENAI_API_KEY to enable real answers."
                ),
                confidence=0.3, model=None,
                metadata={"fallback": True, "role": agent_id},
            )

        try:
            # Use the openai SDK (same path aegean-consensus uses for Praka).
            # Praka rejects some raw-httpx variants with 503 even when the
            # SDK path works — likely due to header/UA expectations on
            # their gateway. Keep this aligned with the consensus client.
            import openai
            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            resp = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                # Empty content from the provider — log enough to diagnose
                # (finish_reason often says 'length' or 'content_filter').
                finish = getattr(resp.choices[0], "finish_reason", None)
                logger.warning(
                    "LLM returned empty content for %s (finish_reason=%s, "
                    "prompt_chars=%d)", agent_id, finish, len(system) + len(user),
                )
                text = (
                    f"({agent['name']}) The model returned no text "
                    f"(finish_reason={finish}). Try asking a more specific "
                    f"question or pick a different agent."
                )
            usage = getattr(resp, "usage", None)
            return QAResponse(
                agent_id=agent_id, question=question, room_id=room_id,
                answer=text,
                confidence=0.75,
                model=getattr(resp, "model", None) or self.model,
                metadata={
                    "role": agent_id,
                    "lang": lang,
                    "tokens": getattr(usage, "total_tokens", None) if usage else None,
                },
            )
        except Exception as e:
            logger.warning("LocalQAHandler LLM call failed (%s); returning fallback", e)
            return QAResponse(
                agent_id=agent_id, question=question, room_id=room_id,
                answer=(
                    f"({agent['name']}) I couldn't reach the LLM just now. "
                    f"Try again in a moment."
                ),
                confidence=0.3,
                metadata={"error": str(e), "role": agent_id},
            )
