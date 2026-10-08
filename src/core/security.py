"""Security safeguards: Prompt injection defense, PII masking, and input sanitation."""

from enum import Enum
import re
from typing import List, Tuple
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    CRITICAL = "CRITICAL"


class SecurityCheckResult(BaseModel):
    is_safe: bool = True
    risk_level: RiskLevel = RiskLevel.SAFE
    sanitized_text: str = ""
    detected_patterns: List[str] = Field(default_factory=list)
    redacted_pii_count: int = 0


class SecurityGuard:
    """Multi-stage security firewall protecting RAG ingestion and retrieval query endpoints."""

    INJECTION_PATTERNS = [
        (re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions\b", re.I), "OVERRIDE_PREVIOUS_INSTRUCTIONS"),
        (re.compile(r"\breveal\s+(?:your\s+)?(?:system\s+)?prompt\b", re.I), "PROMPT_LEAK_ATTEMPT"),
        (re.compile(r"\byou\s+are\s+now\s+(?:in\s+)?(?:dan|god|developer)\s+mode\b", re.I), "JAILBREAK_ROLEPLAY"),
        (re.compile(r"\bdisregard\s+(?:all\s+)?(?:safety|guidelines|guardrails)\b", re.I), "DISREGARD_GUARDRAILS"),
        (re.compile(r"\bprint\s+raw\s+database\s+credentials\b", re.I), "CREDENTIAL_EXFILTRATION"),
        (re.compile(r"\b(?:drop\s+table|union\s+select|;\s*delete\s+from)\b", re.I), "SQL_INJECTION_TOKEN"),
        (re.compile(r"\bзабудь\s+(?:все\s+)?инструкции\b", re.I), "RU_OVERRIDE_INSTRUCTIONS"),
        (re.compile(r"\bпокажи\s+(?:свой\s+)?системный\s+промпт\b", re.I), "RU_PROMPT_LEAK"),
    ]

    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Removes null bytes, terminal escape sequences, and dangerous unicode characters."""
        if not text:
            return ""
        cleaned = "".join(ch for ch in text if ch.isprintable() or ch in ("\n", "\t", "\r", " "))
        return cleaned.strip()

    @classmethod
    def redact_pii(cls, text: str) -> Tuple[str, int]:
        """Redacts sensitive PII like emails, phones, and SSNs from logs/prompts."""
        redacted = text
        count = 0
        for pattern, replacement in [
            (cls.EMAIL_PATTERN, "[EMAIL_REDACTED]"),
            (cls.PHONE_PATTERN, "[PHONE_REDACTED]"),
            (cls.SSN_PATTERN, "[SSN_REDACTED]"),
        ]:
            matches = pattern.findall(redacted)
            count += len(matches)
            redacted = pattern.sub(replacement, redacted)
        return redacted, count

    @classmethod
    def inspect_query(cls, query: str) -> SecurityCheckResult:
        """Evaluates whether an incoming user query is adversarial."""
        cleaned = cls.sanitize(query)
        detected = []

        for pattern, label in cls.INJECTION_PATTERNS:
            if pattern.search(cleaned):
                detected.append(label)

        sanitized, pii_count = cls.redact_pii(cleaned)

        if detected:
            return SecurityCheckResult(
                is_safe=False,
                risk_level=RiskLevel.CRITICAL if len(detected) > 1 else RiskLevel.MEDIUM,
                sanitized_text=sanitized,
                detected_patterns=detected,
                redacted_pii_count=pii_count,
            )

        return SecurityCheckResult(
            is_safe=True,
            risk_level=RiskLevel.SAFE,
            sanitized_text=sanitized,
            detected_patterns=[],
            redacted_pii_count=pii_count,
        )
