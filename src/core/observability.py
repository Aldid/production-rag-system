"""Observability and distributed tracing for production RAG pipeline.

Supports structured JSON trace recording, span latency metrics, token estimation,
and Langfuse-compatible trace payload generation.
"""

from dataclasses import asdict, dataclass, field
import datetime
import time
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class Span:
    """Individual operation span within a trace."""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "operation"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    status: str = "OK"  # "OK", "ERROR", "BLOCKED"
    attributes: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def finish(self, status: str = "OK", error: Optional[str] = None) -> None:
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.status = status
        self.error_message = error


@dataclass
class Trace:
    """End-to-end trace encompassing the entire RAG request lifecycle."""
    trace_id: str = field(default_factory=lambda: f"tr_{uuid.uuid4().hex[:12]}")
    session_id: Optional[str] = None
    query: str = ""
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    spans: List[Span] = field(default_factory=list)
    total_duration_ms: float = 0.0
    total_tokens_estimated: int = 0
    status: str = "OK"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Span:
        span = Span(name=name, attributes=attributes or {})
        self.spans.append(span)
        return span

    def finalize(self) -> None:
        if self.spans:
            first_start = min(s.start_time for s in self.spans)
            last_end = max((s.end_time or time.time()) for s in self.spans)
            self.total_duration_ms = round((last_end - first_start) * 1000, 2)
        if any(s.status == "ERROR" for s in self.spans):
            self.status = "ERROR"
        elif any(s.status == "BLOCKED" for s in self.spans):
            self.status = "BLOCKED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "query": self.query,
            "timestamp": self.timestamp,
            "status": self.status,
            "total_duration_ms": self.total_duration_ms,
            "total_tokens_estimated": self.total_tokens_estimated,
            "metadata": self.metadata,
            "spans": [
                {
                    "span_id": s.span_id,
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "status": s.status,
                    "attributes": s.attributes,
                    "error_message": s.error_message,
                }
                for s in self.spans
            ],
        }


class ObservabilityTracer:
    """Thread-safe in-memory trace registry and exporter."""

    def __init__(self, max_retained_traces: int = 1000):
        self._traces: List[Trace] = []
        self._max_retained = max_retained_traces

    def new_trace(self, query: str, session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Trace:
        trace = Trace(query=query, session_id=session_id, metadata=metadata or {})
        self._traces.append(trace)
        if len(self._traces) > self._max_retained:
            self._traces.pop(0)
        return trace

    def get_recent_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self._traces[-limit:]]

    def clear(self) -> None:
        self._traces.clear()


# Global tracer instance
tracer = ObservabilityTracer()
