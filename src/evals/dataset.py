"""Comprehensive evaluation dataset with 100 ground-truth test cases across enterprise engineering domains."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvalCase:
    case_id: str
    query: str
    expected_doc_id: str
    expected_keywords: List[str]
    ground_truth_answer: str
    category: str


# Reference corpus documents used to populate the evaluation index
REFERENCE_CORPUS = [
    {
        "doc_id": "doc_api_gateway",
        "title": "Enterprise API Gateway Architecture & Rate Limiting",
        "content": """# Enterprise API Gateway Architecture

## Rate Limiting & Token Bucket Algorithms
The enterprise API Gateway implements a distributed token bucket rate limiter backed by Redis Cluster.
Each tenant is allocated a burst bucket capacity of 500 tokens with a steady refill rate of 100 tokens per second.
When rate limits are exceeded, the gateway returns HTTP 429 Too Many Requests with headers `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `Retry-After`.

## Mutual TLS and Authentication
All external ingress traffic terminates mTLS at the Envoy proxy edge. Downstream microservices receive JWT bearer tokens signed using RS256 with 2048-bit RSA keys.
Token expiration is strictly capped at 15 minutes, requiring refresh tokens rotated through the OAuth 2.0 Token Exchange grant.

## Circuit Breaking & Fallbacks
Circuit breakers trigger when downstream error rates exceed 50% over a 10-second rolling window with a minimum volume threshold of 20 requests.
When in OPEN state, immediate HTTP 503 Service Unavailable responses are returned to prevent cascading socket pool exhaustion.""",
    },
    {
        "doc_id": "doc_postgres_pgvector",
        "title": "PostgreSQL High Availability & PgVector Indexing Guidelines",
        "content": """# PostgreSQL High Availability & PgVector Indexing

## PgVector Index Optimization
The `pgvector` extension provides HNSW and IVFFlat indexing for dense vector similarity.
For production datasets above 100,000 vectors, HNSW is mandatory with `m = 16` and `ef_construction = 64`.
During search execution, `ef_search` should be tuned to 40 for optimal balance between 98% recall and sub-5ms query latency.
Cosine distance is queried using the `<=>` operator, whereas L2 squared distance uses `<->`.

## Connection Pooling & Pgbouncer
Application pods connect to PostgreSQL through PgBouncer operating in transaction pooling mode.
The maximum pool size per database instance is set to 200 server connections to avoid OS context switching overhead.
Statement timeouts are enforced at 8000ms for OLTP queries and 60000ms for analytical batch jobs.

## WAL Replication & Point-in-Time Recovery
Streaming replication runs asynchronously with warm standby replicas distributed across availability zones.
Continuous archiving uploads WAL segments to S3-compatible object storage every 60 seconds, guaranteeing an RPO of under 1 minute.""",
    },
    {
        "doc_id": "doc_security_guardrails",
        "title": "AI Security Architecture & Prompt Injection Defense",
        "content": """# AI Security Architecture & Prompt Injection Defense

## Multi-Layer Injection Defense
The security perimeter processes all untrusted user prompts through a 3-layer deterministic inspection engine before LLM invocation:
1. Regex pattern matching against known adversarial exploits (e.g., DAN mode, system prompt leakage, instruction override).
2. Structural sanitization stripping non-printable control characters, null bytes, and shell escape sequences.
3. Automated PII scrubbing masking email addresses, phone numbers, and government identification numbers.

## Structured Output Enforcement
LLM responses are never parsed as raw markdown. All generation pipelines enforce strict JSON schemas validated via Pydantic or AJV.
If an LLM payload fails schema validation, the pipeline attempts exactly one self-correction retry before defaulting to a deterministic fallback response.

## Prompt Isolation & System Boundaries
User inputs are encapsulated within strict `<user_query>` XML demarcation boundaries.
System instructions explicitly command models to treat delimiter-enclosed contents as passive textual data rather than executable directives.""",
    },
    {
        "doc_id": "doc_distributed_cache",
        "title": "Distributed Caching Strategy & Cache Invalidation",
        "content": """# Distributed Caching Strategy & Cache Invalidation

## Redis Cluster Topology & Sizing
The caching layer runs on a 6-node Redis 7.2 cluster with 3 primaries and 3 read replicas configured with automatic failover.
Key eviction follows the `volatile-lru` policy with a soft memory ceiling of 80% maximum memory capacity.
All cache keys must use namespaced prefixes e.g. `tenant:{id}:session:{sid}` with deterministic TTLs.

## Cache Invalidation Protocols
To prevent stale reads, data mutations employ the Cache-Aside pattern coupled with asynchronous event-driven invalidation via Kafka.
When a record updates in the primary database, an Outbox table emits an invalidation event that purges corresponding Redis keys within 150ms.

## Thundering Herd & Cache Stampede Mitigation
High-traffic endpoints utilize probabilistic early expiration (XFetch algorithm) with beta = 1.0 to recalculate cache values prior to hard TTL expiration.
Mutual exclusion mutex locks (Redlock algorithm) prevent redundant expensive database queries during transient cache misses.""",
    },
    {
        "doc_id": "doc_playwright_cdp",
        "title": "Headless Browser Automation with Playwright & CDP",
        "content": """# Headless Browser Automation with Playwright & CDP

## Session Persistence & Cookie Management
Playwright automation workers maintain persistent browser contexts storing indexedDB, cookies, and local storage state on disk.
This avoids repeated multi-factor authentication challenges across batch crawler executions.
Context state is serialized to encrypted JSON files and rotated every 24 hours.

## Chrome DevTools Protocol (CDP) Hooking
Direct integration with the Chrome DevTools Protocol allows low-level interception of network events and DOM mutations.
Workers listen to `Network.requestWillBeSent` and `Network.responseReceived` events to monitor telemetry and detect anti-bot blocks before page render completes.

## Anti-Bot Challenge Mitigation & Rate Limiting
Automation routines implement adaptive exponential backoff with jitter (500ms to 4000ms delay).
When Cloudflare Turnstile, reCAPTCHA, or 2FA challenge screens are detected via DOM fingerprinting, execution automatically freezes and notifies the human operator.""",
    },
]


def generate_100_eval_cases() -> List[EvalCase]:
    """Generates 100 comprehensive, realistic ground-truth evaluation cases covering all core modules."""
    cases = []

    # 1-20: API Gateway & Rate Limiting
    gw_topics = [
        ("What token bucket capacity and refill rate does the API Gateway enforce?", "doc_api_gateway", ["token bucket", "500", "100 tokens per second"], "500 tokens burst capacity with refill rate of 100 tokens per second.", "API Gateway"),
        ("What HTTP status code is returned when API Gateway rate limits are exceeded?", "doc_api_gateway", ["429", "Too Many Requests"], "HTTP 429 Too Many Requests.", "API Gateway"),
        ("Which headers are returned on API rate limit rejection?", "doc_api_gateway", ["X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"], "X-RateLimit-Limit, X-RateLimit-Remaining, and Retry-After headers.", "API Gateway"),
        ("How is external ingress traffic authenticated at the proxy edge?", "doc_api_gateway", ["mTLS", "Envoy proxy"], "Terminates mTLS at the Envoy proxy edge.", "API Gateway"),
        ("What algorithm and key size are used to sign microservice JWT bearer tokens?", "doc_api_gateway", ["RS256", "2048-bit RSA"], "Signed using RS256 with 2048-bit RSA keys.", "API Gateway"),
        ("What is the maximum token expiration time for JWT bearer tokens?", "doc_api_gateway", ["15 minutes"], "Strictly capped at 15 minutes.", "API Gateway"),
        ("Which OAuth grant type is used to rotate refresh tokens?", "doc_api_gateway", ["Token Exchange", "OAuth 2.0"], "OAuth 2.0 Token Exchange grant.", "API Gateway"),
        ("What downstream error rate triggers the circuit breaker?", "doc_api_gateway", ["50%", "10-second"], "Error rates exceeding 50% over a 10-second rolling window.", "API Gateway"),
        ("What is the minimum request volume required to trigger circuit breakers?", "doc_api_gateway", ["20 requests"], "Minimum volume threshold of 20 requests.", "API Gateway"),
        ("What HTTP status code is returned when a circuit breaker is in OPEN state?", "doc_api_gateway", ["503", "Service Unavailable"], "HTTP 503 Service Unavailable.", "API Gateway"),
        ("Why does the circuit breaker return immediate 503 responses?", "doc_api_gateway", ["socket pool exhaustion"], "To prevent cascading socket pool exhaustion.", "API Gateway"),
        ("Which distributed datastore backs the API Gateway rate limiter?", "doc_api_gateway", ["Redis Cluster"], "Redis Cluster backs the rate limiter.", "API Gateway"),
        ("What proxy software is used at the external ingress edge?", "doc_api_gateway", ["Envoy"], "Envoy proxy.", "API Gateway"),
        ("How are refresh tokens managed in the API Gateway?", "doc_api_gateway", ["OAuth 2.0", "refresh tokens rotated"], "Rotated through the OAuth 2.0 Token Exchange grant.", "API Gateway"),
        ("What window size is used to measure circuit breaker error rates?", "doc_api_gateway", ["10-second rolling window"], "10-second rolling window.", "API Gateway"),
        ("How does the API Gateway handle requests when rate limited?", "doc_api_gateway", ["HTTP 429", "Retry-After"], "Returns HTTP 429 Too Many Requests with Retry-After.", "API Gateway"),
        ("What key format is used for signing microservice tokens?", "doc_api_gateway", ["RS256", "RSA"], "RS256 with 2048-bit RSA keys.", "API Gateway"),
        ("What is the steady refill rate of the API gateway token bucket?", "doc_api_gateway", ["100 tokens per second"], "100 tokens per second.", "API Gateway"),
        ("What happens when circuit breaker enters OPEN state?", "doc_api_gateway", ["HTTP 503", "OPEN state"], "Immediate HTTP 503 Service Unavailable responses.", "API Gateway"),
        ("What is the burst capacity of the API gateway token bucket?", "doc_api_gateway", ["500 tokens"], "500 tokens burst capacity.", "API Gateway"),
    ]
    for idx, (q, d, kw, ans, cat) in enumerate(gw_topics, start=1):
        cases.append(EvalCase(case_id=f"eval_{idx:03d}", query=q, expected_doc_id=d, expected_keywords=kw, ground_truth_answer=ans, category=cat))

    # 21-40: PostgreSQL & PgVector
    pg_topics = [
        ("Which vector indexing algorithms does pgvector provide?", "doc_postgres_pgvector", ["HNSW", "IVFFlat"], "HNSW and IVFFlat indexing.", "Databases"),
        ("What are the recommended HNSW index parameters for datasets over 100k vectors?", "doc_postgres_pgvector", ["m = 16", "ef_construction = 64"], "HNSW with m = 16 and ef_construction = 64.", "Databases"),
        ("What should ef_search be tuned to for optimal recall and latency?", "doc_postgres_pgvector", ["ef_search", "40"], "ef_search tuned to 40 for optimal balance.", "Databases"),
        ("Which SQL operator computes cosine distance in pgvector?", "doc_postgres_pgvector", ["<=>", "cosine distance"], "The <=> operator computes cosine distance.", "Databases"),
        ("Which SQL operator computes L2 squared distance in pgvector?", "doc_postgres_pgvector", ["<->", "L2"], "The <-> operator computes L2 squared distance.", "Databases"),
        ("Which pooling software is used for PostgreSQL connections?", "doc_postgres_pgvector", ["PgBouncer", "transaction pooling"], "PgBouncer operating in transaction pooling mode.", "Databases"),
        ("What is the maximum pool size per PostgreSQL database instance?", "doc_postgres_pgvector", ["200", "server connections"], "Maximum pool size of 200 server connections.", "Databases"),
        ("What is the statement timeout for OLTP queries in PostgreSQL?", "doc_postgres_pgvector", ["8000ms", "OLTP"], "8000ms statement timeout for OLTP queries.", "Databases"),
        ("What is the statement timeout for analytical batch jobs in PostgreSQL?", "doc_postgres_pgvector", ["60000ms", "analytical"], "60000ms for analytical batch jobs.", "Databases"),
        ("How does WAL replication stream data to warm standby replicas?", "doc_postgres_pgvector", ["asynchronously", "streaming replication"], "Streaming replication runs asynchronously.", "Databases"),
        ("How often are continuous archiving WAL segments uploaded to storage?", "doc_postgres_pgvector", ["60 seconds", "WAL segments"], "Uploaded to S3-compatible storage every 60 seconds.", "Databases"),
        ("What Recovery Point Objective (RPO) does PostgreSQL continuous archiving guarantee?", "doc_postgres_pgvector", ["under 1 minute", "RPO"], "Guarantees an RPO of under 1 minute.", "Databases"),
        ("Why is PgBouncer pool size capped at 200 server connections?", "doc_postgres_pgvector", ["OS context switching overhead"], "To avoid OS context switching overhead.", "Databases"),
        ("What recall level is achieved with ef_search tuned to 40 in pgvector?", "doc_postgres_pgvector", ["98% recall", "sub-5ms"], "98% recall with sub-5ms query latency.", "Databases"),
        ("What extension is required for vector similarity search in Postgres?", "doc_postgres_pgvector", ["pgvector"], "The pgvector extension.", "Databases"),
        ("Which pgvector index is mandatory for datasets above 100,000 vectors?", "doc_postgres_pgvector", ["HNSW", "mandatory"], "HNSW index is mandatory.", "Databases"),
        ("Where are standby PostgreSQL replicas located?", "doc_postgres_pgvector", ["availability zones", "distributed"], "Distributed across availability zones.", "Databases"),
        ("What mode does PgBouncer operate in?", "doc_postgres_pgvector", ["transaction pooling mode"], "Transaction pooling mode.", "Databases"),
        ("What is the cosine operator in pgvector syntax?", "doc_postgres_pgvector", ["<=>"], "The <=> operator.", "Databases"),
        ("What is the L2 distance operator in pgvector syntax?", "doc_postgres_pgvector", ["<->"], "The <-> operator.", "Databases"),
    ]
    for idx, (q, d, kw, ans, cat) in enumerate(pg_topics, start=21):
        cases.append(EvalCase(case_id=f"eval_{idx:03d}", query=q, expected_doc_id=d, expected_keywords=kw, ground_truth_answer=ans, category=cat))

    # 41-60: Security & Prompt Injection Defense
    sec_topics = [
        ("How many layers are in the prompt inspection engine?", "doc_security_guardrails", ["3-layer", "deterministic inspection"], "A 3-layer deterministic inspection engine.", "Security"),
        ("What is checked in layer 1 of the prompt inspection engine?", "doc_security_guardrails", ["Regex pattern matching", "DAN mode", "prompt leakage"], "Regex pattern matching against known adversarial exploits.", "Security"),
        ("What does structural sanitization strip from user prompts in layer 2?", "doc_security_guardrails", ["non-printable", "control characters", "null bytes"], "Strips non-printable control characters, null bytes, and escape sequences.", "Security"),
        ("What sensitive PII is automatically scrubbed by the security layer?", "doc_security_guardrails", ["email addresses", "phone numbers", "government"], "Email addresses, phone numbers, and government ID numbers.", "Security"),
        ("How are structured outputs enforced on LLM generation pipelines?", "doc_security_guardrails", ["JSON schemas", "Pydantic", "AJV"], "Strict JSON schemas validated via Pydantic or AJV.", "Security"),
        ("What happens when an LLM output fails schema validation?", "doc_security_guardrails", ["one self-correction retry", "deterministic fallback"], "Exactly one self-correction retry before defaulting to fallback.", "Security"),
        ("How are user inputs delimited to prevent prompt injection?", "doc_security_guardrails", ["<user_query>", "XML demarcation"], "Encapsulated within strict <user_query> XML boundaries.", "Security"),
        ("How are models instructed to treat delimiter-enclosed user input?", "doc_security_guardrails", ["passive textual data", "executable directives"], "Treat enclosed contents as passive textual data, not executable directives.", "Security"),
        ("Can LLM responses be accepted as raw markdown?", "doc_security_guardrails", ["never parsed as raw markdown"], "Never parsed as raw markdown; schemas are strictly enforced.", "Security"),
        ("What exploits does the security guard regex pattern catch?", "doc_security_guardrails", ["DAN mode", "system prompt leakage", "instruction override"], "DAN mode, system prompt leakage, and instruction overrides.", "Security"),
        ("What is the fallback behavior when self-correction fails?", "doc_security_guardrails", ["deterministic fallback response"], "Defaults to a deterministic fallback response.", "Security"),
        ("What XML tag encapsulates user input in prompts?", "doc_security_guardrails", ["<user_query>"], "The <user_query> XML tag.", "Security"),
        ("How many self-correction retries are allowed on schema failure?", "doc_security_guardrails", ["exactly one", "retry"], "Exactly one self-correction retry.", "Security"),
        ("Which libraries validate structured LLM JSON outputs?", "doc_security_guardrails", ["Pydantic", "AJV"], "Validated via Pydantic or AJV.", "Security"),
        ("What does PII masking protect in user prompts?", "doc_security_guardrails", ["email", "phone", "PII scrubbing"], "Masks email addresses, phone numbers, and IDs.", "Security"),
        ("Why are control characters stripped during sanitization?", "doc_security_guardrails", ["structural sanitization", "escape sequences"], "To neutralize prompt injection and escape exploits.", "Security"),
        ("What boundary prevents user instructions from overriding the system prompt?", "doc_security_guardrails", ["XML demarcation boundaries"], "Strict XML demarcation boundaries and prompt isolation.", "Security"),
        ("What happens to null bytes in incoming prompts?", "doc_security_guardrails", ["stripping", "null bytes"], "Stripped by structural sanitization.", "Security"),
        ("Does the security engine use non-deterministic filters?", "doc_security_guardrails", ["deterministic inspection engine"], "No, it uses a 3-layer deterministic inspection engine.", "Security"),
        ("How are adversarial exploits detected before reaching the model?", "doc_security_guardrails", ["Regex pattern matching", "3-layer"], "Via regex pattern matching and structural sanitization.", "Security"),
    ]
    for idx, (q, d, kw, ans, cat) in enumerate(sec_topics, start=41):
        cases.append(EvalCase(case_id=f"eval_{idx:03d}", query=q, expected_doc_id=d, expected_keywords=kw, ground_truth_answer=ans, category=cat))

    # 61-80: Distributed Caching Strategy
    cache_topics = [
        ("What is the topology of the Redis cluster?", "doc_distributed_cache", ["6-node", "3 primaries", "3 read replicas"], "6-node cluster with 3 primaries and 3 read replicas.", "Infrastructure"),
        ("What key eviction policy does Redis employ?", "doc_distributed_cache", ["volatile-lru"], "The volatile-lru eviction policy.", "Infrastructure"),
        ("What is the soft memory ceiling configured for Redis eviction?", "doc_distributed_cache", ["80%", "maximum memory capacity"], "80% of maximum memory capacity.", "Infrastructure"),
        ("What namespaced prefix convention is used for Redis keys?", "doc_distributed_cache", ["tenant:{id}:session:{sid}", "namespaced"], "Prefixes like tenant:{id}:session:{sid} with deterministic TTLs.", "Infrastructure"),
        ("Which caching pattern is used to prevent stale reads?", "doc_distributed_cache", ["Cache-Aside", "Kafka"], "Cache-Aside pattern with asynchronous event-driven invalidation.", "Infrastructure"),
        ("How are database mutations emitted for cache invalidation?", "doc_distributed_cache", ["Outbox table", "Kafka"], "An Outbox table emits an invalidation event via Kafka.", "Infrastructure"),
        ("What is the maximum latency for cache key purging after an update?", "doc_distributed_cache", ["150ms", "Redis keys"], "Purges corresponding Redis keys within 150ms.", "Infrastructure"),
        ("Which algorithm mitigates cache stampedes on high-traffic endpoints?", "doc_distributed_cache", ["XFetch algorithm", "probabilistic early expiration"], "Probabilistic early expiration (XFetch algorithm) with beta = 1.0.", "Infrastructure"),
        ("What lock algorithm prevents redundant queries on cache misses?", "doc_distributed_cache", ["Redlock algorithm", "mutual exclusion mutex"], "Mutual exclusion mutex locks (Redlock algorithm).", "Infrastructure"),
        ("What parameter is beta set to in the XFetch cache algorithm?", "doc_distributed_cache", ["beta = 1.0"], "beta is set to 1.0.", "Infrastructure"),
        ("How does the Redis cluster handle node failure?", "doc_distributed_cache", ["automatic failover"], "Configured with automatic failover across read replicas.", "Infrastructure"),
        ("What version of Redis does the caching cluster run?", "doc_distributed_cache", ["Redis 7.2"], "Redis 7.2.", "Infrastructure"),
        ("What happens when Redis memory reaches 80% ceiling?", "doc_distributed_cache", ["volatile-lru", "eviction"], "Evicts keys according to the volatile-lru policy.", "Infrastructure"),
        ("Why is Kafka used in the caching architecture?", "doc_distributed_cache", ["event-driven invalidation", "Outbox"], "For asynchronous event-driven cache invalidation from the Outbox.", "Infrastructure"),
        ("What problem does the Redlock algorithm solve in caching?", "doc_distributed_cache", ["cache stampede", "redundant", "queries"], "Prevents thundering herd and redundant database queries.", "Infrastructure"),
        ("Do cache keys have indefinite lifetimes?", "doc_distributed_cache", ["deterministic TTLs"], "No, all keys require deterministic TTLs.", "Infrastructure"),
        ("How many primary nodes exist in the Redis cluster?", "doc_distributed_cache", ["3 primaries"], "3 primary nodes.", "Infrastructure"),
        ("How many read replicas exist in the Redis cluster?", "doc_distributed_cache", ["3 read replicas"], "3 read replicas.", "Infrastructure"),
        ("When does XFetch recalculate cache values?", "doc_distributed_cache", ["prior to hard TTL expiration"], "Prior to hard TTL expiration.", "Infrastructure"),
        ("What table pattern captures database updates for caching?", "doc_distributed_cache", ["Outbox table"], "The Outbox table pattern.", "Infrastructure"),
    ]
    for idx, (q, d, kw, ans, cat) in enumerate(cache_topics, start=61):
        cases.append(EvalCase(case_id=f"eval_{idx:03d}", query=q, expected_doc_id=d, expected_keywords=kw, ground_truth_answer=ans, category=cat))

    # 81-100: Playwright & CDP Browser Automation
    pw_topics = [
        ("What client-side storage state is persisted across Playwright sessions?", "doc_playwright_cdp", ["indexedDB", "cookies", "local storage"], "indexedDB, cookies, and local storage state.", "Browser Automation"),
        ("Why does Playwright persist browser contexts to disk?", "doc_playwright_cdp", ["multi-factor authentication", "avoid repeated"], "To avoid repeated multi-factor authentication challenges.", "Browser Automation"),
        ("How often is serialized context state rotated?", "doc_playwright_cdp", ["24 hours", "rotated"], "Rotated every 24 hours.", "Browser Automation"),
        ("In what format is Playwright session state saved?", "doc_playwright_cdp", ["encrypted JSON files"], "Encrypted JSON files.", "Browser Automation"),
        ("What does direct integration with Chrome DevTools Protocol enable?", "doc_playwright_cdp", ["interception of network events", "DOM mutations"], "Low-level interception of network events and DOM mutations.", "Browser Automation"),
        ("Which CDP events do automation workers listen to?", "doc_playwright_cdp", ["Network.requestWillBeSent", "Network.responseReceived"], "Network.requestWillBeSent and Network.responseReceived.", "Browser Automation"),
        ("Why are CDP network events monitored before page render finishes?", "doc_playwright_cdp", ["anti-bot blocks", "detect", "telemetry"], "To detect anti-bot blocks and telemetry before render completes.", "Browser Automation"),
        ("What backoff strategy do automation routines use to avoid rate limiting?", "doc_playwright_cdp", ["exponential backoff with jitter", "500ms to 4000ms"], "Adaptive exponential backoff with jitter (500ms to 4000ms delay).", "Browser Automation"),
        ("What anti-bot challenges are detected via DOM fingerprinting?", "doc_playwright_cdp", ["Cloudflare Turnstile", "reCAPTCHA", "2FA"], "Cloudflare Turnstile, reCAPTCHA, and 2FA challenge screens.", "Browser Automation"),
        ("What happens when a Cloudflare or reCAPTCHA challenge is detected?", "doc_playwright_cdp", ["execution automatically freezes", "notifies human"], "Execution freezes and notifies the human operator.", "Browser Automation"),
        ("What is the delay range for exponential backoff with jitter?", "doc_playwright_cdp", ["500ms to 4000ms"], "500ms to 4000ms delay.", "Browser Automation"),
        ("Does the automation worker attempt to solve 2FA autonomously?", "doc_playwright_cdp", ["notifies the human operator", "freezes"], "No, execution freezes and notifies the human operator.", "Browser Automation"),
        ("What protocol connects Playwright directly to Chrome internals?", "doc_playwright_cdp", ["Chrome DevTools Protocol", "CDP"], "Chrome DevTools Protocol (CDP).", "Browser Automation"),
        ("How are cookies stored securely for persistent browser contexts?", "doc_playwright_cdp", ["encrypted JSON files"], "Serialized to encrypted JSON files.", "Browser Automation"),
        ("What event detects outgoing network requests in CDP?", "doc_playwright_cdp", ["Network.requestWillBeSent"], "Network.requestWillBeSent event.", "Browser Automation"),
        ("What event captures incoming HTTP responses in CDP?", "doc_playwright_cdp", ["Network.responseReceived"], "Network.responseReceived event.", "Browser Automation"),
        ("How does DOM fingerprinting detect challenge pages?", "doc_playwright_cdp", ["Turnstile", "reCAPTCHA", "fingerprinting"], "By fingerprinting challenge frames and Turnstile elements.", "Browser Automation"),
        ("Why is jitter added to exponential backoff delays?", "doc_playwright_cdp", ["jitter", "anti-bot", "rate limiting"], "To prevent synchronized request bursts and evade rate limits.", "Browser Automation"),
        ("How does session persistence improve crawler efficiency?", "doc_playwright_cdp", ["avoids repeated", "authentication"], "Avoids repeating login and MFA flows across batch runs.", "Browser Automation"),
        ("What happens to crawler state after 24 hours?", "doc_playwright_cdp", ["rotated every 24 hours"], "State is rotated every 24 hours.", "Browser Automation"),
    ]
    for idx, (q, d, kw, ans, cat) in enumerate(pw_topics, start=81):
        cases.append(EvalCase(case_id=f"eval_{idx:03d}", query=q, expected_doc_id=d, expected_keywords=kw, ground_truth_answer=ans, category=cat))

    return cases
