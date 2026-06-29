"""Realistic distractor decisions from unrelated domains — to stress retrieval (prove recall
discriminates the relevant entry, not a small-store artifact)."""
DISTRACTORS = [
 ("cache", "TTL cache evicts on write, not read", "We evict expired entries on write rather than on read so reads stay O(1); a sweep on read caused latency spikes under load."),
 ("auth", "JWT access tokens expire after 15 minutes", "Access tokens live 15 minutes, refresh tokens 30 days; a longer access TTL widened the window for stolen-token replay."),
 ("pagination", "default page size is 25, max 100", "Pages default to 25 and are capped at 100; larger pages timed out the downstream join."),
 ("serialization", "datetimes serialize as UTC ISO-8601 with Z", "All timestamps serialize in UTC with a trailing Z; mixing local offsets broke cross-region comparisons."),
 ("logging", "PII is redacted from logs at the formatter", "We redact emails and tokens in the log formatter, not at call sites, so nothing leaks if a dev forgets."),
 ("db", "use SELECT ... FOR UPDATE SKIP LOCKED for the job queue", "Workers pull jobs with SKIP LOCKED to avoid lock contention; FOR UPDATE alone serialized all workers."),
 ("http", "retry only idempotent verbs on 5xx", "We retry GET/PUT/DELETE on 5xx but never POST, to avoid duplicate side effects."),
 ("validation", "trim and lowercase emails before storage", "Emails are normalized (trim + lowercase) before persisting so uniqueness checks match."),
 ("concurrency", "the worker pool size is CPU count minus one", "Pool size = cores-1 to leave a core for the event loop; saturating all cores stalled heartbeats."),
 ("currency", "store money as integer minor units, never float", "Amounts are integer cents; floats accumulated rounding error across ledgers."),
 ("search", "stem queries but not exact-match filters", "Free-text is stemmed; quoted/filter terms are matched verbatim so SKU lookups don't fuzz."),
 ("upload", "reject files over 10 MiB at the edge", "The 10 MiB cap is enforced at the proxy, before buffering, to avoid OOM on large uploads."),
 ("email", "send transactional mail through the priority pool", "Transactional mail uses a separate pool from marketing so a campaign can't delay receipts."),
 ("feature_flags", "flags default to OFF when the service is unreachable", "Fail-closed on flag fetch errors; a fail-open default once shipped an unfinished feature."),
 ("rate_limit", "rate limits are per-API-key, not per-IP", "We key limits on the API key because many clients share a NAT IP."),
 ("timezone", "cron schedules are interpreted in UTC", "All schedules are UTC; a DST shift double-ran a nightly job when we used local time."),
 ("csv", "quote every field in exported CSV", "We always quote fields so embedded commas and newlines don't corrupt downstream parsers."),
 ("session", "rotate the session id on privilege change", "Session ids rotate on login and role change to prevent fixation."),
 ("images", "strip EXIF metadata on upload", "EXIF (including GPS) is stripped on upload for privacy."),
 ("api", "version the API in the URL path, not a header", "Path versioning (/v2/) because header versioning broke CDN caching."),
 ("queue", "dead-letter after 5 failed deliveries", "Messages dead-letter after 5 attempts; infinite retries flooded the broker."),
 ("password", "hash passwords with argon2id, not bcrypt", "We moved to argon2id for memory-hardness against GPU cracking."),
 ("slug", "truncate slugs to 60 characters", "Slugs cap at 60 chars to keep URLs and index keys bounded."),
 ("locale", "fall back to en-US when a locale is missing", "Missing translations fall back to en-US rather than showing keys."),
 ("metrics", "sample traces at 1 percent in production", "1% trace sampling keeps overhead low while still catching tail latency."),
 ("backup", "keep daily backups for 30 days, weekly for a year", "Retention is 30 daily + 52 weekly to balance cost and recovery needs."),
 ("import", "skip a malformed row, do not abort the import", "A bad row is logged and skipped so one typo doesn't fail a 10k-row import."),
 ("webhook", "sign webhooks with an HMAC of the raw body", "We HMAC the raw bytes (not the parsed JSON) so re-serialization can't change the signature."),
 ("decimal", "quantize tax to 4 places, round the total to 2", "Tax is computed at 4 dp then the invoice total rounds to 2, to match the tax authority."),
 ("retry", "use full jitter on backoff, capped at 30s", "Full jitter spreads retries; the cap keeps the worst-case wait bounded."),
 ("nulls", "treat empty string and null as distinct in the API", "We keep '' and null distinct because clients use '' to clear and null to leave unchanged."),
 ("sort", "sort search results by score, ties broken by recency", "Equal-score results are ordered newest-first."),
 ("units", "store durations in milliseconds internally", "Internal durations are integer ms to avoid float second/minute conversions."),
 ("cors", "allow-list origins, never reflect the Origin header", "We match against an allow-list; reflecting Origin once allowed any site to call us."),
 ("id", "use ULIDs for public ids, not auto-increment", "ULIDs avoid leaking row counts and are sortable by time."),
 ("encoding", "normalize unicode to NFC before comparison", "We NFC-normalize so visually identical names compare equal."),
 ("flags", "boolean env vars: only 'true' (lowercase) is true", "Exactly the string 'true' enables a flag; '1'/'yes' were ambiguous across services."),
 ("price", "apply percentage discounts before fixed-amount ones", "Percent discounts apply first, then fixed amounts, so stacking is deterministic."),
 ("cache2", "cache keys include the schema version", "Keys are prefixed with the schema version so a migration invalidates stale entries."),
]
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mem import _text_symbols
ENTRIES = [{"project": f"distractor-{k}", "kind": "decision", "title": t, "text": f"{t} — {body}",
            "symbols": _text_symbols(f"{k} {t} {body}")} for k, t, body in DISTRACTORS]
