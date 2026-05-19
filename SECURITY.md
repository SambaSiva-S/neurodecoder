# Security Policy — NeuroDecoder BCI Platform

## Defense-in-Depth: Multi-Zone Shield Architecture

NeuroDecoder implements a 9-layer security architecture across three zones to protect patients, data, and infrastructure.

### Zone 1: Perimeter (Input Security)

| Layer | Protection | Implementation |
|-------|-----------|----------------|
| L1 | Input Validation | File format, size, channels, amplitude, NaN/Inf checks |
| L2 | Rate Limiting | 10/min, 100/hr, 1000/day per session |
| L3 | API Key Management | Environment variables only, never hardcoded |

### Zone 2: Runtime (Processing Security)

| Layer | Protection | Implementation |
|-------|-----------|----------------|
| L4 | Model Armor | Statistical anomaly detection on input features |
| L5 | Budget Guardrails | $5/session, $25/day, $200/month API caps |
| L6 | Circuit Breaker | 3 failures → 15-minute auto-pause |

### Zone 3: Output (Result Security)

| Layer | Protection | Implementation |
|-------|-----------|----------------|
| L7 | Output Validation | Prediction class, confidence range, probability distribution |
| L8 | PII Protection | No data retention, metadata sanitization, HIPAA-aware |
| L9 | Audit Logging | All events tracked with timestamps and session IDs |

## Data Privacy

- **No EEG data is stored on the server.** All processing happens in memory and is discarded after the response.
- **No patient identifiers are logged.** Only anonymized metadata (channel count, sampling rate, duration).
- **HIPAA-aware design.** While this is a research tool (not a medical device), we follow HIPAA principles for data handling.

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it responsibly:

- **Email:** sanapathissds@gmail.com
- **Subject:** [SECURITY] NeuroDecoder Vulnerability Report
- **Please include:** Description of the vulnerability, steps to reproduce, potential impact

We will acknowledge receipt within 48 hours and provide a fix timeline within 7 days.

## API Keys

- Never commit API keys to the repository
- Use the `.env.template` file as a guide
- Store keys in environment variables
- API keys are masked in all logs (only last 4 characters shown)

## Robot Safety

When connected to physical hardware (wheelchair, robotic arm):

- **No actuation below 40% confidence** — prevents uncertain commands from causing harm
- **All commands validated** before transmission to hardware
- **Circuit breaker** stops all hardware commands after repeated errors
- **Kill switch** capability for immediate halt

## Security Philosophy

**"Default Deny"** — All inputs are untrusted, all outputs are validated, all actions are logged. AI agents operate within strict, code-governed boundaries.
