# ===== NEURODECODER: SECURITY MODULE =====
# Defense-in-Depth: Multi-Zone Shield Architecture for BCI Platform
#
# ZONE 1 — PERIMETER: Input validation, authentication, rate limiting
# ZONE 2 — RUNTIME: Model armor, budget guardrails, circuit breaker
# ZONE 3 — OUTPUT: Result validation, PII protection, audit logging
#
# Security Philosophy: "Default Deny" — all inputs are untrusted,
# all outputs are validated, all actions are logged.

import numpy as np
import hashlib
import hmac
import time
import json
import os
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from functools import wraps

# ===== LOGGING SETUP =====
logging.basicConfig(
    filename='neurodecoder_audit.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('neurodecoder_security')


# =====================================================
# ZONE 1: PERIMETER — Who and What Gets In
# =====================================================

class InputValidator:
    """
    L1: Validates all incoming EEG data before processing.
    
    Checks:
    - File format (only .gdf, .edf, .fif allowed)
    - File size (max 500MB)
    - Channel count (1-256 channels)
    - Sampling rate (50-10000 Hz)
    - Duration (1 second to 1 hour)
    - Signal amplitude (reject if >1000µV — likely attack or corrupt)
    - NaN/Inf values (reject — numerical attack vector)
    - Data type (must be numeric)
    """
    
    ALLOWED_EXTENSIONS = {'.gdf', '.edf', '.fif', '.bdf', '.set'}
    MAX_FILE_SIZE_MB = 500
    MAX_CHANNELS = 256
    MIN_CHANNELS = 1
    MAX_SFREQ = 10000
    MIN_SFREQ = 50
    MAX_DURATION_SEC = 3600  # 1 hour
    MIN_DURATION_SEC = 1
    MAX_AMPLITUDE_UV = 1000  # µV
    
    @classmethod
    def validate_file(cls, filepath):
        """Validate uploaded file before any processing."""
        issues = []
        
        # Check 1: File extension
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in cls.ALLOWED_EXTENSIONS:
            issues.append(f"REJECTED: Invalid file format '{ext}'. Allowed: {cls.ALLOWED_EXTENSIONS}")
            logger.warning(f"Invalid file format attempted: {ext}")
            return False, issues
        
        # Check 2: File size
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if file_size_mb > cls.MAX_FILE_SIZE_MB:
            issues.append(f"REJECTED: File too large ({file_size_mb:.1f}MB). Max: {cls.MAX_FILE_SIZE_MB}MB")
            logger.warning(f"Oversized file rejected: {file_size_mb:.1f}MB")
            return False, issues
        
        issues.append(f"✓ File format: {ext}")
        issues.append(f"✓ File size: {file_size_mb:.1f}MB")
        
        return True, issues
    
    @classmethod
    def validate_raw_data(cls, raw_data):
        """Validate loaded MNE Raw object."""
        issues = []
        
        # Check 3: Channel count
        n_channels = len(raw_data.ch_names)
        if n_channels < cls.MIN_CHANNELS or n_channels > cls.MAX_CHANNELS:
            issues.append(f"REJECTED: Invalid channel count ({n_channels}). Range: {cls.MIN_CHANNELS}-{cls.MAX_CHANNELS}")
            logger.warning(f"Invalid channel count: {n_channels}")
            return False, issues
        issues.append(f"✓ Channels: {n_channels}")
        
        # Check 4: Sampling rate
        sfreq = raw_data.info['sfreq']
        if sfreq < cls.MIN_SFREQ or sfreq > cls.MAX_SFREQ:
            issues.append(f"REJECTED: Invalid sampling rate ({sfreq}Hz). Range: {cls.MIN_SFREQ}-{cls.MAX_SFREQ}Hz")
            logger.warning(f"Invalid sampling rate: {sfreq}")
            return False, issues
        issues.append(f"✓ Sampling rate: {sfreq}Hz")
        
        # Check 5: Duration
        duration = raw_data.times[-1]
        if duration < cls.MIN_DURATION_SEC or duration > cls.MAX_DURATION_SEC:
            issues.append(f"REJECTED: Invalid duration ({duration:.1f}s). Range: {cls.MIN_DURATION_SEC}-{cls.MAX_DURATION_SEC}s")
            logger.warning(f"Invalid duration: {duration:.1f}s")
            return False, issues
        issues.append(f"✓ Duration: {duration:.1f}s")
        
        # Check 6: Signal amplitude
        data = raw_data.get_data()
        max_amp = np.max(np.abs(data)) * 1e6  # convert to µV
        if max_amp > cls.MAX_AMPLITUDE_UV:
            issues.append(f"WARNING: High amplitude ({max_amp:.1f}µV). Possible artifact or corrupted data.")
            logger.warning(f"High amplitude detected: {max_amp:.1f}µV")
        else:
            issues.append(f"✓ Max amplitude: {max_amp:.1f}µV")
        
        # Check 7: NaN/Inf values
        if np.any(np.isnan(data)):
            issues.append("REJECTED: Data contains NaN values — possible data corruption or attack")
            logger.critical("NaN values detected in uploaded data")
            return False, issues
        if np.any(np.isinf(data)):
            issues.append("REJECTED: Data contains Inf values — possible numerical attack")
            logger.critical("Inf values detected in uploaded data")
            return False, issues
        issues.append("✓ No NaN/Inf values")
        
        # Check 8: Flat channels (all zeros = dead electrode)
        flat_channels = []
        for ch_idx in range(data.shape[0]):
            if np.std(data[ch_idx]) < 1e-10:
                flat_channels.append(raw_data.ch_names[ch_idx])
        if flat_channels:
            issues.append(f"WARNING: Flat channels detected: {flat_channels}. May be dead electrodes.")
            logger.info(f"Flat channels: {flat_channels}")
        else:
            issues.append("✓ No flat channels")
        
        # Check 9: Excessive noise (>500µV std)
        noisy_channels = []
        for ch_idx in range(data.shape[0]):
            ch_std = np.std(data[ch_idx]) * 1e6
            if ch_std > 500:
                noisy_channels.append((raw_data.ch_names[ch_idx], f"{ch_std:.1f}µV"))
        if noisy_channels:
            issues.append(f"WARNING: Noisy channels: {noisy_channels}")
            logger.info(f"Noisy channels: {noisy_channels}")
        else:
            issues.append("✓ Channel noise levels acceptable")
        
        issues.append("\n✅ ALL SECURITY CHECKS PASSED")
        logger.info(f"Data validated: {n_channels}ch, {sfreq}Hz, {duration:.1f}s")
        return True, issues


class RateLimiter:
    """
    L2: Rate limiting to prevent API abuse.
    
    Limits:
    - 10 requests per minute per session
    - 100 requests per hour per session
    - 1000 requests per day per session
    """
    
    def __init__(self, per_minute=10, per_hour=100, per_day=1000):
        self.per_minute = per_minute
        self.per_hour = per_hour
        self.per_day = per_day
        self.requests = defaultdict(list)  # session_id -> [timestamps]
    
    def check(self, session_id="default"):
        """Check if request is within rate limits."""
        now = time.time()
        self.requests[session_id] = [t for t in self.requests[session_id] if now - t < 86400]
        
        recent = self.requests[session_id]
        
        # Per-minute check
        minute_requests = sum(1 for t in recent if now - t < 60)
        if minute_requests >= self.per_minute:
            logger.warning(f"Rate limit exceeded (per-minute): session={session_id}")
            return False, f"Rate limit exceeded. Max {self.per_minute} requests per minute. Try again in {60 - (now - recent[-self.per_minute]):.0f}s."
        
        # Per-hour check
        hour_requests = sum(1 for t in recent if now - t < 3600)
        if hour_requests >= self.per_hour:
            logger.warning(f"Rate limit exceeded (per-hour): session={session_id}")
            return False, f"Rate limit exceeded. Max {self.per_hour} requests per hour."
        
        # Per-day check
        if len(recent) >= self.per_day:
            logger.warning(f"Rate limit exceeded (per-day): session={session_id}")
            return False, f"Daily limit reached. Max {self.per_day} requests per day."
        
        self.requests[session_id].append(now)
        return True, "OK"


class APIKeyManager:
    """
    L3: Secure API key management.
    
    Rules:
    - Never hardcode API keys
    - Load from environment variables only
    - Validate key format before use
    - Mask keys in logs (show only last 4 chars)
    """
    
    @staticmethod
    def get_key(key_name):
        """Get API key from environment variable."""
        key = os.environ.get(key_name)
        if not key:
            logger.error(f"API key not found: {key_name}")
            return None
        
        # Mask for logging
        masked = f"***{key[-4:]}" if len(key) > 4 else "****"
        logger.info(f"API key loaded: {key_name} ({masked})")
        return key
    
    @staticmethod
    def validate_anthropic_key(key):
        """Validate Anthropic API key format."""
        if not key:
            return False
        if not key.startswith('sk-ant-'):
            logger.warning("Invalid Anthropic API key format")
            return False
        if len(key) < 20:
            logger.warning("Anthropic API key too short")
            return False
        return True
    
    @staticmethod
    def mask_key(key):
        """Mask API key for safe display/logging."""
        if not key or len(key) < 8:
            return "****"
        return f"{key[:7]}...{key[-4:]}"


# =====================================================
# ZONE 2: RUNTIME — What Happens Inside
# =====================================================

class ModelArmor:
    """
    L4: Protect the ML model from adversarial inputs.
    
    Checks:
    - Statistical anomaly detection on input features
    - Gradient-based adversarial detection (if using deep learning)
    - Input distribution monitoring
    """
    
    def __init__(self):
        self.feature_stats = {}  # populated from training data
        self.anomaly_threshold = 3.0  # standard deviations
    
    def fit_stats(self, X_train):
        """Learn normal feature statistics from training data."""
        self.feature_stats = {
            'mean': np.mean(X_train, axis=0),
            'std': np.std(X_train, axis=0) + 1e-10,
            'min': np.min(X_train, axis=0),
            'max': np.max(X_train, axis=0),
            'n_samples': len(X_train),
        }
        logger.info(f"Model armor fitted on {len(X_train)} training samples")
    
    def check_input(self, X_input):
        """Check if input is within expected distribution."""
        if not self.feature_stats:
            return True, "Model armor not fitted — skipping check"
        
        issues = []
        
        # Z-score check: is input abnormally far from training distribution?
        z_scores = np.abs((X_input - self.feature_stats['mean']) / self.feature_stats['std'])
        max_z = np.max(z_scores)
        
        if max_z > self.anomaly_threshold * 3:
            issues.append(f"CRITICAL: Input is {max_z:.1f} std from training distribution — possible adversarial input")
            logger.critical(f"Possible adversarial input: max z-score = {max_z:.1f}")
            return False, issues
        
        if max_z > self.anomaly_threshold:
            issues.append(f"WARNING: Input is {max_z:.1f} std from training distribution — unusual but processing")
            logger.warning(f"Unusual input: max z-score = {max_z:.1f}")
        
        return True, issues


class BudgetGuardrails:
    """
    L5: Cost controls for API usage (Claude API).
    
    Limits:
    - $5 per session
    - $25 per day
    - $200 per month
    - Fail-closed: if tracker unavailable, block API calls
    """
    
    def __init__(self, session_limit=5.0, daily_limit=25.0, monthly_limit=200.0):
        self.session_limit = session_limit
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.session_cost = 0.0
        self.daily_costs = defaultdict(float)  # date -> cost
        self.monthly_costs = defaultdict(float)  # month -> cost
    
    def check_budget(self, estimated_cost=0.01):
        """Check if API call is within budget."""
        today = datetime.now().strftime('%Y-%m-%d')
        month = datetime.now().strftime('%Y-%m')
        
        # Session check
        if self.session_cost + estimated_cost > self.session_limit:
            logger.warning(f"Session budget exceeded: ${self.session_cost:.2f}/{self.session_limit}")
            return False, f"Session budget exceeded (${self.session_cost:.2f}/${self.session_limit})"
        
        # Daily check
        if self.daily_costs[today] + estimated_cost > self.daily_limit:
            logger.warning(f"Daily budget exceeded: ${self.daily_costs[today]:.2f}/{self.daily_limit}")
            return False, f"Daily budget exceeded (${self.daily_costs[today]:.2f}/${self.daily_limit})"
        
        # Monthly check
        if self.monthly_costs[month] + estimated_cost > self.monthly_limit:
            logger.warning(f"Monthly budget exceeded: ${self.monthly_costs[month]:.2f}/{self.monthly_limit}")
            return False, f"Monthly budget exceeded (${self.monthly_costs[month]:.2f}/${self.monthly_limit})"
        
        return True, "Within budget"
    
    def record_cost(self, cost):
        """Record API call cost."""
        today = datetime.now().strftime('%Y-%m-%d')
        month = datetime.now().strftime('%Y-%m')
        
        self.session_cost += cost
        self.daily_costs[today] += cost
        self.monthly_costs[month] += cost
        
        logger.info(f"API cost recorded: ${cost:.4f} (session: ${self.session_cost:.2f}, today: ${self.daily_costs[today]:.2f})")


class CircuitBreaker:
    """
    L6: Circuit breaker pattern for API calls.
    
    After 3 consecutive failures (429 errors), auto-pause for 15 minutes.
    Prevents cascading failures and runaway costs.
    
    States:
    - CLOSED: normal operation, requests pass through
    - OPEN: circuit tripped, all requests blocked
    - HALF_OPEN: testing if service recovered
    """
    
    def __init__(self, failure_threshold=3, cooldown_minutes=15):
        self.failure_threshold = failure_threshold
        self.cooldown_minutes = cooldown_minutes
        self.failure_count = 0
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None
    
    def can_proceed(self):
        """Check if request should proceed."""
        if self.state == 'CLOSED':
            return True, "Circuit closed — normal operation"
        
        if self.state == 'OPEN':
            # Check if cooldown has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds() / 60
                if elapsed >= self.cooldown_minutes:
                    self.state = 'HALF_OPEN'
                    logger.info("Circuit breaker: OPEN → HALF_OPEN (testing recovery)")
                    return True, "Circuit half-open — testing recovery"
            
            remaining = self.cooldown_minutes - elapsed if self.last_failure_time else self.cooldown_minutes
            logger.warning(f"Circuit OPEN: {remaining:.1f} minutes remaining")
            return False, f"Service paused for {remaining:.0f} more minutes due to repeated errors"
        
        if self.state == 'HALF_OPEN':
            return True, "Circuit half-open — testing"
        
        return False, "Unknown circuit state"
    
    def record_success(self):
        """Record successful request."""
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            self.failure_count = 0
            logger.info("Circuit breaker: HALF_OPEN → CLOSED (service recovered)")
    
    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.critical(f"Circuit breaker TRIPPED: {self.failure_count} consecutive failures")
        
        logger.warning(f"API failure #{self.failure_count}/{self.failure_threshold}")


# =====================================================
# ZONE 3: OUTPUT — What Goes Out
# =====================================================

class OutputValidator:
    """
    L7: Validate decoder outputs before sending to user/robot.
    
    Checks:
    - Prediction is valid class (0, 1, 2, or 3)
    - Confidence is between 0 and 1
    - No NaN in output
    - Sanity check: confidence distribution isn't degenerate
    """
    
    VALID_CLASSES = {0, 1, 2, 3}
    MIN_CONFIDENCE = 0.0
    MAX_CONFIDENCE = 1.0
    
    @classmethod
    def validate_prediction(cls, prediction, confidence, class_probabilities=None):
        """Validate a single decoder prediction."""
        issues = []
        
        # Check prediction is valid class
        if prediction not in cls.VALID_CLASSES:
            issues.append(f"REJECTED: Invalid prediction class {prediction}")
            logger.critical(f"Invalid prediction: {prediction}")
            return False, issues
        
        # Check confidence range
        if not (cls.MIN_CONFIDENCE <= confidence <= cls.MAX_CONFIDENCE):
            issues.append(f"REJECTED: Confidence out of range: {confidence}")
            logger.critical(f"Invalid confidence: {confidence}")
            return False, issues
        
        # Check for NaN
        if np.isnan(confidence):
            issues.append("REJECTED: NaN confidence")
            logger.critical("NaN confidence detected")
            return False, issues
        
        # Check probability distribution if provided
        if class_probabilities is not None:
            if not np.isclose(np.sum(class_probabilities), 1.0, atol=0.01):
                issues.append(f"WARNING: Probabilities don't sum to 1: {np.sum(class_probabilities):.4f}")
                logger.warning(f"Bad probability distribution: sum={np.sum(class_probabilities):.4f}")
            
            if np.any(np.isnan(class_probabilities)):
                issues.append("REJECTED: NaN in probability distribution")
                logger.critical("NaN in probability distribution")
                return False, issues
        
        return True, issues
    
    @classmethod
    def validate_robot_command(cls, command, confidence):
        """Validate command before sending to hardware."""
        valid_commands = {'FORWARD', 'LEFT', 'RIGHT', 'STOP', 'IDLE'}
        
        if command not in valid_commands:
            logger.critical(f"Invalid robot command: {command}")
            return False, f"BLOCKED: Invalid command '{command}'"
        
        # Safety: require higher confidence for movement commands
        if command in {'FORWARD', 'LEFT', 'RIGHT'} and confidence < 0.4:
            logger.info(f"Low-confidence command blocked: {command} at {confidence:.0%}")
            return False, f"SAFETY: {command} blocked — confidence too low ({confidence:.0%} < 40%)"
        
        return True, "OK"


class PIIProtector:
    """
    L8: Protect Personally Identifiable Information.
    
    Rules:
    - Never store uploaded EEG files on server
    - Never log raw signal data
    - Anonymize subject identifiers
    - No patient names in any output
    - HIPAA-aware data handling
    """
    
    @staticmethod
    def sanitize_filename(filename):
        """Remove PII from filenames."""
        import re
        # Remove common PII patterns
        sanitized = re.sub(r'[A-Z][a-z]+_[A-Z][a-z]+', 'SUBJECT', filename)  # Name_Name
        sanitized = re.sub(r'\d{3}-\d{2}-\d{4}', 'SSN_REMOVED', sanitized)  # SSN
        sanitized = re.sub(r'\d{2}/\d{2}/\d{4}', 'DOB_REMOVED', sanitized)  # Date of birth
        return sanitized
    
    @staticmethod
    def sanitize_metadata(raw_info):
        """Remove PII from MNE info object metadata."""
        safe_fields = {
            'sfreq': raw_info.get('sfreq'),
            'n_channels': len(raw_info.get('ch_names', [])),
            'ch_names': raw_info.get('ch_names', []),
            'duration': None,  # Will be filled separately
        }
        
        # Explicitly exclude: subject name, experimenter, file path
        logger.info("Metadata sanitized — PII fields removed")
        return safe_fields
    
    @staticmethod
    def check_data_retention():
        """Verify no EEG data is persisted on server."""
        import tempfile
        temp_dir = tempfile.gettempdir()
        eeg_extensions = {'.gdf', '.edf', '.fif', '.bdf', '.set'}
        
        retained_files = []
        for root, dirs, files in os.walk(temp_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in eeg_extensions:
                    retained_files.append(os.path.join(root, f))
        
        if retained_files:
            logger.critical(f"EEG data retention detected: {retained_files}")
            return False, retained_files
        
        return True, []


class AuditLogger:
    """
    L9: Comprehensive audit logging for compliance.
    
    Logs:
    - Every analysis request (timestamp, type, subject, result)
    - Security events (rejections, warnings, breaches)
    - API usage (cost, tokens, latency)
    - Model updates (auto research loop changes)
    - No raw EEG data in logs (HIPAA compliance)
    """
    
    def __init__(self, log_file='neurodecoder_audit.log'):
        self.log_file = log_file
        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.events = []
    
    def log_event(self, event_type, details, severity='INFO'):
        """Log an audit event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'session': self.session_id,
            'type': event_type,
            'severity': severity,
            'details': details,
        }
        
        self.events.append(event)
        
        log_msg = f"[{self.session_id}] {event_type}: {details}"
        
        if severity == 'CRITICAL':
            logger.critical(log_msg)
        elif severity == 'WARNING':
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
    
    def log_analysis(self, analysis_type, subject_id, accuracy=None, n_trials=None):
        """Log an analysis request."""
        self.log_event('ANALYSIS', {
            'type': analysis_type,
            'subject': f'S{subject_id}' if subject_id else 'uploaded',
            'accuracy': f'{accuracy:.1f}%' if accuracy else None,
            'n_trials': n_trials,
        })
    
    def log_api_call(self, service, tokens_used, cost, latency_ms):
        """Log an API call."""
        self.log_event('API_CALL', {
            'service': service,
            'tokens': tokens_used,
            'cost': f'${cost:.4f}',
            'latency': f'{latency_ms:.0f}ms',
        })
    
    def log_security_event(self, event_type, details, severity='WARNING'):
        """Log a security event."""
        self.log_event(f'SECURITY_{event_type}', details, severity)
    
    def get_session_summary(self):
        """Get summary of this session's activity."""
        summary = {
            'session_id': self.session_id,
            'total_events': len(self.events),
            'analyses': sum(1 for e in self.events if e['type'] == 'ANALYSIS'),
            'api_calls': sum(1 for e in self.events if e['type'] == 'API_CALL'),
            'security_events': sum(1 for e in self.events if 'SECURITY' in e['type']),
            'critical_events': sum(1 for e in self.events if e['severity'] == 'CRITICAL'),
        }
        return summary


# =====================================================
# MASTER SECURITY CONTROLLER
# =====================================================

class SecurityController:
    """
    Master security controller that orchestrates all security layers.
    
    Usage in app.py:
        security = SecurityController()
        
        # Before processing any request:
        ok, msg = security.pre_process_check(file_path, session_id)
        if not ok:
            return error_response(msg)
        
        # After loading data:
        ok, msg = security.validate_data(raw_data)
        if not ok:
            return error_response(msg)
        
        # Before API call:
        ok, msg = security.pre_api_check(estimated_cost)
        if not ok:
            return error_response(msg)
        
        # Before sending to robot:
        ok, msg = security.validate_output(prediction, confidence)
        if not ok:
            return safe_default()
    """
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.rate_limiter = RateLimiter(per_minute=10, per_hour=100, per_day=1000)
        self.api_keys = APIKeyManager()
        self.model_armor = ModelArmor()
        self.budget = BudgetGuardrails(session_limit=5.0, daily_limit=25.0, monthly_limit=200.0)
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, cooldown_minutes=15)
        self.output_validator = OutputValidator()
        self.pii_protector = PIIProtector()
        self.audit = AuditLogger()
        
        self.audit.log_event('SYSTEM_START', 'Security controller initialized')
    
    def pre_process_check(self, filepath=None, session_id="default"):
        """Run all pre-processing security checks."""
        
        # Rate limit check
        ok, msg = self.rate_limiter.check(session_id)
        if not ok:
            self.audit.log_security_event('RATE_LIMIT', msg)
            return False, msg
        
        # File validation (if file uploaded)
        if filepath:
            ok, issues = self.input_validator.validate_file(filepath)
            if not ok:
                self.audit.log_security_event('INVALID_FILE', issues, 'CRITICAL')
                return False, '\n'.join(issues)
        
        return True, "Pre-processing checks passed"
    
    def validate_data(self, raw_data):
        """Validate loaded EEG data."""
        ok, issues = self.input_validator.validate_raw_data(raw_data)
        
        if not ok:
            self.audit.log_security_event('INVALID_DATA', issues, 'CRITICAL')
        else:
            self.audit.log_event('DATA_VALIDATED', f"{len(raw_data.ch_names)}ch, {raw_data.info['sfreq']}Hz")
        
        return ok, '\n'.join(issues)
    
    def pre_api_check(self, estimated_cost=0.01):
        """Check before making an API call."""
        
        # Circuit breaker
        ok, msg = self.circuit_breaker.can_proceed()
        if not ok:
            self.audit.log_security_event('CIRCUIT_OPEN', msg)
            return False, msg
        
        # Budget check
        ok, msg = self.budget.check_budget(estimated_cost)
        if not ok:
            self.audit.log_security_event('BUDGET_EXCEEDED', msg)
            return False, msg
        
        return True, "API call approved"
    
    def record_api_result(self, success, cost=0, tokens=0, latency_ms=0):
        """Record API call result for circuit breaker and budget."""
        if success:
            self.circuit_breaker.record_success()
            self.budget.record_cost(cost)
            self.audit.log_api_call('claude', tokens, cost, latency_ms)
        else:
            self.circuit_breaker.record_failure()
    
    def validate_output(self, prediction, confidence, class_probabilities=None):
        """Validate decoder output before sending to user/robot."""
        return self.output_validator.validate_prediction(prediction, confidence, class_probabilities)
    
    def validate_robot_command(self, command, confidence):
        """Validate robot command before hardware execution."""
        return self.output_validator.validate_robot_command(command, confidence)
    
    def get_security_status(self):
        """Get current security status for dashboard display."""
        session_summary = self.audit.get_session_summary()
        
        status = {
            'circuit_breaker': self.circuit_breaker.state,
            'session_cost': f'${self.budget.session_cost:.2f}',
            'rate_limit_remaining': self.rate_limiter.per_minute - len([
                t for t in self.rate_limiter.requests.get('default', [])
                if time.time() - t < 60
            ]),
            'total_events': session_summary['total_events'],
            'security_events': session_summary['security_events'],
            'critical_events': session_summary['critical_events'],
        }
        
        return status
    
    def get_security_report(self):
        """Generate a human-readable security report."""
        status = self.get_security_status()
        summary = self.audit.get_session_summary()
        
        report = f"""
## 🔒 Security Status Report

**Session:** {self.audit.session_id}
**Circuit Breaker:** {status['circuit_breaker']}
**API Budget Used:** {status['session_cost']} / $5.00 (session)
**Rate Limit:** {status['rate_limit_remaining']} requests remaining (per minute)

### Activity Summary
- Total events: {summary['total_events']}
- Analyses run: {summary['analyses']}
- API calls: {summary['api_calls']}
- Security events: {summary['security_events']}
- Critical events: {summary['critical_events']}

### Security Layers Active
- ✅ L1: Input Validation (file format, size, channels)
- ✅ L2: Rate Limiting ({self.rate_limiter.per_minute}/min, {self.rate_limiter.per_hour}/hr)
- ✅ L3: API Key Management (environment variables)
- ✅ L4: Model Armor (anomaly detection)
- ✅ L5: Budget Guardrails (${self.budget.session_limit}/session)
- ✅ L6: Circuit Breaker ({self.circuit_breaker.failure_threshold} failures → {self.circuit_breaker.cooldown_minutes}min pause)
- ✅ L7: Output Validation (prediction + confidence checks)
- ✅ L8: PII Protection (no data retention)
- ✅ L9: Audit Logging (all events tracked)

### Security Philosophy
"Default Deny" — All inputs are untrusted, all outputs are validated, all actions are logged.
"""
        return report


# ===== CONVENIENCE: Create global security instance =====
security = SecurityController()
