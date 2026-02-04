"""Prometheus metrics integration (optional).

Provides lightweight counters and optional Pushgateway pushes.
"""
import os
import threading
from typing import Dict

_ENABLED = os.environ.get('ENABLE_PROMETHEUS', 'false').lower() in ('1', 'true', 'yes')
_PUSHGATEWAY = os.environ.get('PROM_PUSHGATEWAY_URL')

_counters: Dict[str, object] = {}

try:
    if _ENABLED:
        from prometheus_client import Counter

        # Define counters in default registry
        _counters['processed_total'] = Counter('smart_processed_total', 'Total processed URLs')
        _counters['processed_success'] = Counter('smart_processed_success', 'Processed successful')
        _counters['processed_fail'] = Counter('smart_processed_fail', 'Processed failures')
        _counters['processed_partial'] = Counter('smart_processed_partial', 'Processed partial results')
        _counters['processed_cancelled'] = Counter('smart_processed_cancelled', 'Processed cancelled')
        _counters['fallbacks'] = Counter('smart_fallbacks_total', 'Fallback extractions')
        _counters['tables_seen_total'] = Counter('smart_tables_seen_total', 'Total tables seen')

        # Phase A: Decision Gate Metrics (Task 2)
        _counters['decision_proceed_total'] = Counter(
            'smart_decision_proceed_total',
            'Entities passed confidence checks (decision: PROCEED)',
            labelnames=['entity_type', 'reason', 'state']
        )
        _counters['decision_caution_total'] = Counter(
            'smart_decision_caution_total',
            'Entities with mixed signals requiring manual review (decision: CAUTION)',
            labelnames=['entity_type', 'reason', 'state']
        )
        _counters['decision_stop_total'] = Counter(
            'smart_decision_stop_total',
            'Entities failed confidence checks (decision: STOP)',
            labelnames=['entity_type', 'reason', 'state']
        )

        # Add a test-only counter for deterministic test increments
        _counters['test_metrics_increment_total'] = Counter('test_metrics_increment_total', 'Test-only: increments for /test/metrics/increment')
    else:
        # noop if not enabled
        pass
except Exception:
    _ENABLED = False

# Deterministic test increment function for /test/metrics/increment
def increment_test_counter():
    if not _ENABLED:
        return False
    try:
        c = _counters.get('test_metrics_increment_total')
        if c is None:
            return False
        c.inc()
        if _PUSHGATEWAY:
            _push_registry_async()
        return True
    except Exception:
        return False


def _push_registry_async():
    if not _PUSHGATEWAY:
        return
    try:
        from prometheus_client import push_to_gateway
        from prometheus_client.core import REGISTRY
        def _p():
            try:
                push_to_gateway(_PUSHGATEWAY, job='smart_parser', registry=REGISTRY)
            except Exception:
                pass
        t = threading.Thread(target=_p, daemon=True)
        t.start()
    except Exception:
        pass


def increment_prom_counter(name: str, amount: int = 1) -> None:
    if not _ENABLED:
        return
    try:
        c = _counters.get(name)
        if c is None:
            return
        c.inc(amount)
        # Push asynchronously if pushgateway configured
        if _PUSHGATEWAY:
            _push_registry_async()
    except Exception:
        pass
