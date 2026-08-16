"""
Test status reconciliation logic.
"""

from webapp.parser.utils.status_reconciliation import StatusReconciliation, WorklistParser


def test_parser_success_overrides_worklist():
    """Test 1: Parser status takes priority when present."""
    canonical, info = StatusReconciliation.reconcile(
        url='http://example.com/test1',
        parser_status='success',
        worklist_status='QC Loaded'
    )
    assert canonical == 'success', f"Expected 'success' but got '{canonical}'"
    assert info['authority'] == 'parser', f"Expected authority 'parser' but got '{info['authority']}'"
    print("✓ Test 1: Parser success overrides worklist")


def test_worklist_used_when_parser_missing():
    """Test 2: Worklist status used when parser hasn't processed."""
    canonical, info = StatusReconciliation.reconcile(
        url='http://example.com/test2',
        parser_status=None,
        worklist_status='PROD Loaded'
    )
    assert canonical == 'production', f"Expected 'production' but got '{canonical}'"
    assert info['authority'] == 'worklist', f"Expected authority 'worklist' but got '{info['authority']}'"
    print("✓ Test 2: Worklist used when parser missing")


def test_default_to_pending():
    """Test 3: Default to pending when no status tracked."""
    canonical, info = StatusReconciliation.reconcile(
        url='http://example.com/test3',
        parser_status=None,
        worklist_status=None
    )
    assert canonical == 'pending', f"Expected 'pending' but got '{canonical}'"
    assert info['authority'] == 'default', f"Expected authority 'default' but got '{info['authority']}'"
    print("✓ Test 3: Default to pending")


def test_skipped_data_exists_overrides_worklist():
    """Test 4: skipped_data_exists indicates already in production."""
    canonical, info = StatusReconciliation.reconcile(
        url='http://example.com/test4',
        parser_status='skipped_data_exists',
        worklist_status='QC Loaded',
        production_source='database'
    )
    assert canonical == 'skipped_data_exists', f"Expected 'skipped_data_exists' but got '{canonical}'"
    assert info['authority'] == 'parser', f"Expected authority 'parser' but got '{info['authority']}'"
    print("✓ Test 4: skipped_data_exists overrides worklist")


def test_pii_filtering():
    """Test 5: Personal names removed from public view."""
    original = {
        'Year': 2024,
        'Status': 'PROD Loaded',
        'Work in Progress - DL1': 'John Smith',
        'Work in Progress - DL2': 'Jane Doe',
        'State': 'Arizona'
    }
    sanitized = WorklistParser.sanitize_row(original)

    assert 'Work in Progress - DL1' not in sanitized, "PII column DL1 not removed"
    assert 'Work in Progress - DL2' not in sanitized, "PII column DL2 not removed"
    assert 'State' in sanitized, "Safe columns removed"
    assert sanitized['Year'] == 2024, "Data corrupted"
    print("✓ Test 5: PII filtering works")


def test_status_badge_info():
    """Test 6: Badge information is correct."""
    canonical, info = StatusReconciliation.reconcile(
        url='http://example.com/test6',
        parser_status=None,
        worklist_status='PROD Loaded'
    )

    assert 'icon' in info, "Missing 'icon' in status_info"
    assert 'label' in info, "Missing 'label' in status_info"
    assert 'badge_class' in info, "Missing 'badge_class' in status_info"
    assert info['icon'] == '📦', f"Expected icon '📦' but got '{info['icon']}'"
    assert info['label'] == 'Production', f"Expected label 'Production' but got '{info['label']}'"
    assert info['badge_class'] == 'success', f"Expected class 'success' but got '{info['badge_class']}'"
    print("✓ Test 6: Status badge information correct")


def test_status_requires_action():
    """Test 7: Identify statuses needing manual intervention."""
    assert StatusReconciliation.status_requires_action('fail') == True
    assert StatusReconciliation.status_requires_action('error') == True
    assert StatusReconciliation.status_requires_action('qc1_failed') == True
    assert StatusReconciliation.status_requires_action('download_needed') == True
    assert StatusReconciliation.status_requires_action('success') == False
    assert StatusReconciliation.status_requires_action('production') == False
    print("✓ Test 7: Status action requirements correct")


def test_status_is_complete():
    """Test 8: Identify complete statuses."""
    assert StatusReconciliation.status_is_complete('success') == True
    assert StatusReconciliation.status_is_complete('production') == True
    assert StatusReconciliation.status_is_complete('qc_complete') == True
    assert StatusReconciliation.status_is_complete('pending') == False
    assert StatusReconciliation.status_is_complete('fail') == False
    print("✓ Test 8: Status completion check correct")


if __name__ == '__main__':
    test_parser_success_overrides_worklist()
    test_worklist_used_when_parser_missing()
    test_default_to_pending()
    test_skipped_data_exists_overrides_worklist()
    test_pii_filtering()
    test_status_badge_info()
    test_status_requires_action()
    test_status_is_complete()
    print("\n✓✓✓ All reconciliation tests passed!")
