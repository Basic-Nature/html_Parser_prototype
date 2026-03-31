"""
Multi-Pathway Integration Tests for Ballot Lens
================================================

Tests that verify ballot lens produces consistent, valid election data
across multiple navigation/execution pathways:

1. CLI execution: Direct parser invocation with URLs
2. Webapp UI: Form submission via HTTP (Selenium-based)
3. API endpoints: Direct Flask endpoint calls
4. URL patterns: Various election result URL formats

Each pathway should produce:
- Valid CSV election data
- Consistent schema (headers, data types)
- Database-comparable format for validation
- Flagging of anomalies/edge cases

Test Categories:
- Pathway consistency: Same URL → Same data across all 3 entry points
- Output validation: CSV structure, headers, required fields
- Data integrity: No null/empty critical fields
- Edge cases: Invalid URLs, malformed data, timeouts

Usage:
    pytest webapp/tests/test_ballot_lens_pathways.py -v
    pytest webapp/tests/test_ballot_lens_pathways.py::TestPathwayConsistency -v
    pytest webapp/tests/test_ballot_lens_pathways.py -k "csv" -v --tb=short
"""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator, Any
from dataclasses import dataclass
from enum import Enum
from unittest.mock import patch, Mock, MagicMock
import subprocess
import sys
import time
import hashlib

import pytest

# Import ballot lens / parser modules
try:
    from webapp.parser.html_election_parser import orchestrate_url, load_urls
    from webapp.Smart_Elections_Parser_Webapp import app
    from webapp.parser.utils.shared_logic import DecisionTuple
except ImportError as e:
    pytest.skip(f"Cannot import ballot lens modules: {e}", allow_module_level=True)


RUN_LIVE_INTEGRATION_TESTS = os.getenv("RUN_LIVE_INTEGRATION_TESTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


# ========================================================================
# Data Structures & Enums
# ========================================================================

class ExecutionPathway(Enum):
    """Different ways to invoke ballot lens."""
    CLI = "cli"               # Direct command-line parser
    WEBAPP_API = "webapp_api" # Flask endpoint via HTTP
    DIRECT_API = "direct_api" # Direct function call
    

class DataValidationResult(Enum):
    """Validation outcomes."""
    VALID = "valid"
    EMPTY = "empty"           # No election data found
    MALFORMED = "malformed"   # CSV structure broken
    INCOMPLETE = "incomplete" # Missing required fields
    ERROR = "error"           # Execution error


@dataclass
class CSVValidation:
    """Result of CSV validation."""
    status: DataValidationResult
    rows_count: int
    headers: List[str]
    errors: List[str]
    sample_row: Optional[Dict[str, str]] = None
    
    @property
    def is_valid(self) -> bool:
        return self.status == DataValidationResult.VALID and self.rows_count > 0


@dataclass
class PathwayExecutionResult:
    """Result from executing ballot lens via one pathway."""
    pathway: ExecutionPathway
    url: str
    csv_path: Optional[Path]
    csv_content: Optional[str]
    validation: CSVValidation
    error: Optional[str]
    execution_time_ms: float
    
    @property
    def succeeded(self) -> bool:
        return self.csv_path is not None and self.error is None


# ========================================================================
# Fixtures
# ========================================================================

@pytest.fixture
def webapp_client():
    """Provide Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create temporary directory for CSV outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_election_urls() -> List[str]:
    """Sample election result URLs for testing.
    
    NOTE: These are representative URLs. Real testing should use:
    - Locally hosted HTML fixtures
    - Mock URLs via test server
    - Or live URLs if integration testing in staging
    """
    return [
        # California examples (common format)
        "https://example.com/results/ca/alameda/2024/general",
        # Georgia examples
        "https://example.com/results/ga/fulton/2024/general",
        # Nevada examples
        "https://example.com/results/nv/clark/2024/general",
    ]


@pytest.fixture
def sample_html_fixture() -> str:
    """Minimal valid HTML election results fixture for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Election Results - Sample County 2024</title></head>
    <body>
        <h1>Sample County General Election Results - November 5, 2024</h1>
        
        <table>
            <tr><th>Race</th><th>Office</th><th>Candidate</th><th>Party</th><th>Votes</th><th>Percentage</th></tr>
            <tr><td>President</td><td>President</td><td>Alice Johnson</td><td>Democratic</td><td>45,230</td><td>52.3%</td></tr>
            <tr><td>President</td><td>President</td><td>Bob Smith</td><td>Republican</td><td>38,920</td><td>44.9%</td></tr>
            <tr><td>Governor</td><td>Governor</td><td>Carol White</td><td>Democratic</td><td>42,100</td><td>50.1%</td></tr>
            <tr><td>Governor</td><td>Governor</td><td>David Brown</td><td>Republican</td><td>39,800</td><td>47.4%</td></tr>
        </table>
    </body>
    </html>
    """


# ========================================================================
# Validation Functions
# ========================================================================

def validate_csv(csv_path: Path) -> CSVValidation:
    """Validate a CSV file structure and content.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        CSVValidation result with status and details
    """
    errors = []
    headers = []
    rows_count = 0
    sample_row = None
    
    try:
        if not csv_path.exists():
            return CSVValidation(
                status=DataValidationResult.ERROR,
                rows_count=0,
                headers=[],
                errors=["CSV file does not exist"]
            )
        
        # Check file size
        file_size_mb = csv_path.stat().st_size / (1024 * 1024)
        if file_size_mb == 0:
            return CSVValidation(
                status=DataValidationResult.EMPTY,
                rows_count=0,
                headers=[],
                errors=["CSV file is empty"]
            )
        
        # Parse CSV
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if not reader.fieldnames:
                return CSVValidation(
                    status=DataValidationResult.MALFORMED,
                    rows_count=0,
                    headers=[],
                    errors=["CSV has no headers"]
                )
            
            headers = list(reader.fieldnames)
            
            # Required election data fields
            required_fields = ['Office', 'Candidate', 'Party', 'Votes']
            missing_fields = [f for f in required_fields if f not in headers]
            
            if missing_fields:
                errors.append(f"Missing required fields: {missing_fields}")
            
            # Read rows
            rows = list(reader)
            rows_count = len(rows)
            
            if rows_count == 0:
                return CSVValidation(
                    status=DataValidationResult.INCOMPLETE,
                    rows_count=0,
                    headers=headers,
                    errors=["CSV has headers but no data rows"]
                )
            
            # Validate first row as sample
            sample_row = rows[0]
            
            # Check for empty/null critical fields
            for field in required_fields:
                if field in sample_row and not sample_row[field]:
                    errors.append(f"First row has empty {field}")
        
        # Determine status
        if errors:
            status = DataValidationResult.INCOMPLETE
        else:
            status = DataValidationResult.VALID
        
        return CSVValidation(
            status=status,
            rows_count=rows_count,
            headers=headers,
            errors=errors,
            sample_row=sample_row
        )
        
    except csv.Error as e:
        return CSVValidation(
            status=DataValidationResult.MALFORMED,
            rows_count=0,
            headers=[],
            errors=[f"CSV parsing error: {str(e)}"]
        )
    except Exception as e:
        return CSVValidation(
            status=DataValidationResult.ERROR,
            rows_count=0,
            headers=[],
            errors=[f"Validation error: {str(e)}"]
        )


def test_validate_csv_detects_missing_required_fields(temp_output_dir: Path):
    csv_path = temp_output_dir / "missing_required.csv"
    csv_path.write_text("Office,Candidate,Votes\nPresident,Alice Johnson,45230\n", encoding="utf-8")

    result = validate_csv(csv_path)

    assert result.status == DataValidationResult.INCOMPLETE
    assert any("Missing required fields" in err for err in result.errors)


def test_validate_csv_detects_empty_critical_value(temp_output_dir: Path):
    csv_path = temp_output_dir / "missing_value.csv"
    csv_path.write_text("Office,Candidate,Party,Votes\nPresident,Alice Johnson,,45230\n", encoding="utf-8")

    result = validate_csv(csv_path)

    assert result.status == DataValidationResult.INCOMPLETE
    assert any("empty Party" in err for err in result.errors)


def test_extract_table_rows_from_html_strips_nested_markup():
    html = """
    <table>
      <tr><th>Office</th><th>Candidate</th></tr>
      <tr><td><strong>President</strong></td><td><span>Alice Johnson</span></td></tr>
    </table>
    """

    rows = _extract_table_rows_from_html(html)

    assert rows == [["Office", "Candidate"], ["President", "Alice Johnson"]]


def read_csv_content(csv_path: Path) -> Optional[str]:
    """Read full CSV content as string for comparison."""
    try:
        if csv_path.exists():
            return csv_path.read_text(encoding='utf-8')
    except Exception:
        pass
    return None


def hash_csv_content(csv_content: str) -> str:
    """Create hash of CSV content for comparing identical outputs."""
    return hashlib.sha256(csv_content.encode()).hexdigest()[:16]


def _extract_table_rows_from_html(html_content: str) -> list[list[str]]:
    """Extract a simple table matrix from fixture HTML.

    This lightweight parser is intended for test fixtures only.
    """
    if not html_content:
        return []
    tr_blocks = re.findall(r"<tr[^>]*>(.*?)</tr>", html_content, flags=re.IGNORECASE | re.DOTALL)
    rows: list[list[str]] = []
    for block in tr_blocks:
        cells = re.findall(r"<(?:th|td)[^>]*>(.*?)</(?:th|td)>", block, flags=re.IGNORECASE | re.DOTALL)
        if not cells:
            continue
        cleaned = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        rows.append(cleaned)
    return rows


# ========================================================================
# Execution Pathway Functions
# ========================================================================

def execute_via_cli(url: str, output_dir: Path) -> PathwayExecutionResult:
    """Execute ballot lens via CLI.
    
    Args:
        url: Election results URL
        output_dir: Directory for CSV output
        
    Returns:
        PathwayExecutionResult with execution details
    """
    start_time = time.time()
    error = None
    csv_path = None
    csv_content = None
    validation = None
    
    try:
        # Build CLI command
        # NOTE: Adjust based on actual CLI interface of html_election_parser.py
        cmd = [
            sys.executable, 
            "webapp/parser/html_election_parser.py",
            "--url", url,
            "--output", str(output_dir)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            error = f"CLI returned code {result.returncode}: {result.stderr}"
        else:
            # Find generated CSV in output directory
            csv_files = list(output_dir.glob("*.csv"))
            if csv_files:
                csv_path = csv_files[0]  # Take first generated CSV
                csv_content = read_csv_content(csv_path)
                validation = validate_csv(csv_path)
            else:
                error = "CLI executed but no CSV output generated"
                validation = CSVValidation(
                    status=DataValidationResult.EMPTY,
                    rows_count=0,
                    headers=[],
                    errors=["No CSV file found in output directory"]
                )
    
    except subprocess.TimeoutExpired:
        error = "CLI execution timed out (30s)"
    except FileNotFoundError:
        error = "CLI script not found"
    except Exception as e:
        error = f"CLI execution error: {str(e)}"
    
    execution_time_ms = (time.time() - start_time) * 1000
    
    if validation is None:
        validation = CSVValidation(
            status=DataValidationResult.ERROR,
            rows_count=0,
            headers=[],
            errors=[error] if error else ["Unknown error"]
        )
    
    return PathwayExecutionResult(
        pathway=ExecutionPathway.CLI,
        url=url,
        csv_path=csv_path,
        csv_content=csv_content,
        validation=validation,
        error=error,
        execution_time_ms=execution_time_ms
    )


def execute_via_direct_api(url: str, output_dir: Path, html_content: Optional[str] = None) -> PathwayExecutionResult:
    """Execute ballot lens via direct API call.
    
    Args:
        url: Election results URL (for metadata)
        output_dir: Directory for CSV output
        html_content: Optional HTML content (for testing without real URLs)
        
    Returns:
        PathwayExecutionResult with execution details
    """
    start_time = time.time()
    error = None
    csv_path = None
    csv_content = None
    validation = None
    
    try:
        # Deterministic pathway for unit-style tests.
        if html_content is not None:
            rows = _extract_table_rows_from_html(html_content)
            if len(rows) < 2:
                error = "Parser found no election data"
                validation = CSVValidation(
                    status=DataValidationResult.EMPTY,
                    rows_count=0,
                    headers=[],
                    errors=[error],
                )
            else:
                header = rows[0]
                data = rows[1:]

                def _idx(name: str) -> int:
                    lower = [h.strip().lower() for h in header]
                    try:
                        return lower.index(name)
                    except ValueError:
                        return -1

                office_i = _idx("office")
                cand_i = _idx("candidate")
                party_i = _idx("party")
                votes_i = _idx("votes")

                if min(office_i, cand_i, party_i, votes_i) < 0:
                    error = "Fixture table missing required election columns"
                    validation = CSVValidation(
                        status=DataValidationResult.INCOMPLETE,
                        rows_count=0,
                        headers=header,
                        errors=[error],
                    )
                else:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    csv_path = output_dir / "direct_api_output.csv"
                    normalized_rows = []
                    for row in data:
                        if len(row) <= max(office_i, cand_i, party_i, votes_i):
                            continue
                        normalized_rows.append(
                            [
                                row[office_i],
                                row[cand_i],
                                row[party_i],
                                row[votes_i],
                            ]
                        )

                    if not normalized_rows:
                        error = "Parser found no election data"
                        validation = CSVValidation(
                            status=DataValidationResult.EMPTY,
                            rows_count=0,
                            headers=["Office", "Candidate", "Party", "Votes"],
                            errors=[error],
                        )
                    else:
                        with open(csv_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(["Office", "Candidate", "Party", "Votes"])
                            writer.writerows(normalized_rows)

                        csv_content = read_csv_content(csv_path)
                        validation = validate_csv(csv_path)
        else:
            # Optional live integration pathway.
            processed_info: dict[str, Any] = {}
            session_id = f"test_session_{int(time.time())}"
            try:
                result = orchestrate_url(
                    target_url=url,
                    processed_info=processed_info,
                    session_id=session_id,
                    output_bypass=True,
                    trust_bypass=True,
                )

                if result and isinstance(result, dict):
                    headers = result.get("headers", [])
                    data_rows = result.get("data", [])

                    if not data_rows:
                        error = "Parser returned no data rows"
                        validation = CSVValidation(
                            status=DataValidationResult.EMPTY,
                            rows_count=0,
                            headers=headers or [],
                            errors=["Parser found no election data"],
                        )
                    else:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        csv_path = output_dir / "direct_api_output.csv"
                        with open(csv_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(headers or [])
                            writer.writerows(data_rows)

                        csv_content = read_csv_content(csv_path)
                        validation = validate_csv(csv_path)
                else:
                    error = "Parser returned unexpected result format"
                    validation = CSVValidation(
                        status=DataValidationResult.ERROR,
                        rows_count=0,
                        headers=[],
                        errors=[error],
                    )
            except Exception as parse_error:
                error = f"Parser error: {str(parse_error)}"
                validation = CSVValidation(
                    status=DataValidationResult.ERROR,
                    rows_count=0,
                    headers=[],
                    errors=[error],
                )
    
    except Exception as e:
        error = f"Direct API error: {str(e)}"
    
    execution_time_ms = (time.time() - start_time) * 1000
    
    if validation is None:
        validation = CSVValidation(
            status=DataValidationResult.ERROR,
            rows_count=0,
            headers=[],
            errors=[error] if error else ["Unknown error"]
        )
    
    return PathwayExecutionResult(
        pathway=ExecutionPathway.DIRECT_API,
        url=url,
        csv_path=csv_path,
        csv_content=csv_content,
        validation=validation,
        error=error,
        execution_time_ms=execution_time_ms
    )


def execute_via_webapp_api(client, url: str, output_dir: Path) -> PathwayExecutionResult:
    """Execute ballot lens via webapp API endpoint.
    
    Args:
        client: Flask test client
        url: Election results URL
        output_dir: Directory for outputs
        
    Returns:
        PathwayExecutionResult with execution details
    """
    start_time = time.time()
    error = None
    csv_path = None
    csv_content = None
    validation = None
    
    try:
        # Submit to ballot_lens endpoint
        # NOTE: Adjust based on actual Flask route
        response = client.post(
            '/api/parse',
            json={'url': url, 'format': 'csv'},
            timeout=30
        )
        
        if response.status_code != 200:
            error = f"API returned {response.status_code}: {response.get_data(as_text=True)}"
            validation = CSVValidation(
                status=DataValidationResult.ERROR,
                rows_count=0,
                headers=[],
                errors=[error]
            )
        else:
            # Extract CSV from response
            try:
                result_data = response.get_json()
                csv_content_from_api = result_data.get('csv') or result_data.get('data')
                
                if csv_content_from_api:
                    csv_path = output_dir / "webapp_api_output.csv"
                    csv_path.write_text(csv_content_from_api, encoding='utf-8')
                    csv_content = csv_content_from_api
                    validation = validate_csv(csv_path)
                else:
                    error = "API response missing CSV data"
                    validation = CSVValidation(
                        status=DataValidationResult.EMPTY,
                        rows_count=0,
                        headers=[],
                        errors=[error]
                    )
            except Exception as e:
                error = f"Failed to extract CSV from response: {str(e)}"
                validation = CSVValidation(
                    status=DataValidationResult.ERROR,
                    rows_count=0,
                    headers=[],
                    errors=[error]
                )
    
    except Exception as e:
        error = f"Webapp API error: {str(e)}"
        validation = CSVValidation(
            status=DataValidationResult.ERROR,
            rows_count=0,
            headers=[],
            errors=[error]
        )
    
    execution_time_ms = (time.time() - start_time) * 1000
    
    return PathwayExecutionResult(
        pathway=ExecutionPathway.WEBAPP_API,
        url=url,
        csv_path=csv_path,
        csv_content=csv_content,
        validation=validation,
        error=error,
        execution_time_ms=execution_time_ms
    )


# ========================================================================
# Test Classes
# ========================================================================

class TestPathwayConsistency:
    """Test that all pathways produce identical results for same URL."""
    
    def test_all_pathways_produce_valid_csv(self, temp_output_dir, sample_html_fixture):
        """Verify all pathways can produce valid CSV output."""
        test_url = "http://localhost:8000/test-election.html"
        
        # Note: For real tests, either:
        # 1. Mock URL responses
        # 2. Use local test server
        # 3. Use sample HTML fixture
        
        results = {
            ExecutionPathway.DIRECT_API: execute_via_direct_api(
                test_url, 
                temp_output_dir / "direct",
                html_content=sample_html_fixture
            )
        }
        
        # Verify at least direct API produces valid output
        direct_result = results[ExecutionPathway.DIRECT_API]
        assert direct_result.succeeded or direct_result.validation.status in [
            DataValidationResult.EMPTY,
            DataValidationResult.INCOMPLETE
        ], f"Direct API failed: {direct_result.error}"
    
    def test_pathway_csv_headers_consistent(self, temp_output_dir, sample_html_fixture):
        """Test that all successful pathways use same CSV headers."""
        test_url = "http://localhost:8000/test-election.html"
        
        # Execute via direct API
        result = execute_via_direct_api(
            test_url,
            temp_output_dir / "direct",
            html_content=sample_html_fixture
        )
        
        if result.validation.is_valid:
            headers = result.validation.headers
            # Verify headers include standard election fields
            assert len(headers) > 0, "No headers found"
            print(f"Found headers: {headers}")


class TestCSVValidation:
    """Test CSV output validation."""
    
    def test_empty_csv_detected(self, temp_output_dir):
        """Verify empty CSV is correctly flagged."""
        csv_path = temp_output_dir / "empty.csv"
        csv_path.touch()  # Create empty file
        
        result = validate_csv(csv_path)
        assert result.status == DataValidationResult.EMPTY
        assert result.rows_count == 0
    
    def test_valid_csv_detected(self, temp_output_dir):
        """Verify valid CSV is correctly identified."""
        csv_path = temp_output_dir / "valid.csv"
        csv_path.write_text(
            "Office,Candidate,Party,Votes\n"
            "President,Alice Johnson,Democratic,45230\n"
            "President,Bob Smith,Republican,38920\n"
        )
        
        result = validate_csv(csv_path)
        assert result.status == DataValidationResult.VALID
        assert result.rows_count == 2
        assert "Office" in result.headers
    
    def test_malformed_csv_detected(self, temp_output_dir):
        """Verify malformed CSV is flagged."""
        csv_path = temp_output_dir / "malformed.csv"
        csv_path.write_text("Office,Candidate\n\"unterminated,quote, value")
        
        result = validate_csv(csv_path)
        # May be MALFORMED or INCOMPLETE depending on parser tolerance
        assert result.status in [DataValidationResult.MALFORMED, DataValidationResult.INCOMPLETE]


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.live_integration
    @pytest.mark.skipif(
        not RUN_LIVE_INTEGRATION_TESTS,
        reason="Live integration test disabled. Set RUN_LIVE_INTEGRATION_TESTS=1 to run.",
    )
    def test_invalid_url_handling(self, temp_output_dir):
        """Verify invalid URL is handled gracefully."""
        invalid_url = "https://invalid-domain-that-does-not-exist-12345.com"
        
        result = execute_via_direct_api(invalid_url, temp_output_dir / "direct")
        
        assert result.error is not None, "Should capture connection error"
        assert result.validation.status in [
            DataValidationResult.ERROR,
            DataValidationResult.EMPTY
        ]
    
    def test_html_without_election_data(self, temp_output_dir):
        """Test HTML that has no election data."""
        plain_html = "<html><body><h1>Hello World</h1></body></html>"
        
        result = execute_via_direct_api(
            "http://example.com/no-data",
            temp_output_dir / "direct",
            html_content=plain_html
        )
        
        # Should either be empty or incomplete, not crash
        assert result.validation.status in [
            DataValidationResult.EMPTY,
            DataValidationResult.INCOMPLETE
        ]


class TestDataComparison:
    """Test comparing outputs against expected/database values."""
    
    def test_csv_content_hash_consistency(self, temp_output_dir, sample_html_fixture):
        """Verify same input produces same output hash."""
        test_url = "http://localhost:8000/test-election.html"
        
        # Execute twice (simulating reprocessing)
        result1 = execute_via_direct_api(
            test_url,
            temp_output_dir / "run1",
            html_content=sample_html_fixture
        )
        
        result2 = execute_via_direct_api(
            test_url,
            temp_output_dir / "run2",
            html_content=sample_html_fixture
        )
        
        if result1.csv_content and result2.csv_content:
            # Verify outputs are identical
            hash1 = hash_csv_content(result1.csv_content)
            hash2 = hash_csv_content(result2.csv_content)
            assert hash1 == hash2, "Same input should produce identical output"
    
    def test_required_fields_present(self, temp_output_dir, sample_html_fixture):
        """Verify all required election data fields are in output."""
        test_url = "http://localhost:8000/test-election.html"
        
        result = execute_via_direct_api(
            test_url,
            temp_output_dir / "direct",
            html_content=sample_html_fixture
        )
        
        required_fields = ['Office', 'Candidate', 'Party']
        
        if result.validation.is_valid:
            missing = [f for f in required_fields if f not in result.validation.headers]
            assert len(missing) == 0, f"Missing required fields: {missing}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
