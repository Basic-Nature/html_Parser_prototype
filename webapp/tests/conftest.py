"""Pytest configuration and shared fixtures for all tests."""
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch
import importlib.machinery
import types

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Detect localhost/development environment for warning suppression
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
IS_LOCALHOST = POSTGRES_HOST in ("localhost", "127.0.0.1")

# Silence warnings in test environment when on localhost
if IS_LOCALHOST:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*eventlet.*")
    warnings.filterwarnings("ignore", message=".*socketio.*")

# Force DB utilities to use in-memory SQLite during tests (before any db_utils import)
os.environ.setdefault("TEST_SQLITE_URL", "sqlite:///:memory:")
# Avoid loading heavyweight ML dependencies during test collection/import.
os.environ.setdefault("DISABLE_SENTENCE_TRANSFORMERS", "1")

# Mock external openai dependency for tests (pre-import)
openai_mock: types.ModuleType = types.SimpleNamespace(__name__="openai")  # type: ignore[assignment]
openai_mock.__spec__ = importlib.machinery.ModuleSpec("openai", None)
sys.modules["openai"] = openai_mock

# Mock database connection before importing config
with patch('webapp.parser.utils.db_utils.SessionLocal', Mock()):
    from webapp.parser.config import OUTPUT_DIR, UPLOADS_DIR
    from webapp.parser.utils.models import Base


@pytest.fixture(scope="session")
def test_db_engine():
    """Create a temporary test database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(test_db_engine) -> Generator[Session, None, None]:
    """Provide a transactional test database session."""
    from sqlalchemy.orm import sessionmaker
    SessionMaker = sessionmaker(bind=test_db_engine)
    connection = test_db_engine.connect()
    transaction = connection.begin()
    session = SessionMaker(bind=connection)
    
    yield session
    
    session.close()
    if transaction.is_active:
        transaction.rollback()
    connection.close()


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary output directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_html_content() -> str:
    """Sample HTML content for testing."""
    return """
    <html>
    <body>
        <h1>2024 General Election Results</h1>
        <table>
            <thead>
                <tr><th>Candidate</th><th>Votes</th><th>Percent</th></tr>
            </thead>
            <tbody>
                <tr><td>John Doe</td><td>12,500</td><td>55.2%</td></tr>
                <tr><td>Jane Smith</td><td>10,100</td><td>44.8%</td></tr>
            </tbody>
        </table>
    </body>
    </html>
    """


@pytest.fixture
def sample_csv_data() -> list[dict]:
    """Sample CSV data for testing."""
    return [
        {"Candidate": "John Doe", "Party": "Democratic", "Votes": "12500", "Percent": "55.2%"},
        {"Candidate": "Jane Smith", "Party": "Republican", "Votes": "10100", "Percent": "44.8%"},
    ]


@pytest.fixture
def sample_contest_data() -> dict:
    """Sample contest data."""
    return {
        "title": "U.S. Representative District 3",
        "year": 2024,
        "type_": "General",
        "state": "New York",
        "county": "Rockland"
    }


@pytest.fixture
def mock_coordinator():
    """Mock ContextCoordinator for testing."""
    coordinator = Mock()
    coordinator.extract_entities = Mock(return_value=[("New York", "GPE"), ("John Doe", "PERSON")])
    coordinator.get_state_county_patterns = Mock(return_value={})
    return coordinator


@pytest.fixture
def mock_page():
    """Mock Playwright page object."""
    page = Mock()
    page.url = "https://example.com/election-results"
    page.content = Mock(return_value="<html><body>Test content</body></html>")
    page.query_selector_all = Mock(return_value=[])
    return page
