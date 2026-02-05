---
layout: default
title: Developer Guides & How-To
---

# Developer Guides & How-To

Comprehensive guides for developers extending and maintaining the Smart Elections Parser, including handler development, architecture, and integration patterns.

> **Note**: This document consolidates content from:
> - [HANDLER_MIGRATION_GUIDE.md](../HANDLER_MIGRATION_GUIDE.md) - Handler development patterns
> - [MODERN_UI_FEATURES.md](../MODERN_UI_FEATURES.md) - Frontend features
> - Architecture & Handler documentation
>
> For complete details, consult the source documents linked above.

## 🏗️ Architecture Overview

See [System Architecture](../CORE/ARCHITECTURE.md) for the complete architecture. Quick overview:

```
URL Input
  ↓
state_router → state-specific handler
    OR
format_router → format-specific handler
  ↓
html_context extraction
  ↓
table_core/dynamic_table_extractor
  ↓
table_builder, contest_selector
  ↓
CSV/JSON output + metadata
```

## 🛠️ Creating a New State Handler

### Step 1: Create Handler File

Create `webapp/handlers/states/your_state.py`:

```python
"""Handler for [State] election results."""

import logging
from typing import Tuple, List, Dict, Any
from utils.shared_logic import normalize_contested_office
from handlers.shared.templates import BaseStateHandler

logger = logging.getLogger(__name__)

class YourStateHandler(BaseStateHandler):
    """Parse election results from [State] sources."""
    
    def parse(self, page, html_context: dict) -> Tuple[List[str], List[Dict], Dict, Dict]:
        """
        Parse election results page.
        
        Args:
            page: Playwright page object
            html_context: Extracted context from page
            
        Returns:
            (headers, data_rows, contest, metadata)
        """
        # 1. Extract table from page
        table_html = page.locator('#results-table').inner_html()
        
        # 2. Parse table into rows
        data_rows = self._parse_table(table_html)
        
        # 3. Identify contests
        contest = self._identify_contests(data_rows)
        
        # 4. Generate metadata
        metadata = self._generate_metadata(page)
        
        return ['Name', 'Votes', 'Vote %'], data_rows, contest, metadata
    
    def _identify_contests(self, data):
        """Identify contests/races from parsed data."""
        # Implementation specific to state
        pass
```

### Step 2: Register Handler

Add to `webapp/handlers/__init__.py`:

```python
from handlers.states.your_state import YourStateHandler

HANDLERS = {
    'your_state': YourStateHandler(),
    ...
}
```

### Step 3: Register Routing Pattern

Add to `webapp/parser/state_router.py`:

```python
ROUTE_PATTERNS = {
    'your_state': [
        r'results\.your_state\.gov',
        r'elections\.your_state\.gov/results',
        # ... state-specific URL patterns
    ]
}
```

### Step 4: Test Handler

```python
# webapp/tests/test_your_state.py
import pytest
from handlers.states.your_state import YourStateHandler

class TestYourStateHandler:
    def test_parse_sample_page(self):
        handler = YourStateHandler()
        # Load sample HTML
        with open('tests/fixtures/your_state_sample.html') as f:
            html = f.read()
        
        # Mock page object
        # ... test implementation
        
        assert headers == ['Name', 'Votes', 'Vote %']
        assert len(data_rows) > 0
        assert contest is not None
```

Run test:
```bash
pytest webapp/tests/test_your_state.py::TestYourStateHandler::test_parse_sample_page
```

## 📋 Handler Development Checklist

Before submitting a new handler:

### Functionality
- [ ] Handler parses target election page correctly
- [ ] Returns proper tuple structure: (headers, data, contest, metadata)
- [ ] Handles edge cases: no data, partial data, format variations
- [ ] Works with multiple sample documents
- [ ] Proper error handling (non-fatal vs fatal)

### Code Quality
- [ ] Code follows project style guide
- [ ] Proper type hints on all functions
- [ ] Comprehensive docstrings
- [ ] No hardcoded values (use config/constants)
- [ ] Logging at appropriate levels
- [ ] No print statements (use logger)

### Testing
- [ ] Unit tests for parsing logic
- [ ] Integration tests with real/sample pages
- [ ] Edge case testing
- [ ] Test coverage > 80%
- [ ] All tests passing

### Documentation
- [ ] Handler documented in [handlers.md](../handlers.md)
- [ ] Known limitations documented
- [ ] Configuration options explained
- [ ] Sample output provided

### Security
- [ ] Input validation on all data sources
- [ ] No SQL injection risks
- [ ] No path traversal vulnerabilities
- [ ] Credentials properly managed (env vars)
- [ ] No sensitive data in logs

## 🎨 Frontend Development

### UI Framework

The parser uses a modern UI with:
- **Neon accent colors** (#00FF41, #FF006E)
- **Metallic backgrounds** (silver/aluminum tones)
- **CSS-in-JS** via `static/css/run_parser.css`
- **Responsive design** (mobile-first)

### JavaScript Modules

Key modules in `static/js/`:
- `run_parser.js` - Main application logic
- `quality_assurance_integration.js` - QA panel integration
- `form_handlers.js` - Form state management
- `result_display.js` - Results rendering

### Adding a Feature

1. **Create CSS classes** (no inline styles):
   ```css
   /* static/css/run_parser.css */
   @layer components;
   
   .my-feature {
     /* styling */
   }
   ```

2. **Add JavaScript handler**:
   ```javascript
   // static/js/my_feature.js
   export function initMyFeature() {
     // Initialize feature
   }
   ```

3. **Integrate in run_parser.js**:
   ```javascript
   import { initMyFeature } from './my_feature.js';
   initMyFeature();
   ```

## 🔌 Plugin Architecture

### Creating a Custom Plugin

```python
# plugins/my_extractor.py
from handlers.shared.plugin_interface import ExtractorPlugin

class MyExtractorPlugin(ExtractorPlugin):
    """Custom extraction logic."""
    
    def extract(self, html_content: str) -> List[Dict]:
        """Extract data from HTML."""
        # Custom extraction logic
        return extracted_rows
    
    @property
    def name(self) -> str:
        return "my_extractor"
    
    @property
    def priority(self) -> int:
        return 50  # Higher = tried first (0-100)
```

Register in `plugins/__init__.py`:
```python
from plugins.my_extractor import MyExtractorPlugin
PLUGINS = [MyExtractorPlugin()]
```

## 🔐 Context Integration

### Using HTML Context

The `html_context` parameter provides pre-extracted information:

```python
def parse(self, page, html_context: dict):
    # Pre-extracted tables
    tables = html_context.get('tables', [])
    
    # Identified regions
    candidate_section = html_context.get('candidate_section')
    
    # Detected metadata
    election_date = html_context.get('election_date')
    
    # Use to inform extraction
    if election_date:
        logger.info(f"Election date: {election_date}")
```

### Providing Context

From table_core:
```python
html_context = {
    'tables': extracted_tables,
    'candidate_section': detected_section,
    'confidence': extraction_confidence,
    'election_date': parsed_date,
    'metadata': parsed_metadata
}
```

## 📊 Testing Patterns

### Unit Test Template

```python
import pytest
from unittest.mock import Mock
from handlers.states.your_state import YourStateHandler

@pytest.fixture
def handler():
    return YourStateHandler()

@pytest.fixture
def mock_page():
    page = Mock()
    page.locator.return_value.inner_html.return_value = \
        "<table>...</table>"
    return page

def test_parse_returns_tuple(handler, mock_page):
    headers, data, contest, metadata = handler.parse(mock_page, {})
    
    assert isinstance(headers, list)
    assert isinstance(data, list)
    assert isinstance(contest, dict)
    assert isinstance(metadata, dict)
```

### Integration Test Template

```python
def test_parse_real_sample(handler):
    with open('tests/fixtures/state_sample.html') as f:
        html = f.read()
    
    # Create mock page
    page = create_mock_page(html)
    
    headers, data, contest, metadata = handler.parse(page, {})
    
    # Validate structure
    assert len(headers) > 0
    assert all(isinstance(row, dict) for row in data)
    
    # Sample assertions
    assert contest['office'] == 'Governor'
    assert contest['state'] == 'YourState'
```

## 🚀 Deployment

### Staging Testing
1. Deploy handler to staging environment
2. Test with 5-10 real election documents
3. Verify output quality
4. Collect timing metrics
5. Iterate if needed

### Production Deployment
1. Code review approval
2. All tests passing
3. Performance benchmarks acceptable
4. Documentation complete
5. Merge to main branch
6. Deploy via CI/CD

---

**Related Documents**:
- [System Architecture](../CORE/ARCHITECTURE.md) - Architecture overview
- [Data Models & Schema](../CORE/DATA_MODELS.md) - Data structures
- [Verification Framework](../QUALITY/VERIFICATION.md) - Testing & QA

**Sources**:
- [HANDLER_MIGRATION_GUIDE.md](../HANDLER_MIGRATION_GUIDE.md)
- [MODERN_UI_FEATURES.md](../MODERN_UI_FEATURES.md)

**Last Updated**: Consolidated developer guides
