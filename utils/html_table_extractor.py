# utils/html_table_extractor.py
# ===================================================================
# Election Data Cleaner
# This script is part of the Election Data Cleaner project, which is licensed under the MIT License.
# ===================================================================
from utils.shared_logger import logger
def extract_table_data(table):
    try:
        headers = [th.inner_text().strip() for th in table.query_selector_all("thead tr th")]
        if not headers:
            headers = [th.inner_text().strip() for th in table.query_selector_all("tbody tr:first-child th")]
        if not headers:
            raise RuntimeError("No headers found in the table.")
        data = []
        for row in table.query_selector_all("tbody tr"):
            if not row.query_selector("td"):
                continue
            cells = row.query_selector_all("td")
            row_data = {headers[i]: cells[i].inner_text().strip() for i in range(min(len(headers), len(cells)))}
            data.append(row_data)
        if not data:
            logger.info(f"[HTML Handler] Extracted {len(data)} rows from the table.")
            raise RuntimeError("No rows found in the table.")
    except RuntimeError as e:
        logger.error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}
    except AttributeError as e:
        logger.error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}
    except TypeError as e:
        logger.error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}
    except ValueError as e:
        logger.error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}
    except KeyError as e:
        logger.error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}
    except IndexError as e:
        logger.error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}
    except Exception as e:
        # Catch-all for any other exceptions
        # This is a generic error handler for any unexpected errors
        logger.error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}

    return headers, data