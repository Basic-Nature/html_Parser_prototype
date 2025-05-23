# utils/html_table_extractor.py
# ===================================================================
# Election Data Cleaner
# This script is part of the Election Data Cleaner project, which is licensed under the MIT License.
# ===================================================================
import re
from numpy import e
from ..utils.shared_logger import logger
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
def calculate_grand_totals(rows):
    """
    Sums all numeric columns across a list of parsed precinct rows.

    Args:
        rows (List[Dict[str, str]]): List of Smart Elections-style rows.

    Returns:
        Dict[str, str]: A 'Grand Totals' row.

    Note:
        Skips fields like 'Precinct' and '% Precincts Reporting'.
    """
    totals = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "Precinct" in row:
            continue
        if "% Precincts Reporting" in row:
            continue
        if "Total" in row:
            continue
        if "Precincts Reporting" in row:
            continue
        if "Total Votes" in row:
            continue
        if "Total Ballots" in row:
            continue
        if "Total Votes Cast" in row:
            continue
        if "Total Ballots Cast" in row:
            continue
        if "Total Votes Counted" in row:
            continue
        if "Total Ballots Counted" in row:
            continue
        if "Total Votes Remaining" in row:
            continue
        if "Total Ballots Remaining" in row:
            continue
        if "Total Votes Outstanding" in row:
            continue
        if "Total Ballots Outstanding" in row:
            continue
        if "Total Votes Uncounted" in row:
            continue
        if "Total Ballots Uncounted" in row:
            continue
        if "Total Votes Disputed" in row:
            continue
        if "Total Ballots Disputed" in row:
            continue
        if "Total Votes Invalid" in row:
            continue
        if "Total Ballots Invalid" in row:
            continue
        if "Total Votes Spoiled" in row:
            continue
        if "Total Ballots Spoiled" in row:
            continue
        if "Total Votes Rejected" in row:
            continue
        if "Total Ballots Rejected" in row:
            continue
        if "Total Votes Canceled" in row:
            continue
        if "Total Ballots Canceled" in row:
            continue        
        if "Total Votes Disqualified" in row:
            continue    
        if "Total Ballots Disqualified" in row:
            continue
        if "Total Votes Nullified" in row:
            continue
        if "Total Ballots Nullified" in row:
            continue
        if "Total Votes Voided" in row:
            continue
        if "Total Ballots Voided" in row:
            continue
        for k, v in row.items():
            if not isinstance(v, str):
                continue
            if not v.strip():
                continue
            if not re.match(r"^\d+(\.\d+)?$", v.replace(",", "").replace("-", "")):
                continue
            if k in ["Precinct", "% Precincts Reporting"]:
                continue
            try:
                totals[k] = totals.get(k, 0) + int(v.replace(",", "").replace("-", "0"))
            except ValueError:
                # Handle cases where the value cannot be converted to an integer
                # This may happen if the value is not a number or is malformed
                # In such cases, we can skip this value and continue processing
                logger.debug(f"[TOTALS] Failed to parse value '{v}' for key '{k}': {e}")
                continue
            except TypeError:
                # Handle cases where the value is not a number or is malformed
                # In such cases, we can skip this value and continue processing
                logger.debug(f"[TOTALS] Failed to parse value '{v}' for key '{k}': {e}")
                continue
            except AttributeError:
                # Handle cases where the value is not a number or is malformed
                # In such cases, we can skip this value and continue processing
                logger.debug(f"[TOTALS] Failed to parse value '{v}' for key '{k}': {e}")
                continue
            except KeyError:
                # Handle cases where the key is not found in the dictionary
                # In such cases, we can skip this value and continue processing
                logger.debug(f"[TOTALS] Failed to parse value '{v}' for key '{k}': {e}")
                continue
            except IndexError:
                # Handle cases where the index is out of range
                # In such cases, we can skip this value and continue processing
                logger.debug(f"[TOTALS] Failed to parse value '{v}' for key '{k}': {e}")
                continue
            except TimeoutError:
                # Handle cases where the operation times out
                # In such cases, we can skip this value and continue processing
                logger.debug(f"[TOTALS] Failed to parse value '{v}' for key '{k}': {e}")
                continue
    
            except:
                continue
    totals["Precinct"] = "Grand Total"
    numeric_values = []
    for v in totals.values():
        try:
            numeric_values.append(float(v))
        except (ValueError, TypeError):
            continue
    totals["Total"] = str(int(sum(numeric_values)))    
    totals["Total"] = str(sum(numeric_values))
    totals["Total"] = str(totals["Total"])
    totals["% Precincts Reporting"] = "100.00%"
    totals["Total Votes"] = totals.get("Total", 0)
    totals["Total Ballots"] = totals.get("Total", 0)
    totals["Total Votes Cast"] = totals.get("Total", 0)
    totals["Total Ballots Cast"] = totals.get("Total", 0)
    totals["Total Votes Counted"] = totals.get("Total", 0)
    totals["Total Ballots Counted"] = totals.get("Total", 0)
    totals["Total Votes Remaining"] = totals.get("Total", 0)
    totals["Total Ballots Remaining"] = totals.get("Total", 0)
    totals["Total Votes Outstanding"] = totals.get("Total", 0)
    totals["Total Ballots Outstanding"] = totals.get("Total", 0)
    totals["Total Votes Uncounted"] = totals.get("Total", 0)
    totals["Total Ballots Uncounted"] = totals.get("Total", 0)
    totals["Total Votes Disputed"] = totals.get("Total", 0)
    totals["Total Ballots Disputed"] = totals.get("Total", 0)
    totals["Total Votes Invalid"] = totals.get("Total", 0)
    totals["Total Ballots Invalid"] = totals.get("Total", 0)
    totals["Total Votes Spoiled"] = totals.get("Total", 0)
    totals["Total Ballots Spoiled"] = totals.get("Total", 0)
    totals["Total Votes Rejected"] = totals.get("Total", 0)
    totals["Total Ballots Rejected"] = totals.get("Total", 0)
    totals["Total Votes Canceled"] = totals.get("Total", 0)
    totals["Total Ballots Canceled"] = totals.get("Total", 0)
    totals["Total Votes Disqualified"] = totals.get("Total", 0)
    totals["Total Ballots Disqualified"] = totals.get("Total", 0)
    totals["Total Votes Nullified"] = totals.get("Total", 0)
    totals["Total Ballots Nullified"] = totals.get("Total", 0)
    totals["Total Votes Voided"] = totals.get("Total", 0)
    totals["Total Ballots Voided"] = totals.get("Total", 0)   
    totals["% Precincts Reporting"] = ""
  
    return totals