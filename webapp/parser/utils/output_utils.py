import re
from ..utils.shared_logger import logger

def clean_candidate_name(name):
    """
    Cleans and normalizes candidate names:
    - Strips whitespace and punctuation
    - Handles suffixes (Jr., Sr., II, III, etc.)
    - Capitalizes names properly
    - Removes party abbreviations if attached
    """
    import re
    name = name.strip()
    # Remove common party abbreviations at the end (if present)
    name = re.sub(r'\b(DEM|REP|IND|LIB|GRE|CON|WFP|NPP|NPA|U|D|R|G|L|I|C|S|N|P|NP|NONPARTISAN|UNAFFILIATED)$', '', name, flags=re.IGNORECASE).strip()
    # Remove extra punctuation except hyphens and apostrophes
    name = re.sub(r"[^\w\s\-\']", '', name)
    # Handle suffixes
    suffixes = ['Jr', 'Sr', 'II', 'III', 'IV', 'V']
    parts = name.split()
    if parts and parts[-1].replace('.', '') in suffixes:
        suffix = parts.pop(-1)
        name = ' '.join(parts)
        name = f"{name} {suffix}"
    else:
        name = ' '.join(parts)
    # Proper capitalization (handles Mc/Mac, O', hyphens, etc.)
    def smart_cap(word):
        if word.lower().startswith("mc") and len(word) > 2:
            return "Mc" + word[2:].capitalize()
        if word.lower().startswith("mac") and len(word) > 3:
            return "Mac" + word[3:].capitalize()
        if "'" in word:
            return "'".join([w.capitalize() for w in word.split("'")])
        if "-" in word:
            return "-".join([w.capitalize() for w in word.split("-")])
        return word.capitalize()
    name = ' '.join(smart_cap(w) for w in name.split())
    return name

def parse_candidate_vote_table(table_element, current_precinct, method_names, reporting_pct="0.00%"):
    """
    Converts a DOM table element into a Smart Elections-style row for a single precinct.
    Args:
        table_element: Playwright DOM element (the <table>)
        current_precinct (str): Name of the precinct currently being processed.
        method_names (List[str]): List of vote methods detected.
        reporting_pct (str): The percentage reporting for this precinct.
    Returns:
        Dict[str, str]: Row with standardized candidate-method vote fields and metadata.
    """
    import re

    def is_total_row(cell_text):
        # List of phrases that indicate a summary/total row
        totals = {
            "total", "total votes", "total ballots", "total votes cast", "total ballots cast",
            "total votes counted", "total ballots counted", "total votes remaining", "total ballots remaining",
            "total votes outstanding", "total ballots outstanding", "total votes uncounted", "total ballots uncounted",
            "total votes disputed", "total ballots disputed", "total votes invalid", "total ballots invalid",
            "total votes spoiled", "total ballots spoiled", "total votes rejected", "total ballots rejected",
            "total votes canceled", "total ballots canceled", "total votes disqualified", "total ballots disqualified",
            "total votes nullified", "total ballots nullified", "total votes voided", "total ballots voided"
        }
        return cell_text.strip().lower() in totals

    row = {"Precinct": current_precinct, "% Precincts Reporting": reporting_pct}
    try:
        header_locator = table_element.locator('thead tr th')
        if header_locator.count() == 0:
            header_locator = table_element.locator('tbody tr:first-child th')
        row_locator = table_element.locator('tbody tr')
        for r_idx in range(row_locator.count()):
            r = row_locator.nth(r_idx)
            cells = [r.locator('td').nth(i) for i in range(r.locator('td').count())]
            if len(cells) < 2:
                continue
            full_name = cells[0].inner_text().strip()
            if is_total_row(full_name):
                continue

            # Candidate/party extraction
            name_parts = full_name.split()
            candidate_name, party = "", ""
            if len(name_parts) >= 3:
                candidate_name = " ".join(name_parts[1:-1])
                party = name_parts[-1]
            elif len(name_parts) == 2:
                candidate_name, party = name_parts
            elif len(name_parts) == 1 and not name_parts[0].isdigit():
                candidate_name = name_parts[0]
            else:
                candidate_name = full_name

            candidate_name = clean_candidate_name(candidate_name)
            party = party.strip().upper()
            if party in {"DEM", "REP", "IND", "LIB", "GRE"}:
                party = party.title()
            canonical = f"{candidate_name} ({party})" if party else candidate_name

            # Avoid duplicate candidates
            if canonical in row:
                continue

            # Extract method votes and total
            method_votes = [c.inner_text().strip().replace(",", "").replace("-", "0") for c in cells[1:-1]]
            total = cells[-1].inner_text().strip().replace(",", "").replace("-", "0")

            # Validate method votes
            if len(method_votes) != len(method_names):
                logger.debug(f"[TABLE] Number of method votes ({len(method_votes)}) does not match number of method names ({len(method_names)}).")
                continue
            if not re.match(r"^\d+(\.\d+)?$", total):
                logger.debug(f"[TABLE] Total '{total}' is not a valid number.")
                continue

            # Add method votes
            for method, vote in zip(method_names, method_votes):
                if not re.match(r"^\d+(\.\d+)?$", vote):
                    logger.debug(f"[TABLE] Vote '{vote}' is not a valid number.")
                    continue
                row[f"{canonical} - {method}"] = vote
            row[f"{canonical} - Total"] = total

    except Exception as e:
        logger.error(f"[TABLE] Failed to parse candidate vote table: {e}")
    return row