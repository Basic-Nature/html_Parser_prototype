from webapp.parser.handlers.formats.pdf_handler import parse_pdf_election_results

pdf_path = r"c:\Users\edu-loaner\html_Parser_prototype\uploads\Democratic District Attorney New York 2025.pdf"
headers, data, contest, metadata = parse_pdf_election_results(pdf_path, session_id="debug_run", coordinator=None)
print("contest:", contest)
print("rows:", len(data))
print("columns:", len(headers))
print("first row keys:", list(data[0].keys())[:10])
print("metadata keys:", [k for k in metadata.keys() if k.startswith("statement")])
print("statement info:", metadata.get("statement_blocks_available"))
print("decision:", metadata.get("statement_blocks_decision"))
print("used:", metadata.get("statement_blocks_used"))
print("all keys sample:", list(metadata.keys())[:15])
print("layout rows:", metadata.get("layout_table_rows"), metadata.get("layout_table_available_rows"))
