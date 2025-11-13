import argparse
import pathlib

from webapp.parser.handlers.formats.pdf_handler import parse_pdf_election_results


DEFAULT_PDF = pathlib.Path(
	r"c:\Users\edu-loaner\html_Parser_prototype\uploads\Democratic District Attorney New York 2025.pdf"
)


def _describe_result(pdf_path: pathlib.Path, headers, data, metadata, contest, show_recon_debug: bool = False):
	print(f"\n=== {pdf_path} ===")
	print("contest:", contest)
	print("rows:", len(data))
	print("columns:", len(headers))
	if data:
		print("first row keys:", list(data[0].keys())[:10])
	statement_keys = [k for k in metadata.keys() if k.startswith("statement")]
	print("metadata keys:", statement_keys)
	print("statement info:", metadata.get("statement_blocks_available"))
	print("decision:", metadata.get("statement_blocks_decision"))
	print("used:", metadata.get("statement_blocks_used"))
	print("all keys sample:", list(metadata.keys())[:15])
	print(
		"layout rows:",
		metadata.get("layout_table_rows"),
		metadata.get("layout_table_available_rows"),
	)
	columnar_meta = metadata.get("columnar_reconstruction") or {}
	if show_recon_debug and columnar_meta:
		debug_events = columnar_meta.get("debug_events") or []
		print("columnar scope:", columnar_meta.get("scope"))
		print("columnar rows (wide/final):", columnar_meta.get("rows"), columnar_meta.get("final_rows"))
		if debug_events:
			print("-- reconstruction debug events --")
			for event in debug_events:
				print(event)
		else:
			print("-- no reconstruction debug events captured --")


def main():
	parser = argparse.ArgumentParser(
		description="Run a quick parse of one or more PDF statement files."
	)
	parser.add_argument(
		"pdf",
		nargs="*",
		help="Path(s) to PDF file(s). If omitted the default test PDF is used.",
	)
	parser.add_argument(
		"--show-recon-debug",
		action="store_true",
		help="Print columnar reconstruction debug events (requires SMART_ELECTIONS_RECON_DEBUG=1)",
	)
	args = parser.parse_args()

	pdf_paths = [pathlib.Path(p) for p in (args.pdf or [DEFAULT_PDF])]

	for pdf_path in pdf_paths:
		if not pdf_path.exists():
			print(f"Skipping {pdf_path}: file not found.")
			continue
		headers, data, contest, metadata = parse_pdf_election_results(
			str(pdf_path),
			session_id="debug_run",
			coordinator=None,
		)
		_describe_result(pdf_path, headers, data, metadata, contest, show_recon_debug=args.show_recon_debug)


if __name__ == "__main__":
	main()
