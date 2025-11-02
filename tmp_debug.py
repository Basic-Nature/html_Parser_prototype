from webapp.parser.handlers.formats.pdf_handler import _pdf_to_images
import pytesseract

pdf_path = r"c:\Users\edu-loaner\html_Parser_prototype\uploads\Democratic District Attorney New York 2025.pdf"
images = _pdf_to_images(pdf_path, dpi=400, max_pages=10)
for idx, image in enumerate(images):
	text = pytesseract.image_to_string(image, config="--oem 3 --psm 4")
	lines = [l.strip() for l in text.splitlines() if l.strip()]
	total_votes = [l for l in lines if l.lower().startswith('total votes')]
	print(f"page {idx}: total vote occurrences {len(total_votes)} -> {total_votes[:5]}")
