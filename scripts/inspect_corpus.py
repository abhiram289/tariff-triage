import pypdf
import sys
from pathlib import Path

corpus_dir = Path("data/corpus")
keywords = ["tailor", "mixed use", "mixed load", "cottage", "consulting room", "professional", "residential", "lt-2", "lt-3", "conditions of supply"]

for pdf_path in corpus_dir.glob("*.pdf"):
    print(f"\n==========================================")
    print(f"Scanning: {pdf_path.name}")
    print(f"==========================================")
    reader = pypdf.PdfReader(pdf_path)
    print(f"Total pages: {len(reader.pages)}")

    matches = {kw: [] for kw in keywords}

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        lower_text = text.lower()
        for kw in keywords:
            if kw in lower_text:
                matches[kw].append(page_idx + 1)

    for kw, pages in matches.items():
        if pages:
            print(f"  Keyword '{kw}': found on {len(pages)} pages -> {pages[:10]}")
