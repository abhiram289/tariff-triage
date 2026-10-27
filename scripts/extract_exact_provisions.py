import pypdf

def print_page_snippet(pdf_name, pages):
    reader = pypdf.PdfReader(f"data/corpus/{pdf_name}")
    for p in pages:
        print(f"\n==========================================")
        print(f"{pdf_name} - Page {p}")
        print(f"==========================================")
        text = reader.pages[p - 1].extract_text()
        print(text[:2500])

print_page_snippet("tariff_order.pdf", [547, 548, 552])
print_page_snippet("supply_code.pdf", [36, 40])
