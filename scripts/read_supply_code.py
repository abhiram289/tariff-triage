import pypdf
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

reader = pypdf.PdfReader("data/corpus/supply_code.pdf")
for p in [36, 40, 96, 97]:
    print(f"\n==========================================")
    print(f"supply_code.pdf - Page {p}")
    print(f"==========================================")
    text = reader.pages[p - 1].extract_text()
    print(text[:2000])
