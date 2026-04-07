"""
make_endnote_docx.py
Creates a DOCX with EndNote temporary (unformatted) citations.

EndNote's CWYW plugin recognizes temporary citations in this format:
    {Author, Year #RecordNumber}
and converts them to live field codes when you click
"Update Citations and Bibliography" in Word.

Usage:
  1. Create a NEW empty EndNote library
  2. Import references.ris into it (File → Import → RIS)
     → This gives records numbered 1–27 in import order
  3. Open the generated manuscript_endnote.docx in Word
  4. In the EndNote toolbar: "Update Citations and Bibliography"
     → All temporary citations become live EndNote field codes
"""

import re
from pathlib import Path

BASE = Path(__file__).parent
PANDOC = r"C:\Users\indan\AppData\Local\Pandoc\pandoc.exe"

# ── Author-Year mapping for each reference ──────────────────────────────
# Format: {FirstAuthor, Year #RecordNumber}
# RecordNumber matches import order when importing references.ris into a NEW library

CITE_MAP = {
    1:  "GBD 2016 TBI Collaborators, 2019",
    2:  "Maas, 2022",
    3:  "Badjatia, 2025",
    4:  "Kim, 2024",
    5:  "LaBuzetta, 2019",
    6:  "Rhodes, 2025",
    7:  "Sorger, 2011",
    8:  "Pantanowitz, 2024",
    9:  "GBD 2016 Neurology Collaborators, 2019",
    10: "McCrea, 2021",
    11: "Meehan, 2022",
    12: "Abril-Pla, 2023",
    13: "Caruana, 1997",
    14: "Ruder, 2017",
    15: "Angelopoulos, 2021",
    16: "Lundberg, 2017",
    17: "Huang, 2016",
    18: "Bienvenu, 2018",
    19: "Gu, 2019",
    20: "Mortazavi, 2012",
    21: "Maas, 2015",
    22: "Dewan, 2019",
    23: "Jorge, 2004",
    24: "Bryant, 2010",
    25: "Mikolic, 2025",
    26: "Steyerberg, 2019",
    27: "Schwimmbeck, 2021",
}


def convert_to_temp_citations(md_text: str) -> str:
    """Replace [N] and [N,M,...] with EndNote temporary citation format."""

    def replace_cite(m):
        nums = [int(n.strip()) for n in m.group(1).split(",")]
        parts = []
        for n in nums:
            label = CITE_MAP.get(n, f"Unknown, {n}")
            parts.append(f"{label} #{n}")
        return "{" + "; ".join(parts) + "}"

    # Match [1], [5,6,23,24], etc. but not ![...](...) image syntax
    converted = re.sub(
        r'(?<!\!)\[(\d+(?:\s*,\s*\d+)*)\](?!\()',
        replace_cite,
        md_text
    )
    return converted


def main():
    import subprocess

    print("=== EndNote Temporary Citations DOCX Generator ===\n")

    # Read manuscript
    md_path = BASE / "manuscript.md"
    md_text = md_path.read_text(encoding="utf-8")

    # Convert citations
    converted = convert_to_temp_citations(md_text)

    # Count conversions
    n_orig = len(re.findall(r'(?<!\!)\[(\d+(?:\s*,\s*\d+)*)\](?!\()', md_text))
    n_remaining = len(re.findall(r'(?<!\!)\[(\d+(?:\s*,\s*\d+)*)\](?!\()', converted))
    n_temp = len(re.findall(r'\{[^}]+#\d+[^}]*\}', converted))
    print(f"Converted {n_orig} citation markers → {n_temp} temporary citations")
    print(f"Remaining plain [N] citations: {n_remaining} (should be 0)")

    # Write intermediate markdown
    temp_md = BASE / "manuscript_temp_cite.md"
    temp_md.write_text(converted, encoding="utf-8")
    print(f"\nWrote {temp_md.name}")

    # Convert to DOCX via pandoc (plain conversion, NO citeproc)
    out_docx = BASE / "manuscript_endnote.docx"
    cmd = [PANDOC, str(temp_md), "-o", str(out_docx)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: pandoc failed:\n{result.stderr}")
        return

    size_mb = out_docx.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_docx.name} ({size_mb:.1f} MB)")

    # Show a sample citation
    print("\n--- Sample citations in the document ---")
    for line in converted.splitlines():
        if "{" in line and "#" in line and len(line) < 200:
            # Show first few lines with citations
            start = line.find("{")
            end = line.find("}", start) + 1
            if start >= 0 and end > 0:
                print(f"  ...{line[max(0,start-30):end+10]}...")
                break

    print("\n=== Instructions ===")
    print("1. In EndNote: create a NEW library (File → New)")
    print("2. Import references.ris: File → Import → File → select references.ris")
    print("   Import Option: 'Reference Manager (RIS)'")
    print("   → Records will be numbered 1–27 in import order")
    print(f"3. Open {out_docx.name} in Microsoft Word")
    print("4. In the EndNote tab: click 'Update Citations and Bibliography'")
    print("   → All {Author, Year #N} markers become live EndNote citations")
    print("5. Choose your desired output style (e.g., Vancouver, APA, etc.)")


if __name__ == "__main__":
    main()
