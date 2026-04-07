"""
convert_to_endnote.py
Converts manuscript.md to an EndNote-compatible DOCX using pandoc --citeproc.

Steps:
  1. Parse references.ris → references.json (CSL-JSON)
  2. Convert manuscript.md in-text citations [N] → [@refN] pandoc syntax → manuscript_cited.md
  3. Run pandoc --citeproc to produce manuscript_endnote.docx
"""

import json
import re
import subprocess
import sys
from pathlib import Path

PANDOC = r"C:\Users\indan\AppData\Local\Pandoc\pandoc.exe"
BASE = Path(__file__).parent

# ── Step 1: Parse references.ris → CSL-JSON ──────────────────────────────

def parse_ris(ris_path: Path) -> list[dict]:
    """Parse RIS file into a list of CSL-JSON entries."""
    entries = []
    current = {}
    authors = []

    ty_map = {
        "JOUR": "article-journal",
        "RPRT": "report",
        "CONF": "paper-conference",
        "ELEC": "webpage",
        "UNPB": "manuscript",
    }

    for line in ris_path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip()
        if not line:
            continue
        tag = line[:2].strip()
        val = line[6:].strip() if len(line) > 6 else ""

        if tag == "TY":
            current = {"type": ty_map.get(val, "article-journal")}
            authors = []
        elif tag == "ID":
            current["id"] = val.lower().replace("ref", "ref")  # ref1, ref2 …
        elif tag == "AU":
            if "," in val:
                parts = val.split(",", 1)
                authors.append({"family": parts[0].strip(), "given": parts[1].strip()})
            else:
                authors.append({"literal": val})
        elif tag == "TI":
            current["title"] = val
        elif tag == "JO":
            current["container-title"] = val
        elif tag == "VL":
            current["volume"] = val
        elif tag == "IS":
            current["issue"] = val
        elif tag == "SP":
            current["_sp"] = val
        elif tag == "EP":
            current["_ep"] = val
        elif tag == "PY":
            try:
                current["issued"] = {"date-parts": [[int(val)]]}
            except ValueError:
                pass
        elif tag == "DO":
            current["DOI"] = val
        elif tag == "AN":
            current["PMID"] = val
        elif tag == "UR":
            current["URL"] = val
        elif tag == "PB":
            current["publisher"] = val
        elif tag == "CY":
            current["publisher-place"] = val
        elif tag == "ER":
            if authors:
                current["author"] = authors
            sp = current.pop("_sp", "")
            ep = current.pop("_ep", "")
            if sp and ep and sp != ep:
                current["page"] = f"{sp}-{ep}"
            elif sp:
                current["page"] = sp
            entries.append(current)
            current = {}
            authors = []

    return entries


# ── Step 2: Convert manuscript citations to pandoc syntax ─────────────────

def convert_citations(md_text: str) -> str:
    """
    Convert [N] and [N,M,...] style citations to pandoc [@refN] syntax.
    Also removes the manual References section at the end.
    """
    def replace_cite(m):
        inner = m.group(1)
        refs = [f"@ref{n.strip()}" for n in inner.split(",")]
        return "[" + "; ".join(refs) + "]"

    # Match citation patterns: [1], [5,6,23,24], [21,26], etc.
    # Be careful not to match markdown image syntax ![...](...) or link text
    converted = re.sub(
        r'(?<!\!)\[(\d+(?:\s*,\s*\d+)*)\](?!\()',
        replace_cite,
        md_text
    )

    # Remove the manual References section (pandoc citeproc generates its own)
    ref_pattern = re.compile(
        r'\n## References\n.*',
        re.DOTALL
    )
    converted = ref_pattern.sub('\n## References\n\n::: {#refs}\n:::\n', converted)

    return converted


# ── Step 3: Vancouver CSL style (inline) ─────────────────────────────────

VANCOUVER_CSL = r"""<?xml version="1.0" encoding="utf-8"?>
<style xmlns="http://purl.org/net/xbiblio/csl" class="in-text" version="1.0"
  demote-non-dropping-particle="sort-only" default-locale="en-US">
  <info>
    <title>Vancouver</title>
    <id>http://www.zotero.org/styles/vancouver</id>
    <link href="http://www.zotero.org/styles/vancouver" rel="self"/>
    <link href="http://www.nlm.nih.gov/bsd/uniform_requirements.html" rel="documentation"/>
    <author><name>Sebastian Karcher</name></author>
    <category citation-format="numeric"/>
    <category field="medicine"/>
    <updated>2023-04-01T00:00:00+00:00</updated>
    <rights license="http://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA-3.0</rights>
  </info>
  <locale xml:lang="en">
    <terms>
      <term name="accessed">cited</term>
      <term name="no date">date unknown</term>
    </terms>
  </locale>
  <macro name="author">
    <names variable="author">
      <name sort-separator=" " initialize-with="" name-as-sort-order="all" delimiter=", "
        delimiter-precedes-last="always"/>
      <label form="long" prefix=", "/>
      <substitute>
        <names variable="editor"/>
      </substitute>
    </names>
  </macro>
  <macro name="editor">
    <names variable="editor" suffix=".">
      <name sort-separator=" " initialize-with="" name-as-sort-order="all" delimiter=", "
        delimiter-precedes-last="always"/>
      <label form="long" prefix=", "/>
    </names>
  </macro>
  <macro name="publisher">
    <group delimiter=": ">
      <text variable="publisher-place"/>
      <text variable="publisher"/>
    </group>
  </macro>
  <macro name="access">
    <choose>
      <if variable="DOI">
        <text variable="DOI" prefix="doi:"/>
      </if>
      <else-if variable="URL">
        <group delimiter=". ">
          <text variable="URL"/>
          <group prefix="[" suffix="]">
            <text term="accessed"/>
          </group>
        </group>
      </else-if>
    </choose>
  </macro>
  <macro name="title">
    <text variable="title"/>
  </macro>
  <macro name="edition">
    <choose>
      <if is-numeric="edition">
        <group delimiter=" ">
          <number variable="edition" form="ordinal"/>
          <text term="edition" form="short"/>
        </group>
      </if>
      <else>
        <text variable="edition" suffix="."/>
      </else>
    </choose>
  </macro>
  <macro name="issued">
    <date variable="issued">
      <date-part name="year"/>
    </date>
  </macro>
  <citation collapse="citation-number">
    <sort>
      <key variable="citation-number"/>
    </sort>
    <layout prefix="[" suffix="]" delimiter=",">
      <text variable="citation-number"/>
    </layout>
  </citation>
  <bibliography et-al-min="7" et-al-use-first="6" second-field-align="flush">
    <layout>
      <text variable="citation-number" suffix=". "/>
      <text macro="author" suffix=". "/>
      <text macro="title" suffix=". "/>
      <choose>
        <if type="article-journal">
          <group suffix=".">
            <text variable="container-title" form="short" strip-periods="true"/>
            <text macro="edition" prefix=" "/>
            <text macro="issued" prefix=" "/>
            <group prefix=";">
              <text variable="volume"/>
              <text variable="issue" prefix="(" suffix=")"/>
            </group>
            <text variable="page" prefix=":"/>
          </group>
          <text macro="access" prefix=" "/>
        </if>
        <else-if type="report">
          <group suffix=".">
            <text macro="publisher"/>
            <text macro="issued" prefix="; "/>
          </group>
        </else-if>
        <else-if type="paper-conference">
          <group suffix=".">
            <text variable="container-title" form="short" strip-periods="true"/>
            <text macro="issued" prefix=" "/>
            <group prefix=";">
              <text variable="volume"/>
            </group>
            <text variable="page" prefix=":"/>
          </group>
        </else-if>
        <else>
          <group suffix=".">
            <text variable="container-title"/>
            <text macro="issued" prefix=" "/>
          </group>
          <text macro="access" prefix=" "/>
        </else>
      </choose>
    </layout>
  </bibliography>
</style>
"""


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=== EndNote-Compatible DOCX Conversion ===\n")

    # Step 1: RIS → CSL-JSON
    ris_path = BASE / "references.ris"
    json_path = BASE / "references.json"
    print(f"Step 1: Parsing {ris_path.name} → {json_path.name}")
    entries = parse_ris(ris_path)
    json_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → {len(entries)} entries written to {json_path.name}")

    # Step 2: Convert citations in manuscript.md
    md_path = BASE / "manuscript.md"
    cited_path = BASE / "manuscript_cited.md"
    print(f"\nStep 2: Converting citations in {md_path.name} → {cited_path.name}")
    md_text = md_path.read_text(encoding="utf-8")
    cited_text = convert_citations(md_text)

    # Count conversions
    n_orig = len(re.findall(r'(?<!\!)\[(\d+(?:\s*,\s*\d+)*)\](?!\()', md_text))
    n_remaining = len(re.findall(r'(?<!\!)\[(\d+(?:\s*,\s*\d+)*)\](?!\()', cited_text))
    print(f"  → {n_orig} citation markers converted, {n_remaining} remaining (should be 0)")

    cited_path.write_text(cited_text, encoding="utf-8")

    # Step 3: Write Vancouver CSL
    csl_path = BASE / "vancouver.csl"
    print(f"\nStep 3: Writing {csl_path.name}")
    csl_path.write_text(VANCOUVER_CSL.strip(), encoding="utf-8")

    # Step 4: Run pandoc --citeproc
    out_path = BASE / "manuscript_endnote.docx"
    print(f"\nStep 4: Running pandoc --citeproc → {out_path.name}")
    cmd = [
        PANDOC,
        str(cited_path),
        "--citeproc",
        f"--bibliography={json_path}",
        f"--csl={csl_path}",
        "-o", str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: pandoc failed:\n{result.stderr}")
        sys.exit(1)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  → {out_path.name} generated ({size_mb:.1f} MB)")

    print("\n=== Done ===")
    print(f"\nOutput files:")
    print(f"  {json_path.name}  — CSL-JSON bibliography ({len(entries)} entries)")
    print(f"  {cited_path.name}  — Manuscript with pandoc citation syntax")
    print(f"  {out_path.name}  — EndNote-compatible DOCX with citeproc citations")
    print(f"\nTo use with EndNote:")
    print(f"  1. Import references.ris into your EndNote library")
    print(f"  2. Open {out_path.name} in Word")
    print(f"  3. Use CWYW 'Convert Citations and Bibliography' to link to your library")


if __name__ == "__main__":
    main()
