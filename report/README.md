# Report and paper

| What | Folder | Overleaf compiler |
|---|---|---|
| Project report (guide's 5-chapter outline) | `report/latex/` (`main.tex`) | **XeLaTeX** (Menu → Compiler), needed for Kannada |
| 6-page IEEE paper | `paper/latex/` (`paper.tex`) | pdfLaTeX (default) |

Both compile without errors (checked 2026-10-13 with Tectonic, a XeTeX-based engine).
The paper is exactly 6 pages.

## The only things left to fill in
- **Report:** edit `report/latex/details.tex` (names, USNs, guide, HOD, principal, college,
  branch, year, place and date). The title page, certificate, declaration and
  acknowledgement all take their details from that one file. Optional college logo:
  instructions are inside `details.tex`.
- **Paper:** the author block at the top of `paper/latex/paper.tex` (names, department,
  college, emails).
- **Before signing:** read the declaration (`report/latex/frontmatter.tex`). It states that
  AI tools assisted with development and drafting; edit if your department wants different
  wording.

Red `[TODO: ...]` text in the PDF marks exactly these fields. Nothing else is missing.

## Uploading to Overleaf
1. Right-click `report/latex` → Compress to ZIP. On overleaf.com: New Project → Upload
   Project → choose the zip.
2. Menu (top left) → Compiler → **XeLaTeX**. Click Recompile.
3. Repeat with `paper/latex` (leave the compiler on pdfLaTeX).

## Sources
- Every reference in `references.bib` was checked on its official page (arXiv, ACL
  Anthology, publisher, PIB) on 2026-10-13. Literature-survey summaries and the numbers in
  them come from each paper's abstract.
- `literature_candidates.md`: how the survey papers were found.
- The Kannada font in `report/latex/fonts/` is Noto Sans Kannada (SIL Open Font License),
  included so the report compiles the same everywhere.

Outline source: `DocScanner Aug 18, 2026 1-02 PM.pdf` (guide's handwritten format).
