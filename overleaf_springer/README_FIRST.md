# Neural Computing and Applications (Springer Nature) submission

Converted from the MDPI Journal of Imaging package on 2026-09-14. **Nothing was cut**: every
section, table and figure of the MDPI version is here; only the class and front/back matter
changed. Trimming for NCA is still to come.

Compiler: **pdfLaTeX**. Main document: `main.tex`.

```
main.tex              sn-jnl class, title, abstract, keywords, back matter, appendix, bibliography
title_page.tex        separate title page (authors, ORCIDs, declarations) for double-blind review
sn-jnl.cls            Springer Nature journal class (template v3.1, December 2024)
sn-basic.bst          "Springer Basic" numbered reference style, the one NCA specifies
main.bib              references (shared with the MDPI version)
sec/1_intro.tex ... sec/7_appendix.tex   the manuscript body, one file per section
figures/              all figures (shared with the MDPI version)
mdpi_full_version/    frozen MDPI package: main.tex, sec/, main.bib, Definitions/ (paths patched
                      with the mdpi_full_version/ prefix so it compiles if chosen as main document)
```

## NCA requirements handled here

* **Double-blind review.** `\blindtrue` in `main.tex` prints "Anonymous Authors" and moves the
  declarations to `title_page.tex`. Switch to `\blindfalse` for the full author version.
* Abstract 150 to 250 words (ours is about 200, unstructured); 4 to 6 keywords (six).
* Numbered citations in square brackets via `sn-basic`.
* Declarations: funding, competing interests, ethics approval, consent, data availability,
  code availability, author contributions, acknowledgements (title page and `\blindfalse` build).
* Figures reference as "Fig. N" in running text and "Figure N" at sentence start (cleveref).

## What changed from the MDPI layout

| | MDPI | Springer |
|---|---|---|
| Class | `Definitions/mdpi` | `sn-jnl` with `sn-basic` |
| Wide floats | `adjustwidth{-\extralength}` + `\fulllength` | plain floats at `\textwidth` |
| Table placement | `[H]` | `[htbp]` (sn-jnl rejects the float package's `H`) |
| Abstract | MDPI numbered (1) to (4) structure | one unstructured paragraph, same sentences |
| Keywords | 8, semicolon separated | 6, comma separated |
| Back matter | MDPI macros | `\backmatter` + `\bmhead{}` declarations |
| Appendix | `\appendixstart` | `appendices` environment |

At final submission Springer wants one `.tex` file: flatten the `\input{}`s before uploading.
