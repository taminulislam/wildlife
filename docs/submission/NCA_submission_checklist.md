# Neural Computing and Applications: submission checklist

Submission system: https://www.editorialmanager.com/nca/ (article type: Original Article).
Guidelines checked 2026-09-15 at https://link.springer.com/journal/521/submission-guidelines.

## Files to upload

| Editorial Manager item | File | Status |
|---|---|---|
| Blinded manuscript (LaTeX source) | `NCA_submission_<date>/main.tex` (single file, no `\input`), `main.bbl`, `main.bib`, `sn-jnl.cls`, `sn-basic.bst`, `figures/` | built by `scripts/make_submission_package.sh` |
| Manuscript PDF (blinded) | `main.pdf` from the package | built; author metadata checked |
| Title page (separate file) | `title_page.pdf` (authors, affiliations, ORCIDs, corresponding author, all declarations, acknowledgements) | ready |
| Cover letter | text below, pasted into the EM box | draft below |
| Figures | embedded in the LaTeX source; Springer may ask for TIFF/EPS at production | all raster figures >= 300 dpi at print width except `evidence_two.jpg` (190 dpi), see below |
| Supplementary material | none | n/a |

## Manuscript requirements

- Double-blind: `\blindtrue` in `main.tex` prints "Anonymous Authors"; funding, acknowledgements and
  contributions are on the title page only. The blinded PDF contains no author name or siu.edu address
  (checked by the package script). The study area (southern Illinois) stays in the text; a study area is
  not an author identifier.
- Abstract 204 words (limit 150 to 250). Keywords 6 (limit 4 to 6). Numbered references.
- Declarations present (title page and the `\blindfalse` build): Funding (U.S. Fish and Wildlife Service, Federal Aid in Wildlife Restoration Program via the Illinois
  Department of Natural Resources, project W-87-R; funder registry IDs 10.13039/100000202 and 10.13039/100004887), Competing interests, Ethics approval, Consent (not applicable), Data
  availability, Code availability, Author contributions, Acknowledgements.
- No page limit. Current length: 44 pages single column, about 13,300 words of body text.

## Still to do before pressing Submit

1. Decide whether to trim further (optional; no limit).
2. `figures/evidence_two.jpg` is 190 dpi at its print width; regenerate at >= 300 dpi
   (`src/viz/evidence_figure.py`) or accept a production request later.
3. Seth Morelock's ORCID (optional; only the corresponding author's is used by the system).
4. Zenodo: the data statement promises deposit on acceptance. A reserved Zenodo DOI (embargoed deposit)
   before submission would let the statement carry a link, which the policy prefers.
5. Confirm the manuscript is not under consideration anywhere else (the MDPI Journal of Imaging
   invitation must not have been used).
6. Suggested reviewers (optional): 3 to 5 independent names with institutional e-mails. Candidates from
   the reference list, none connected to SIU: Sara Beery (MIT) and Justin Kay (MIT), Caltech fish
   counting; Devis Tuia (EPFL), digital ecology; Benjamin Kellenberger (Yale), aerial wildlife
   detection; Grant Hamilton (Queensland University of Technology), thermal drone koala counts;
   Anwaar Ulhaq (Central Queensland University), thermal animal detection; Duane Diefenbach
   (Penn State / USGS), deer distance sampling. Take e-mails from the papers, not from memory.
7. Springer Nature's AI policy: use of a large language model beyond copy-editing must be documented
   in the Methods section. Decide how to describe the assistance used in preparing this manuscript.
8. Authorship is final at acceptance; confirm the five-author list and order now.
9. Open access choice at acceptance: Springer hybrid, so subscription route costs nothing; SIU's
   Springer Nature agreement may cover the APC if a 2026 waiver remains (opensiuc@lib.siu.edu).

## Cover letter (draft)

Dear Editor,

We submit "Automated Counting of White-Tailed Deer in Vehicle-Mounted Thermal Video: Detector
Benchmarking and a Staged Error Decomposition" for consideration as an original article in Neural
Computing and Applications.

The paper asks whether detection accuracy predicts counting accuracy when a deep-learning pipeline
counts animals in thermal road-transect video. On a new corpus of 32 fully annotated transects with
236 individually tracked white-tailed deer, we benchmark eleven detectors and decompose the counting
error of an end-to-end pipeline into detection, association and confirmation. Detection is close to
solved once scored per animal, while every improvement to the candidate pool makes the final count
worse, and learned confirmation stages lose to a three-parameter rule. We show why, formally and
empirically, and give the operating point an abundance survey should use instead of the
minimum-error one.

The manuscript has not been published and is not under consideration elsewhere. No part of it has
appeared in a conference or preprint. The corpus, annotations and evaluation code will be deposited
in Zenodo on acceptance. All authors approved the submission and have no competing interests. The
study was observational, recorded from public roads, and required no ethics approval.

We have prepared the manuscript for double-blind review and upload the title page separately.

Yours sincerely,
Taminul Islam (corresponding author), on behalf of all authors
School of Computing, Southern Illinois University, Carbondale, IL 62901, USA
taminul.islam@siu.edu
