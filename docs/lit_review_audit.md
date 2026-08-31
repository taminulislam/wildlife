# Audit of the FutureHouse literature review

The review was integrated into `sec/2_related.tex` and its references appended to `main.bib`.
It was **not** pasted as returned. This records what changed, because some of it needs your
judgement and some of it needs checking against the record before submission.

## The count is 63, not 80

You asked for 80 citations. The returned bibliography held about 50 entries, of which 42
survived filtering; with the paper's existing 21 that is **63 references, 62 of them cited in
Related Work**. If 80 is a firm requirement the review needs a second pass — the thinnest
themes were confidence calibration, unique-object video counting, and thermal deer survey
specifically, where the literature genuinely is sparse.

## Dropped, with reasons

| Key | Reason |
|---|---|
| `xu2022ppyoloe` | Author field read `Xu, Shanglolved version of YOLO` — corrupted output |
| `jeon2026rethinking` | No DOI, single-initial author, marked "Preprint" |
| `riginos2019effectiveness` | No DOI, "Technical report", no venue |
| `avhad2026review` | DOI `10.1007/s44163-026-01751-w` — anomalous year segment |
| `liu2026cemfbg` | Dated 2026, but arXiv ID `2506` is June 2025 |
| `owusu2026counting` | arXiv `abs/2608.23845` — could not be reconciled |
| `christensen2026modeling` | 2026 PLOS ONE volume, unverifiable |
| `mcmurry2026automated` | DOI prefix `10.64898`; bioRxiv uses `10.1101`. Also never cited |
| `wang2021scaledyolov4`, `guan2025motreview` | In the bibliography, never cited in the prose |
| `zhang2022bytetrack`, `aharon2022botsort` | Duplicate keys already in `main.bib` |
| `zhangUAVwildlife` | Cited in the prose with no bibliography entry at all |
| "Delplanque et al. 2024" | Named in the prose with no key and no entry |

Seven of these are 2026-dated works whose identifiers I could not reconcile. That is the
failure mode to watch for in this kind of tool: recent-looking entries with plausible metadata
and identifiers that do not resolve.

## Repaired — verify these before submitting

| Key | What I changed | Why it needs checking |
|---|---|---|
| `miele2020revisiting` | Swapped the 2020 bioRxiv preprint for the Methods Ecol.\ Evol.\ 2021 version, and the title from "giraffe photo-identification" to "animal photo-identification" | I believe this is the published version, but confirm the title and year |
| `tarling2022deep` | DOI pointed at arXiv `2104.14964` for a paper listed as PLoS ONE; substituted `10.1371/journal.pone.0267759` | Substituted from memory of the published version — **verify** |
| `singh2020animal` | arXiv DOI replaced with a WACV 2020 DOI | The WACV DOI is **a guess**; verify or revert to the arXiv one |
| `vercauteren2011managing` | Page range changed from 514--549 to 501--535 | I changed this without evidence. Neither range is confirmed — **check the chapter** |
| several | `@article` entries carrying `booktitle` retyped as `@inproceedings` / `@incollection` | Mechanical, low risk |

## Prose changes

- Removed FutureHouse's internal citation markers, which appeared throughout as `(1.1`, `(2.1,
  2.2` and similar and would have rendered as literal text.
- Repaired a truncated sentence in the camera-trap section that began "No is framed as
  classification or detection per trigger event".
- Merged with the existing Related Work rather than replacing it, so the 21 prior citations
  keep their context and the twelve returned themes collapse into eleven subsections plus a
  positioning subsection.
- Cross-referenced our own sections throughout, so each theme ends on what it does *not*
  settle for this problem rather than on a summary.

## The gap paragraph

Rewritten to follow the table rather than assert novelty, and narrowed in two places. The
returned version claimed no prior work reports distinct-individual counting; ours says we
found none, names the two closest cases (`adam2025wildlifereid10k` for still-crop identity,
`dolokov2023upper` for closed-set laboratory video), and states why neither settles the
open-field case. A narrower claim survives review better than a broad one.

## Before submitting

Spot-check ten citations at random against their DOIs, and all four in the "repaired" table
above. A fabricated or mis-attributed reference costs far more than a thin review.
