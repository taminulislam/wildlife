# FutureHouse prompt — literature review for the thermal deer counting paper

Everything between the rules below is the prompt. Paste it whole. The paper's own 21
current citations are listed inside it so they are not re-derived as "new" findings.

---

## PROMPT BEGINS

You are writing the **Related Work** section for a peer-reviewed methods paper submitted to
*Journal of Imaging* (MDPI). I need a comprehensive, critical review covering **at least 80
distinct peer-reviewed or archival papers**, every one of which must be cited in the text.
Depth matters more than breadth-for-its-own-sake: a paper that is cited must be *used* —
its finding, dataset, or limitation stated — not listed in a group citation.

### The work you are reviewing for

A system that **counts individual white-tailed deer in nocturnal thermal video** shot from a
vehicle-mounted FLIR camera along road transects, and — more importantly — a **measurement
study of why such systems fail**.

Concretely, the paper contributes:

1. **A corpus.** 32 fully annotated nocturnal thermal road-transect videos from four sites in
   southern Illinois; 521,930 frames at 640×512, 60 fps, single annotated class; **236
   individual deer, each annotated as a distinct track**, 21,646 boxes. Because one
   annotated track is one animal, the same annotation serves as detection, association and
   count ground truth simultaneously. Median animal is 29×24 px; 71% fall below the COCO
   small-object threshold; zero boxes in the COCO-large category.

2. **A three-stage decomposition of counting error.** For each ground-truth animal the paper
   asks whether it is REACHED (touched by any candidate track — measures detection), PRIMARY
   (some candidate covers it better than it covers any other animal — measures association),
   and COUNTED (the confirmation stage accepted it). On held-out video the pipeline reaches
   98.8%, gives 88.0% a distinct track, and counts 62.7%. **Detection is not the bottleneck.**

3. **A negative result the authors consider the main finding.** Four independent
   interventions that each demonstrably improve candidate generation — lowering detector
   confidence 0.10→0.02, removing the tracker's track-initialisation gate, loosening NMS from
   IoU 0.50 to 0.90, and adding appearance re-identification — each make the *final count
   worse*, because the confirmation stage cannot exploit a pool it did not shrink. Learned
   confirmation heads (temporal transformer ~60k params, gradient boosting, logistic
   regression) all lose to a three-parameter hand-tuned rule, under three evaluation
   protocols, and performance degrades *monotonically with model capacity*.

4. **A matching-criterion finding.** The standard IoU≥0.50 criterion re-ranks detectors
   relative to a counting-appropriate "any overlap" criterion — seven of eleven models shift
   two or more places, one falls from 2nd to 8th. Selecting a detector by its detection
   metric does not select the model that counts best: the best AP50 model counts fewer
   animals than the model actually adopted.

5. **A leak-free protocol.** The confirmation rule is swept on the 19 detector-training
   videos, frozen, and reported on 13 held out. Evaluating on all 32 inflates coverage by 23
   points (89.0% vs 66.3%).

6. Twelve detector architectures benchmarked under identical preprocessing and splits, plus a
   quantified re-identification ceiling at this pixel scale (a trained thermal encoder reaches
   rank-1 0.899, but **box size alone reaches 0.638** — most apparent identification is range
   matching, not identity).

### What I need you to produce

**(A) A structured Related Work review, ~2,500–3,500 words**, organised into the themed
subsections below. Use the indicated target counts as a guide; the total must reach 80+
distinct works. For each theme, do not merely summarise — state what the prior work
established, on what data, and **what it does not settle** for the problem above.

| # | Theme | Target |
|---|---|---|
| 1 | Wildlife abundance and density estimation in ecology — distance sampling, N-mixture models, mark–recapture, spotlight and aerial transect surveys, and their stated error budgets | 8 |
| 2 | Thermal and infrared imaging for wildlife detection and survey, ground and airborne | 10 |
| 3 | UAV / aerial wildlife survey and automated counting | 10 |
| 4 | Camera traps and machine learning for ecology, including domain shift and generalisation to new sites | 8 |
| 5 | Small-object detection: architectures, augmentation, resolution, and the tiny-object benchmarks (AI-TOD, TinyPerson, VisDrone, DOTA and similar) | 8 |
| 6 | Object detection architectures — emphasise 2023–2026 work not already cited below, including recent YOLO variants, DETR successors, and anchor-free / assignment advances | 8 |
| 7 | Multi-object tracking: tracking-by-detection, camera-motion compensation, appearance versus motion association, and MOT evaluation (MOTA, IDF1, HOTA) | 8 |
| 8 | Counting as a task in its own right: crowd counting and density regression, and especially **unique-object / cross-frame counting in video**, where an object must be counted once across many frames | 8 |
| 9 | Animal re-identification, including performance at low resolution and in thermal imagery | 6 |
| 10 | Evaluation methodology: matching criteria, IoU threshold sensitivity for small objects, localisation-versus-recognition error taxonomies, and dataset-leak / train–test contamination in vision benchmarks | 6 |
| 11 | Confidence calibration and uncertainty quantification in object detection, and human-in-the-loop review for ecological monitoring | 5 |
| 12 | White-tailed deer specifically: density estimation methods, deer–vehicle collision modelling, disease surveillance (CWD), and any published thermal or camera-based deer survey | 6 |

**(B) A comparison table** positioning this work against the most relevant prior systems and
datasets — aim for **12–20 rows**, chosen as the closest comparables (thermal wildlife
datasets, video wildlife counting systems, animal-counting benchmarks). Columns:

| Column | Content |
|---|---|
| Work | first author + year, with citation key |
| Modality | thermal / RGB / multispectral |
| Platform | vehicle, handheld, UAV, fixed camera trap, manned aircraft, satellite |
| Species / taxa | |
| Data scale | images or frames; number of videos if video |
| Individuals | number of distinct annotated animals, or "n/r" |
| Per-individual identity? | Yes/No — is each animal annotated as a distinct track |
| Continuous video? | Yes/No |
| Task evaluated | detection / tracking / counting / re-ID |
| End-to-end count reported? | Yes/No |
| Train–test split by video or site? | Yes/No/unclear |
| Public release? | Yes/No |

Fill every cell from the paper itself; write "n/r" where not reported rather than guessing.
Add our work as the final row, using the numbers in the description above.

**(C) One closing paragraph, 150–250 words, stating the research gap.** It must follow from
the table and the review rather than asserting novelty. It should make clear that: thermal
wildlife datasets with **per-individual track identity over continuous video** are essentially
absent; that counting is nearly always evaluated as detection accuracy or per-image density
rather than as distinct-individual accuracy; that train–test separation by video or site is
frequently unclear; and that no prior work isolates *which stage* of a detect-track-confirm
pipeline loses animals. Do not overclaim — if you find prior work that does any of these,
say so explicitly and narrow the gap accordingly. **A narrower, defensible gap is worth far
more to me than a broad one.**

**(D) A BibTeX file containing every cited work.**

### Rules on citations — these matter more than coverage

- **Cite only papers you have actually retrieved.** Do not construct a plausible-looking
  reference. If you cannot verify a work exists, omit it and say so.
- Every entry needs a **DOI or arXiv ID**. List separately any work you cite without one.
- Prefer peer-reviewed venues; arXiv is acceptable for recent vision work but mark it.
- Keys in the form `lastnameYEARkeyword`, e.g. `kellenberger2018detecting`.
- Correct entry types (`@article`, `@inproceedings`, `@book`), full author lists, venue,
  year, volume/pages, publisher where applicable.
- **Prioritise 2020–2026** for the vision themes; the ecology and survey-methods themes may
  reach further back where the foundational work is older.
- Include ecology and wildlife-management journals, not only vision venues. A reviewer for
  this journal will be looking for the domain literature.

### Already cited — do not present these as new, but do situate them

`ren2015faster` (Faster R-CNN), `redmon2016yolo`, `wang2024yolov9`, `wang2024yolov10`,
`zhao2024rtdetr`, `carion2020detr`, `zhang2023dino`, `lyu2022rtmdet`, `zhang2020atss`,
`feng2021tood`, `bewley2016sort`, `wojke2017deepsort`, `zhang2022bytetrack`,
`aharon2022botsort`, `lin2014coco`, `zuiderveld1994clahe` (CLAHE),
`norouzzadeh2018automatically`, `beery2018recognition`, `kellenberger2018detecting`,
`corcoran2019automated`, `zhang2016single`.

### Output format

1. The review prose, in Markdown, with `\cite{key}` markers inline so it drops into LaTeX.
2. The comparison table in Markdown.
3. The gap paragraph, clearly labelled.
4. A fenced ```bibtex block containing all entries.
5. A short closing list of: works you could not verify, themes where the literature was
   thinner than the target, and any claim in my description above that the literature
   **contradicts** — I want to know that before a reviewer tells me.

## PROMPT ENDS

---

## After it comes back

- Save the BibTeX to `overleaf_MDPI/refs_review.bib` and add
  `\bibliography{main,refs_review}` in `main.tex`.
- Spot-check ten citations at random against their DOIs before trusting the rest. Automated
  literature tools do fabricate references, and a fabricated citation in a submitted
  manuscript is far more damaging than a thin review.
- The gap paragraph is the piece to read hardest. If it claims something the comparison table
  does not support, cut it back to what the table shows.
