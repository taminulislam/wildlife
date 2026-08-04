# Publication plan — thermal deer detection & counting

**Written 2026-07-30**, after the scope thread with Guillaume Bastille (wildlife ecology)
and Khaled Ahmed. All experiments are complete; no compute is outstanding.

---

## Decisions already made by the team (do not relitigate)

| Question | Decision | Who |
|---|---|---|
| Individual deer ID across videos? | **No.** Not credible at this resolution to a wildlife ecologist. | Guillaume |
| Additional labelling for ReID? | **No — explicitly declined.** | Guillaume |
| What the system does | Prevent double counting **within a survey** | Guillaume: *"what you have accomplished is what we want/needed"* |
| Number of manuscripts | **Two** — one CV, one for ecologists | Guillaume |
| Meeting | Optional, his preference is either; he is travelling | Guillaume |

The ReID work is **not wasted**: it produced the project's best candidate pool (§6.9) and a
measured resolution limit that justifies the scope decision in print.

---

## Paper 1 — computer vision venue (the one due next week)

**Working title:** *Counting is not detecting: the confirmation bottleneck in thermal
wildlife video*

### The thesis, in one line

On unseen video the pipeline **detects 98.8%** of deer, gives **88.0%** their own track,
and **counts 62.7%**. Detection is solved; counting is not; and the loss is located
entirely in one stage.

### Why this is publishable

Four independent interventions improved candidate generation. Every one made the
hand-tuned counter *worse*:

| pool | unseen reached | unseen primary | held-out counted |
|---|---|---|---|
| C orphan | 74/83 | 66/83 | **55/83 = 66.3%** |
| E max-recall | 79/83 | 69/83 | 54/83 |
| F loose NMS | 79/83 | 70/83 | 41/83 |
| **G ReID** | **82/83** | **73/83** | 52/83 |

That inversion is the paper. It is not a negative result — it *localises* the problem, and
it is the argument for a learned temporal confirmation head.

### Structure

1. **Intro** — thermal wildlife survey; counting ≠ detection; contribution = locating and
   quantifying the confirmation bottleneck.
2. **Dataset** — 32 videos, 521,930 frames, 236 deer, 21,646 boxes, 4 sites. Include the
   annotation-quality finding (§1.1): 94% of GT is CVAT-interpolated, deer travel a median
   2.28 box-widths between human keyframes. This is a genuine methodological contribution —
   it accounts for ~⅓ of apparent detection error.
3. **Detection benchmark** — 13 architectures, one split (§3, §3.2). Two findings: DETR
   family underperforms at 27 px; RTMDet-m's COCO-mAP lead is a precision artefact that
   inverts under the counting criterion.
4. **Evaluation protocol** ★ — the any-overlap / "touch" criterion, and why IoU≥0.50 is the
   wrong metric when the question is *how many animals*, not *where exactly*. Report as
   presence/counting recall, never as mAP.
5. **Counting pipeline** — detection → BoT-SORT (optical-flow GMC) → confirmation.
6. **The bottleneck** ★★★ — §6.7.1 and §6.9. The reach → primary → counted funnel, the
   four-pool inversion table, the held-out protocol (fit the rule on detector-train videos,
   freeze, report on 13 unseen).
7. **ReID as a bounded negative** — §6.9(a). Rank-1 0.899 looks strong until you see box
   size alone reaches 0.638 and a pretrained CNN scores below plain geometry. Honest,
   quantified, and it justifies the scope.
8. **Limitations** — 236 deer; SHB holds 56%; MAS has 15; no cross-video identity data;
   single season, single region.

### Figures needed (none exist yet — this is the critical path)

| # | Figure | Status |
|---|---|---|
| 1 | **The funnel**: reach 98.8% → primary 88.0% → counted 62.7% | **Not built — highest priority** |
| 2 | Four-pool inversion (candidates ↑ vs counted ↓) | Not built |
| 3 | Detector comparison, IoU≥0.50 vs any-overlap side by side | Data ready (§3.2) |
| 4 | Qualitative: group of 4 deer, all counted | ✅ `docs/email_figures/fig1` |
| 5 | Per-deer evidence sheet (identity over time + confidence) | ✅ `docs/email_figures/fig3` |
| 6 | Annotation drift: keyframe vs interpolated GT | Not built |
| 7 | ReID rank-1 by cue (bar chart) | Data ready (§6.9) |

**Regenerate qualitative figures restricted to the 13 unseen videos.** Current ones mix
training footage — a reviewer will catch it. ~1 GPU-hour.

### Numbers to publish (and the ones not to)

- ✅ **MAE 2.38, 66.3% counted, bias −1.92**, held out on 13 unseen videos
- ✅ 97.9% detection, 94.2% precision on the counting criterion
- ✅ 92.1% of counted deer at calibrated confidence ≥0.80
- ❌ **Never headline 89% / MAE 1.88** — that includes 19 detector-training videos. It
  appears only as the labelled "optimistic, all-32" row.

---

## Paper 2 — ecologist-facing (Guillaume's second manuscript)

**His framing:** *"a product that would be targeting the ecologists to make sure the
approach is available to other users."*

Different paper, not a rewrite. Venue candidates: *Methods in Ecology and Evolution*,
*Remote Sensing in Ecology and Conservation*, *Ecological Informatics*.

### What changes

| Paper 1 (CV) | Paper 2 (ecology) |
|---|---|
| Why counting fails | How to run a survey with it |
| mAP, rank-1, ablations | Deer/km, effort, detection probability |
| Held-out MAE | **Bias −1.92 = conservative** — the number ecologists care about |
| Architecture comparison | Hardware, cost, field protocol |
| — | Reproducible workflow: annotate → train → count → verify |

### The strongest asset for this paper

**Calibrated per-deer confidence.** 92.1% of counted deer at ≥0.80 (mean 0.965), with 793
uncertain tracks auto-flagged for human review. That is exactly the human-in-the-loop
property a field ecologist needs — a defensible count *plus* a review queue, not a black box.

### Also belongs here

- The Jetson edge-camera concept (see below) as deployment/future work
- Under-counting as a **survey-design property**: never over-counts, so estimates are
  conservative — arguably preferable for management decisions
- Per-site performance and what drives it (dense groups at SHB vs sparse at MAS)

### Sequencing

Draft **after** Paper 1 is submitted. Do not run in parallel — same underlying results, and
Paper 1 establishes the technical claims Paper 2 will cite. Ask Guillaume to co-lead this
one; the framing and venue judgement are his expertise, not ours.

---

## Timeline

| When | What |
|---|---|
| **Days 1–2** | Build figures 1, 2, 6, 7. Regenerate qualitative figures on unseen videos only. |
| **Days 3–5** | Draft Paper 1. `RESULTS_LOG.md` is self-sufficient — every number has a job ID and a repro command. |
| **Day 6** | Internal read. Send to Khaled and Guillaume. |
| **Next week** | Revise, submit. |
| **After** | Meeting with Guillaume on his return; scope Paper 2 with him. |

---

## Open items

1. **Reply to Guillaume** — sent; meeting offered. Nothing blocking.
2. **Send the ReID counting result** — promised in the last email, and it is good news:
   ReID gave the best pool in the project (§6.9). Worth one short message.
3. **Jetson edge camera** — Guillaume's trail-camera architecture is a strong Paper 2
   future-work section. A static camera removes the optical-flow GMC cost entirely, which
   is our heaviest CPU component. Would need retraining for the fixed-camera viewpoint.
4. **Disk** — `/work/hdd` at 16 GB; `runs/` + `mmdet_runs/` are 96% of it. Clean up only
   **after** submission.

---

## What is NOT being done, and why

- **Individual re-identification** — declined by the ecology co-author; not credible at
  29×24 px; no validation data.
- **Additional labelling** — explicitly declined by Guillaume.
- **Deformable DETR rerun** — DINO already represents the DETR family on a full schedule
  and lands below Faster R-CNN. Reported as excluded.
- **Species classification** — out of scope since project start.
