# Reply — ReID scope question (Dr. Bastille)

**Subject:** Re: ReID — temporary ID vs. identifying individuals

---

Both are possible in principle, but our measurements say only the first is real, and you
are right about the resolution. What is running is a temporary ID, used within a video to
keep deer separate and avoid double counting, not identification of individuals across
videos.

I tested this directly. Our CVAT tracks are already identity labels, so the model trained
on deer from one set of videos and was tested on 69 deer it had never seen, in 7 videos it
had never seen. Rank-1 accuracy, every cue scored on that same held-out set:

| Cue | Rank-1 |
|---|---|
| Chance | 10% |
| Brightness alone | 10% |
| Thermal silhouette alone (size normalised out) | 44% |
| Raw thermal crop | 49% |
| **Box size alone** | **64%** |
| Off-the-shelf ResNet-50 features | 73% |
| Simple geometry (size, aspect, intensity) | 80% |
| Our trained thermal encoder | 90% |

The 90 percent overstates what we have. Box size alone already reaches 64 percent, and
within one video a deer's range changes slowly, so most of what looks like identification
is matching distance rather than matching the animal. Appearance on its own is weak:
brightness carries nothing, silhouette with scale removed gets 44 percent, and a pretrained
CNN still lands below plain geometry. At a median 29 by 24 pixels there is simply not
enough detail on the animal to key an identity to.

So I would like your steer. Do we frame this as track association only, stating the
resolution limit and making no individual-ID claim? Or pursue individual re-identification,
which would mean labelling more of our footage so the claim can actually be tested across
sightings? I am happy either way and would rather you and the team set the scope.

A run measuring whether this improves the count finishes shortly; I will send that number.

---

*Supporting detail: `docs/RESULTS_LOG.md`; jobs 2788117 (feasibility), 2788317 (trained
encoder), 2788316 (counting impact, running).*
