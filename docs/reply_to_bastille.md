# Reply — ReID scope question (Dr. Bastille)

**Subject:** Re: ReID — temporary ID vs. identifying individuals

---

Both are possible in principle, but our measurements say only the first is real — and you
are right about the resolution.

What is currently running is a **temporary ID**: used within a video to keep deer separate
and avoid double counting. It is not identification of individuals across videos.

I tested whether true individual ID is even feasible on our data before building anything.
Our CVAT tracks are already identity labels — one track is one animal — so I could score
this properly, training on deer from one set of videos and testing on **69 deer the model
had never seen, in videos it had never seen**:

| Cue | Rank-1 accuracy |
|---|---|
| Chance | 10% |
| Brightness alone | 10% — *no signal at all* |
| Thermal silhouette alone (size-normalised) | 44% |
| Off-the-shelf ResNet-50 features | 58% |
| Simple geometry (size, aspect, intensity) | 78% |
| **Box size alone** | **64%** |
| Our trained thermal encoder | 90% |

The 90% overstates what we actually have. Box size alone already reaches 64%, and within a
single video a deer's range changes slowly — so most of that "identification" is matching
**distance**, not the animal. Appearance on its own is weak: note that the pretrained CNN
scores *below* plain geometry, and brightness carries no signal whatsoever.

There is also a hard limit on validation. **No deer in our corpus appears in two videos.**
The only two road segments filmed twice have a deer on one pass and none on the other, so
we have zero examples of the same animal seen twice — nothing to test cross-video
identification against, even if the resolution supported it.

---

## The decision I would like your steer on

Given the above, how would you like this framed and scoped?

1. **Track association only** — present it as preventing double counting within a video,
   state the resolution limit explicitly, and make no individual-ID claim.
2. **Individual re-identification** — keep the stronger framing, which would mean
   collecting repeat-visit footage so the same animals are recorded more than once and the
   claim can actually be tested.
3. Something else you have in mind.

I have a run finishing shortly that measures whether this improves the count at all; I will
send that number as soon as it lands. Happy to go either direction — I would rather you
both set the scope than have me pick it.

---

*Supporting detail: `docs/RESULTS_LOG.md`; jobs 2788117 (feasibility), 2788317 (trained
encoder), 2788316 (counting impact, running).*
