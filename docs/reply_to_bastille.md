# Reply — ReID scope question (Dr. Bastille)

**Subject:** Re: ReID — temporary ID vs. identifying individuals

---

Both are possible in principle, but our measurements say only the first is real, and you
are right about the resolution. What is running is a temporary ID, used within a video to
keep deer separate and avoid double counting, not identification of individuals across
videos.

I tested this directly. Our CVAT tracks are already identity labels, so the model trained
on deer from one set of videos and was tested on 69 deer it had never seen, in videos it
had never seen. Against a chance level of 10 percent, brightness alone gave 10 percent
meaning no signal, thermal silhouette alone gave 44 percent, an off-the-shelf ResNet-50
gave 58 percent, simple geometry gave 78 percent, and our trained thermal encoder reached
90 percent. That 90 overstates it: box size alone already reaches 64 percent, and within
one video a deer's range changes slowly, so most of it is matching distance rather than the
animal.

There is also no way to validate cross-video ID here. No deer in our corpus appears in two
videos, and the two road segments we filmed twice have a deer on one pass and none on the
other.

So I would like your steer. Do we frame this as track association only, stating the
resolution limit and making no individual-ID claim? Or keep the individual
re-identification framing, which would mean collecting repeat-visit footage so the claim
can actually be tested? I am happy either way and would rather you and the team set the
scope.

A run measuring whether this improves the count finishes shortly; I will send that number.

---

*Supporting detail: `docs/RESULTS_LOG.md`; jobs 2788117 (feasibility), 2788317 (trained
encoder), 2788316 (counting impact, running).*
