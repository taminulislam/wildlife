# Reply — ReID scope question (Dr. Bastille)

**Subject:** Re: ReID — temporary ID vs. identifying individuals

---

Both are possible in principle, but our measurements say only the first is real, and you
are right about the resolution. What is currently running is a temporary ID, used within a
video to keep deer separate and avoid double counting. It is not identification of
individuals across videos.

I tested whether true individual ID is even feasible on our data before building anything
on top of it. Our CVAT tracks are already identity labels, since one track is one animal,
so this could be scored properly: the model trained on deer from one set of videos and was
then tested on 69 deer it had never seen, in videos it had never seen. Against a chance
level of 10 percent, brightness alone gave 10 percent, meaning no signal at all; thermal
silhouette alone, with size normalised out, gave 44 percent; an off-the-shelf ResNet-50
gave 58 percent; simple geometry combining size, aspect and intensity gave 78 percent; and
our trained thermal encoder reached 90 percent.

That 90 percent overstates what we actually have. Box size on its own already reaches 64
percent, and within a single video a deer's range changes slowly, so most of what looks
like identification is really matching distance rather than matching the animal. Appearance
by itself is weak, and the clearest sign of that is the pretrained CNN scoring below plain
geometry, with brightness carrying no information whatsoever.

There is also a hard limit on validation that I do not think we can engineer around. No
deer in our corpus appears in two videos. The only two road segments we filmed twice have a
deer on one pass and none on the other, so we have zero examples of the same animal
recorded more than once, and therefore nothing to test cross-video identification against
even if the resolution supported it.

Given all of that, I would like your steer on how to frame and scope this. One option is to
present it as track association only, meaning we describe it as preventing double counting
within a video, state the resolution limit explicitly, and make no individual-ID claim. The
other is to keep the stronger individual re-identification framing, which would mean
collecting repeat-visit footage so the same animals are recorded more than once and the
claim can actually be tested. If you have a third direction in mind I am happy to follow
it. I would rather you and the team set the scope than have me pick it.

A run measuring whether any of this actually improves the count is finishing shortly, and I
will send that number as soon as it lands.

---

*Supporting detail: `docs/RESULTS_LOG.md`; jobs 2788117 (feasibility), 2788317 (trained
encoder), 2788316 (counting impact, running).*
