// TRACT lab-talk deck (17 slides) with a full speaking script in the notes of every slide.
// Run: node build_deck.js <paper_figures_dir> <step_images_dir> <out.pptx> <script.md>
// Every number comes from the manuscript tables, results/counting_eval/*.csv|json, or the
// pipeline's own track table for the example scene (steps/scene_tracks.csv). No new result.
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const [FIG, STEP, OUT, SCRIPT] = process.argv.slice(2);

const NIGHT = "12161C", INK = "1B2129", SLATE = "5B6775", MIST = "EEF1F4", WHITE = "FFFFFF";
const AMBER = "F2911B", STEEL = "4F6F8F", PALE = "9DB4C8", SOFT = "C9D3DD", RED = "D9534F", GRID = "E3E7EB";
const HEAD = "Cambria", BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "TRACT: counting white-tailed deer in thermal road-transect video";
pres.author = "Taminul Islam";

const scriptParts = [];
let slideNo = 0;
function newSlide(heading, dark, script) {
  const s = pres.addSlide();
  slideNo += 1;
  s.background = { color: dark ? NIGHT : WHITE };
  if (heading) {
    s.addText(heading, { x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontFace: HEAD, fontSize: 30, bold: true,
      color: dark ? WHITE : INK, margin: 0, valign: "middle", isTextBox: true });
  }
  s.addText(String(slideNo), { x: 12.2, y: 7.0, w: 0.55, h: 0.3, fontFace: BODY, fontSize: 10, color: dark ? PALE : SLATE,
    align: "right", margin: 0, isTextBox: true });
  s.addNotes(script);
  scriptParts.push(`## Slide ${slideNo}. ${heading || "Title"}\n\n${script}\n`);
  return s;
}
function card(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || MIST }, line: { color: fill || MIST } });
}
function chip(s, n, x, y) {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.46, h: 0.46, fill: { color: AMBER }, line: { color: AMBER } });
  s.addText(String(n), { x, y, w: 0.46, h: 0.46, fontFace: BODY, fontSize: 14, bold: true, color: NIGHT, align: "center",
    valign: "middle", margin: 0, isTextBox: true });
}
function txt(s, t, x, y, w, h, o) {
  s.addText(t, Object.assign({ x, y, w, h, fontFace: BODY, fontSize: 13, color: INK, margin: 0, valign: "top", isTextBox: true }, o || {}));
}
function headCard(s, x, y, w, h, head, body, o) {
  o = o || {};
  card(s, x, y, w, h, o.fill);
  txt(s, head, x + 0.2, y + 0.15, w - 0.4, 0.4, { fontSize: 15, bold: true, color: o.headColor || INK, valign: "middle" });
  txt(s, body, x + 0.2, y + 0.6, w - 0.4, h - 0.7, { fontSize: 12.5, color: o.bodyColor || INK });
}
function statCard(s, x, y, w, h, big, small) {
  card(s, x, y, w, h);
  txt(s, big, x + 0.2, y + 0.1, w - 0.4, 0.6, { fontFace: HEAD, fontSize: 24, bold: true, color: AMBER, valign: "middle" });
  txt(s, small, x + 0.2, y + 0.72, w - 0.4, h - 0.8, { fontSize: 12, color: SLATE });
}
function stepTag(s, n, name) {
  txt(s, `STEP ${n} OF 4  |  ${name}`, 0.6, 0.12, 8, 0.3, { fontSize: 11, bold: true, color: AMBER, charSpacing: 1, valign: "middle" });
}
const axis = { catAxisLabelColor: SLATE, catAxisLabelFontFace: BODY, catAxisLabelFontSize: 10, valAxisLabelColor: SLATE,
  valAxisLabelFontFace: BODY, valAxisLabelFontSize: 10 };
const frame = (title) => ({ showTitle: true, title, titleFontFace: BODY, titleFontSize: 14, titleColor: INK,
  valGridLine: { color: GRID, size: 0.5 }, catGridLine: { style: "none" } });
const legend = { showLegend: true, legendPos: "b", legendFontFace: BODY, legendFontSize: 12, legendColor: INK };

// ------------------------------------------------------------------ 1. title
{
  const s = newSlide(null, true,
    "Good morning everyone, and thank you for having me. I am Taminul, from the School of Computing. " +
    "Today I will show you how we count white-tailed deer automatically in the thermal videos that were recorded from the vehicle on the road transects. " +
    "We call the system TRACT. I will go through it step by step, so you can see what the computer does with a video, what comes out, how accurate it is, and where it still fails. " +
    "I am a computer scientist, not an ecologist, so please stop me at any point if something is unclear or if I say something about deer that is wrong.");
  txt(s, "TRACT  |  Thermal Road-transect Animal Counting and Tracking", 0.6, 0.9, 6.6, 0.4, { fontSize: 13, bold: true, color: AMBER, charSpacing: 1 });
  txt(s, "Counting White-Tailed Deer in Vehicle-Mounted Thermal Video", 0.6, 1.5, 6.6, 2.3, { fontFace: HEAD, fontSize: 38, bold: true, color: WHITE });
  txt(s, "How the pipeline works, what it produces, and where it loses animals", 0.6, 3.95, 6.6, 0.5, { fontSize: 18, italic: true, color: SOFT });
  s.addText([
    { text: "Taminul Islam, Toqi Tahamid Sarker, Seth J. Morelock,", options: { breakLine: true, color: WHITE } },
    { text: "Guillaume Bastille-Rousseau, Khaled R. Ahmed", options: { breakLine: true, color: WHITE } },
    { text: "School of Computing and Center for Wildlife Sustainability, Southern Illinois University Carbondale", options: { color: PALE } },
  ], { x: 0.6, y: 5.2, w: 6.9, h: 1.3, fontFace: BODY, fontSize: 12.5, margin: 0, valign: "top", paraSpaceAfter: 6, isTextBox: true });
  s.addImage({ path: path.join(FIG, "dataset/p2_gt.jpg"), x: 7.85, y: 1.25, w: 4.9, h: 3.92,
    altText: "Thermal frame of a white-tailed deer close to the vehicle with its ground-truth box" });
  txt(s, "One annotated frame from the corpus: 640 x 512 px thermal, contrast-normalized", 7.85, 5.27, 4.9, 0.4, { fontSize: 11, color: PALE });
}

// ------------------------------------------------------------------ 2. problem
{
  const s = newSlide("A survey needs animals counted, not boxes placed", false,
    "Let me start with the question. In computer vision, a system like this is normally judged as an object detector. " +
    "That means we ask, frame by frame, whether the computer drew a box in the right place. The usual score is called mean average precision. " +
    "But a survey does not need boxes. A survey needs the number of different animals on the transect. " +
    "One deer is visible for many frames, so between a detection and a count there are more decisions. " +
    "The computer must link the boxes of one animal over time, and it must decide whether a track is a new animal or one that was already counted. " +
    "So our question was simple: if we make detection better, does the count get better? " +
    "On the right you can see what the camera gives us. The top row is what the computer receives, and the bottom row is the annotation.");
  headCard(s, 0.6, 1.5, 2.85, 2.3, "What a detector score asks", "Is each box in the right place, frame by frame? Scored with mean average precision (mAP).");
  headCard(s, 3.65, 1.5, 2.85, 2.3, "What a survey asks", "How many different animals were on this transect? Each animal must be counted once.", { fill: NIGHT, headColor: AMBER, bodyColor: WHITE });
  s.addText([
    { text: "Between the two sit three decisions that no detection score measures:", options: { breakLine: true, bold: true } },
    { text: "linking the boxes of one animal over time", options: { bullet: true, breakLine: true } },
    { text: "deciding which tracks are real animals", options: { bullet: true, breakLine: true } },
    { text: "deciding which tracks repeat an animal already counted", options: { bullet: true } },
  ], { x: 0.6, y: 4.1, w: 5.9, h: 2.1, fontFace: BODY, fontSize: 15, color: INK, margin: 0, valign: "top", paraSpaceAfter: 8, isTextBox: true });
  txt(s, "Our question: does better detection give a better count?", 0.6, 6.35, 5.9, 0.45, { fontSize: 16, bold: true, color: AMBER, valign: "middle" });
  const imgs = [["dataset/p1_raw.jpg", 7.0, 1.5, "Thermal frame as the detector receives it, two deer at different ranges"],
                ["dataset/p3_raw.jpg", 10.0, 1.5, "Thermal frame as the detector receives it, a group of deer"],
                ["dataset/p1_gt.jpg", 7.0, 4.25, "The same frame with ground-truth boxes on two deer"],
                ["dataset/p3_gt.jpg", 10.0, 4.25, "The same frame with ground-truth boxes on the group"]];
  imgs.forEach(im => s.addImage({ path: path.join(FIG, im[0]), x: im[1], y: im[2], w: 2.8, h: 2.24, altText: im[3] }));
  txt(s, "What the computer receives", 7.0, 3.78, 5.8, 0.3, { fontSize: 11, color: SLATE });
  txt(s, "Annotation: one box per deer, the same identity in every frame", 7.0, 6.53, 5.8, 0.3, { fontSize: 11, color: SLATE });
}

// ------------------------------------------------------------------ 3. data
{
  const s = newSlide("The data: 32 transects, every deer followed by hand", false,
    "This is the data. We have 32 videos from four sites, eight videos per site. The camera is a FLIR thermal camera on the vehicle. " +
    "Each frame is 640 by 512 pixels, and the camera records 60 frames per second, so in total there are more than half a million frames. " +
    "Every deer in every frame was annotated by hand in a tool called CVAT. The important point is that each animal is one track: the same identity from the first frame it appears to the last. " +
    "That gives us 236 individual deer. Because one track is one animal, the number of tracks in a video is the true count for that video. " +
    "The sites are not equal. SHB has more than half of the animals, and MAS has only 15. " +
    "The animals are also small. The median deer is about 29 by 24 pixels. " +
    "We split the data by video, never by frame. 19 videos were used to train the detector, and 13 videos were kept away from it. All results I show are from those 13 videos, which have 83 deer.");
  s.addChart(pres.charts.BAR, [{ name: "Annotated deer", labels: ["SHB", "TON", "SHW", "MAS"], values: [132, 51, 38, 15] }],
    Object.assign({ x: 0.5, y: 1.4, w: 5.6, h: 4.0, barDir: "col", barGapWidthPct: 60, chartColors: [AMBER], showLegend: false,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelColor: INK, dataLabelFontFace: BODY,
      valAxisMinVal: 0, valAxisMaxVal: 150, valAxisMajorUnit: 50 }, axis, frame("Annotated deer per site (236 in total)")));
  const st = [["32", "videos, 8 per site, recorded at night from the vehicle"], ["521,930", "frames, 640 x 512 px at 60 frames per second"],
              ["236", "individual deer, each one a track from first to last frame"], ["29 x 24 px", "median size of a deer in the image"]];
  st.forEach((r, i) => statCard(s, 6.5 + (i % 2) * 3.2, 1.5 + Math.floor(i / 2) * 1.95, 3.0, 1.75, r[0], r[1]));
  const sp = [["Training", "19 videos, 153 deer", "The detector learns from these."],
              ["Validation", "4 videos, 45 deer", "Used only to pick the best detector checkpoint."],
              ["Test", "9 videos, 38 deer", "Never seen during training."]];
  sp.forEach((r, i) => {
    const x = 0.6 + i * 4.1, held = i > 0;
    card(s, x, 5.65, 3.9, 1.25, held ? NIGHT : MIST);
    txt(s, r[0], x + 0.2, 5.72, 1.6, 0.4, { fontSize: 14, bold: true, color: held ? AMBER : INK, valign: "middle" });
    txt(s, r[1], x + 1.8, 5.72, 1.95, 0.4, { fontSize: 12.5, bold: true, color: held ? WHITE : INK, valign: "middle", align: "right" });
    txt(s, r[2], x + 0.2, 6.18, 3.5, 0.6, { fontSize: 12, color: held ? SOFT : SLATE });
  });
}

// ------------------------------------------------------------------ 4. overview
{
  const s = newSlide("The pipeline in one picture", false,
    "Here is the whole system in one picture. I will explain each part on the next four slides, so this slide is only the map. " +
    "A video goes in on the left. First we improve the contrast of each frame. Second, a detector looks at each frame alone and draws boxes on the deer. " +
    "Third, a tracker links the boxes of the same animal from frame to frame. The result is a list of candidate tracks. " +
    "Fourth, a simple rule decides which candidate tracks are accepted. The count is the number of accepted tracks. " +
    "Only one part is trained with deep learning, and that is the detector. The tracker and the rule have no learned weights. " +
    "One more point, because Guillaume asked about it. The system is deterministic. We ran it three times on the same videos and every box, every track and every count was exactly the same.");
  s.addImage({ path: path.join(FIG, "wildlife_pipeline.png"), x: 0.6, y: 1.4, w: 12.1, h: 2.757,
    altText: "Pipeline diagram: contrast normalization, YOLO11m detection, BoT-SORT tracking with motion compensation, confirmation rule, count" });
  const steps = [["1", "Contrast", "Make faint animals visible (CLAHE)."], ["2", "Detection", "Box and confidence for each deer, in each frame."],
                 ["3", "Tracking", "Link the boxes of one animal over time."], ["4", "Confirmation", "Accept a track or reject it. Count the accepted tracks."]];
  steps.forEach((c, i) => {
    const x = 0.6 + i * 3.1;
    card(s, x, 4.5, 2.9, 1.55);
    chip(s, c[0], x + 0.2, 4.65);
    txt(s, c[1], x + 0.8, 4.65, 1.95, 0.46, { fontSize: 16, bold: true, valign: "middle" });
    txt(s, c[2], x + 0.2, 5.22, 2.5, 0.75, { fontSize: 12.5 });
  });
  card(s, 0.6, 6.25, 12.1, 0.7, NIGHT);
  txt(s, "Deterministic: the same video always gives the same boxes, tracks and count (three repeated runs were identical).",
    0.85, 6.25, 11.6, 0.7, { fontSize: 13.5, color: WHITE, valign: "middle" });
}

// ------------------------------------------------------------------ 5. step 1 CLAHE
{
  const s = newSlide("Contrast normalization makes faint animals visible", false,
    "Step one is contrast. On the left is the frame as the camera recorded it. The deer are only a little warmer than the background, so the image is flat and gray. " +
    "On the right is the same frame after a method called CLAHE. It stretches the contrast in small regions of the image, one region at a time, and it limits how far it stretches so that noise does not explode. " +
    "Nothing is added to the image. We only make the differences that are already there easier to see. " +
    "We apply it to every frame, in training and when we count, and for every model we tested. " +
    "This one step mattered more than the choice of the detection model. Without it, the detection score on the test videos was 0.30. With it, the score was 0.52. " +
    "For the rest of the talk, all the images are contrast-normalized, because that is what the computer sees.");
  stepTag(s, 1, "CONTRAST");
  s.addImage({ path: path.join(STEP, "step1_raw.png"), x: 0.6, y: 1.45, w: 5.95, h: 2.975, altText: "Frame as recorded: five faint deer under trees, low contrast" });
  s.addImage({ path: path.join(STEP, "step2_clahe.png"), x: 6.78, y: 1.45, w: 5.95, h: 2.975, altText: "Same frame after CLAHE: deer and trees clearly visible" });
  txt(s, "As recorded by the camera", 0.6, 4.48, 5.95, 0.3, { fontSize: 11, color: SLATE });
  txt(s, "After contrast normalization (CLAHE)", 6.78, 4.48, 5.95, 0.3, { fontSize: 11, color: SLATE });
  headCard(s, 0.6, 5.0, 3.9, 1.9, "What it does", "Stretches contrast inside small tiles of the image (8 x 8 grid), with a limit so that noise is not amplified.");
  headCard(s, 4.7, 5.0, 3.9, 1.9, "Why we need it", "Thermal contrast is low and it drifts during a transect. A deer is only slightly warmer than the ground behind it.");
  statCard(s, 8.8, 5.0, 3.93, 1.9, "0.30 to 0.52", "detection score (mAP50) on test videos, without and with normalization. Larger than any difference between models.");
}

// ------------------------------------------------------------------ 6. step 2 detection
{
  const s = newSlide("The detector finds deer in each frame, one frame at a time", false,
    "Step two is detection. The detector is a neural network called YOLO11. It looks at one frame, and it returns boxes. " +
    "Each box has a confidence between zero and one, which says how sure the network is that the box contains a deer. You can see the confidence on each box here. " +
    "The network learned this from 12,852 frames taken from the 19 training videos. About 60 percent of those frames have no deer at all. " +
    "That is on purpose. A survey system must learn not to fire on warm rocks, tree trunks or buildings. " +
    "Two things are important. First, the detector has no memory. It does not know that the deer in this frame is the same deer as in the last frame. " +
    "Second, look at the confidences. These are clear deer, but the scores are only between 0.5 and 0.7, because the animals are so small. " +
    "So we keep every box with confidence above 0.10, and we let the later steps decide.");
  stepTag(s, 2, "DETECTION");
  s.addImage({ path: path.join(STEP, "step3_detect.png"), x: 0.6, y: 1.45, w: 7.7, h: 3.85, altText: "Detections on the frame: five green boxes with confidences 0.53 to 0.69" });
  txt(s, "Detector output on one frame of a held-out video (GolfDr, SHB). Numbers are the confidence of each box.", 0.6, 5.36, 7.7, 0.3, { fontSize: 11, color: SLATE });
  s.addText([
    { text: "Input: one contrast-normalized frame.", options: { bullet: true, breakLine: true } },
    { text: "Output: a box and a confidence (0 to 1) for each deer it finds.", options: { bullet: true, breakLine: true } },
    { text: "No memory: every frame is processed alone.", options: { bullet: true, breakLine: true } },
    { text: "Small animals give modest confidence, so all boxes above 0.10 are passed on.", options: { bullet: true } },
  ], { x: 8.7, y: 1.5, w: 4.05, h: 3.8, fontFace: BODY, fontSize: 15, color: INK, margin: 0, valign: "top", paraSpaceAfter: 10, isTextBox: true });
  statCard(s, 0.6, 5.8, 3.9, 1.15, "YOLO11m", "detector, 640 px input");
  statCard(s, 4.7, 5.8, 3.9, 1.15, "12,852 frames", "for training, 7,839 of them with no deer");
  statCard(s, 8.8, 5.8, 3.93, 1.15, "11 models", "compared on the same data (slide 12)");
}

// ------------------------------------------------------------------ 7. step 3 tracking
{
  const s = newSlide("The tracker links boxes into one track per animal", false,
    "Step three is tracking. Now we connect the boxes over time. Each colour here is one track, and the number is the identity the tracker gave it. " +
    "The two pictures are the same scene, 0.3 seconds apart. The whole picture has moved, because the vehicle is driving. " +
    "But every animal keeps its colour and its number. The thin lines show where each track has been. " +
    "The tracker is called BoT-SORT. It does three things. It predicts where each animal will be in the next frame from its recent movement. " +
    "It estimates how the camera moved between two frames and removes that movement. With a vehicle at 25 to 90 kilometres per hour, this is necessary. " +
    "Then it matches the new boxes to the predicted positions. " +
    "I want to be clear about one thing. The tracker uses position and movement only. It does not recognise an individual deer by its appearance. " +
    "We tested appearance, and at this image size it does not carry identity. So this prevents counting the same animal twice inside one transect, and nothing more. " +
    "We also added a small recovery step for boxes that the tracker left without a track. That recovered 17 animals.");
  stepTag(s, 3, "TRACKING");
  s.addImage({ path: path.join(STEP, "step4_track_1.png"), x: 0.6, y: 1.45, w: 5.95, h: 2.975, altText: "Five tracked deer with coloured boxes and identity numbers at 60.87 seconds" });
  s.addImage({ path: path.join(STEP, "step4_track_3.png"), x: 6.78, y: 1.45, w: 5.95, h: 2.975, altText: "The same five deer 0.3 seconds later with the same colours and numbers, shifted left as the vehicle moves" });
  txt(s, "Same scene 0.3 s apart. Colour and ID stay with the animal while the camera moves; lines are the path of each track.", 0.6, 4.48, 12.1, 0.3, { fontSize: 11, color: SLATE });
  headCard(s, 0.6, 5.0, 2.9, 1.9, "Predict", "A Kalman filter predicts where each animal will be in the next frame.");
  headCard(s, 3.67, 5.0, 2.9, 1.9, "Remove camera motion", "Optical flow estimates how the vehicle moved the image, and that shift is removed.");
  headCard(s, 6.74, 5.0, 2.9, 1.9, "Match", "New boxes are matched to predicted positions by overlap. Unmatched boxes are recovered if they line up in time (17 animals).");
  headCard(s, 9.81, 5.0, 2.92, 1.9, "Not identification", "Position and motion only. The system does not recognise individual deer by appearance.", { fill: NIGHT, headColor: AMBER, bodyColor: WHITE });
}

// ------------------------------------------------------------------ 8. step 4 confirmation
{
  const s = newSlide("A simple rule decides which tracks are counted", false,
    "Step four is confirmation, and this is the step that decides the count. The tracker gives far more tracks than animals. " +
    "Over the 32 videos it gives about 7,000 candidate tracks for 236 deer. Most of them are short pieces, false alarms, or second tracks on an animal that already has one. " +
    "So we need a rule that says yes or no to each track. Our rule has two conditions. " +
    "The track must have at least 20 detections, which is one third of a second. And the average of its five best confidences must be at least 0.65. " +
    "We did not choose these numbers by hand. The computer tried 378 combinations on the training videos and kept the one with the smallest counting error. Then we froze it. " +
    "Look at the example. Four tracks are accepted. The small deer in the middle is rejected. Its track has 23 detections, so it is long enough, but its best confidences average 0.60. " +
    "This is a real deer. It is in the annotation. So here you already see the main weakness: the rule is careful, and it loses some real animals. " +
    "Also, two deer on the right stand together, and one box covers both. Six deer are annotated in this frame, and the system counts four.");
  stepTag(s, 4, "CONFIRMATION");
  s.addImage({ path: path.join(STEP, "step5_confirm.png"), x: 0.6, y: 1.45, w: 6.6, h: 3.3, altText: "Four tracks labelled counted in green and one small deer labelled rejected in red" });
  txt(s, "Decision of the rule for the five tracks in this scene. The rejected animal is a real, annotated deer.", 0.6, 4.81, 6.6, 0.3, { fontSize: 11, color: SLATE });
  const ids = ["ID 232 (110)", "ID 238 (108)", "ID 246 (23)", "ID 249 (41)", "ID 250 (49)"];
  s.addChart(pres.charts.BAR, [
    { name: "Accepted", labels: ids, values: [0.735, 0.796, 0, 0.749, 0.776] },
    { name: "Rejected (below 0.65)", labels: ids, values: [0, 0, 0.602, 0, 0] },
  ], Object.assign({ x: 7.4, y: 1.35, w: 5.4, h: 3.8, barDir: "col", barGrouping: "stacked", barGapWidthPct: 50, chartColors: [AMBER, RED],
      showValue: true, dataLabelPosition: "inEnd", dataLabelFormatCode: "0.00;;;", dataLabelFontSize: 11, dataLabelColor: WHITE,
      dataLabelFontFace: BODY, valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.00" },
      axis, legend, frame("Mean of the five best confidences per track")));
  headCard(s, 0.6, 5.3, 3.9, 1.6, "Condition 1: length", "At least 20 detections, which is one third of a second of video.");
  headCard(s, 4.7, 5.3, 3.9, 1.6, "Condition 2: confidence", "The five best detections of the track must average 0.65 or more.");
  headCard(s, 8.8, 5.3, 3.93, 1.6, "Not set by hand", "Chosen by a search over 378 settings on the training videos, then frozen.", { fill: NIGHT, headColor: AMBER, bodyColor: WHITE });
}

// ------------------------------------------------------------------ 9. measurement
{
  const s = newSlide("How we measure where animals are lost", false,
    "Before the results, I need to explain how we measure. The usual way is to report one number, the final count. " +
    "But then a deer that was never detected and a deer that was detected and rejected look the same. " +
    "So for every annotated deer we ask three questions, one after another. " +
    "First, was it reached? That means at least one candidate track touched it in at least one frame. If not, the detector or the tracker lost it. " +
    "Second, is it primary? That means it has a track of its own, which it does not share with another deer. If not, association lost it. " +
    "Third, was it counted? That means the rule accepted a track on it. If not, confirmation lost it. " +
    "The chart shows the 83 deer in the 13 held-out videos. 74 were reached, 66 had their own track, and 55 were counted. " +
    "So you can read directly how many animals each step loses: 9 at detection, 8 at association, and 11 at confirmation. " +
    "All numbers are from videos the detector never saw. If we evaluate on all 32 videos, the result looks 15 points better, and that would be misleading.");
  s.addChart(pres.charts.BAR, [{ name: "Deer", labels: ["Annotated", "Reached", "Primary", "Counted"], values: [83, 74, 66, 55] }],
    Object.assign({ x: 0.5, y: 1.4, w: 5.9, h: 5.5, barDir: "bar", barGapWidthPct: 45, chartColors: [STEEL], showLegend: false,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelColor: INK, dataLabelFontFace: BODY,
      catAxisOrientation: "maxMin", valAxisMinVal: 0, valAxisMaxVal: 90, valAxisMajorUnit: 30 }, axis,
      { catAxisLabelFontSize: 13 }, frame("The 83 deer in the 13 held-out videos")));
  const q = [["1", "REACHED", "Did any candidate track touch the deer in at least one frame?", "If not: lost by detection or tracking"],
             ["2", "PRIMARY", "Does the deer have a track of its own, not shared with another deer?", "If not: lost by association"],
             ["3", "COUNTED", "Did the rule accept a track on this deer?", "If not: lost by confirmation"]];
  q.forEach((c, i) => {
    const y = 1.5 + i * 1.4;
    card(s, 6.7, y, 6.03, 1.25);
    chip(s, c[0], 6.9, y + 0.13);
    txt(s, c[1], 7.5, y + 0.13, 1.6, 0.46, { fontSize: 15, bold: true, valign: "middle" });
    txt(s, c[3], 9.1, y + 0.13, 3.45, 0.46, { fontSize: 12, italic: true, color: STEEL, valign: "middle", align: "right" });
    txt(s, c[2], 6.9, y + 0.68, 5.65, 0.5, { fontSize: 12.5 });
  });
  headCard(s, 6.7, 5.7, 6.03, 1.2, "Held-out videos only",
    "Results come from 13 videos the detector never trained on. Measured on all 32 videos, coverage looks 15.5 points higher.",
    { fill: NIGHT, headColor: AMBER, bodyColor: WHITE });
}

// ------------------------------------------------------------------ 10. output: tracking
{
  const s = newSlide("Output: tracks through overlap and low contrast", false,
    "Now let me show you what comes out. These are six frames from four held-out videos. The colour of a box is the track identity, and the label shows the track number and its score. " +
    "The first two pictures in the top row are the same scene, 0.3 seconds apart. There are five deer, five tracks, and the tracks keep their identity while the camera moves. " +
    "There are two situations where counting usually fails. The first is overlap. When two deer stand close, their boxes overlap, and a tracker can merge them into one animal. " +
    "In these examples they stay separate. The second is low contrast. In the bottom row, in the middle, two deer stand against an almost uniform background. " +
    "Even a person would hesitate there. They are still carried as two tracks. " +
    "The picture at the bottom left shows how different the sizes can be in one frame: one deer is 87 pixels wide and another is 22. " +
    "You can also see that the score goes down when the animal is farther away.");
  s.addImage({ path: path.join(FIG, "qualitative.png"), x: 0.6, y: 1.4, w: 9.0, h: 5.109, altText: "Six thermal frames with coloured track boxes on deer, including overlapping animals and low-contrast scenes" });
  headCard(s, 9.85, 1.4, 2.88, 1.55, "Identity persists", "Top left pair: same scene, 0.3 s apart, five deer, five tracks.");
  headCard(s, 9.85, 3.18, 2.88, 1.55, "Overlap", "Deer standing together are kept as separate tracks.");
  headCard(s, 9.85, 4.96, 2.88, 1.55, "Low contrast", "Two faint deer on a uniform background are still two tracks.");
  txt(s, "Frames from four held-out transects (NShelbyRd, GolfDr, Robinson, NWolfCreek).", 0.6, 6.57, 9.0, 0.3, { fontSize: 11, color: SLATE });
}

// ------------------------------------------------------------------ 11. output: evidence
{
  const s = newSlide("Output: evidence and a confidence for every counted deer", false,
    "The second output is evidence. A count alone is hard to trust. So for every deer that the system counts, it also saves a small sheet like this one. " +
    "Each row is one animal. Each picture is one moment of its track, with the time in the video and the confidence of the detector at that moment. " +
    "So if the system says there were seven deer on a road, you can open seven sheets and check each animal yourself. " +
    "Look at animal B. The confidence moves between 0.66 and 0.81, although every picture is a clean detection of the same deer. " +
    "That is why we give one score per track and not per frame. " +
    "We also convert the score into a calibrated probability, so that 0.8 really means about 80 percent. " +
    "92 percent of the counted animals have a probability of 0.80 or more. And 793 tracks with low probability are not counted automatically. They are marked for a person to review. " +
    "One honest limit: at the very top of the scale the probability is too optimistic. Tracks that are reported as 1.0 are correct about 85 percent of the time.");
  s.addImage({ path: path.join(FIG, "evidence_two.jpg"), x: 1.165, y: 1.35, w: 11.0, h: 4.159, altText: "Two rows of six thermal crops, each row following one deer over time with time stamp and confidence under each crop" });
  statCard(s, 0.6, 5.7, 3.9, 1.25, "92.1%", "of counted deer have confidence 0.80 or higher");
  statCard(s, 4.7, 5.7, 3.9, 1.25, "793 tracks", "with low confidence are flagged for human review");
  statCard(s, 8.8, 5.7, 3.93, 1.25, "One sheet per deer", "so every count can be checked by eye");
}

// ------------------------------------------------------------------ 12. result: detection
{
  const s = newSlide("Result 1: detection is close to solved", false,
    "Now the results. First, detection. We trained eleven different detection models on exactly the same data. " +
    "The blue bars are the usual detection score. They are between 0.37 and 0.51, which looks poor. " +
    "But this score is per box, and it is very strict for small animals. For a deer of 27 pixels, a box that is nine pixels off is counted as wrong, even if it is clearly on the deer. " +
    "The orange bars ask the question a survey asks: was this animal found at least once while it was visible? " +
    "Every model finds between 82 and 99.6 percent of the deer. The model we use, YOLO11m, finds 95 percent. " +
    "So the models look different on the blue score and almost the same on the orange one. " +
    "For counting, the detector is not the weak part. " +
    "We also did not choose the model with the best blue score. We chose the one with the fewest false alarms, because false alarms are what makes the next steps difficult.");
  const models = ["YOLOv8m", "ATSS R50", "TOOD R50", "YOLO11m", "RT-DETR-L", "YOLO12m", "YOLOv9m", "YOLOv10m", "DINO R50", "Faster R-CNN", "RTMDet-m"];
  s.addChart(pres.charts.BAR, [
    { name: "Detection score per box (AP50)", labels: models, values: [0.459, 0.380, 0.396, 0.460, 0.434, 0.508, 0.506, 0.510, 0.365, 0.418, 0.466] },
    { name: "Share of deer found at least once", labels: models, values: [0.906, 0.962, 0.979, 0.949, 0.983, 0.864, 0.821, 0.885, 0.983, 0.970, 0.996] },
  ], Object.assign({ x: 0.5, y: 1.4, w: 7.6, h: 5.6, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60, chartColors: [PALE, AMBER],
      catAxisLabelRotate: 315, valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0" },
      axis, legend, frame("Eleven detectors, same data and preprocessing")));
  statCard(s, 8.5, 1.5, 4.25, 1.65, "82 to 99.6%", "of deer are found at least once by every detector");
  statCard(s, 8.5, 3.35, 4.25, 1.65, "0.37 to 0.51", "is the usual per-box score for the same models. It is strict for 27-pixel animals.");
  headCard(s, 8.5, 5.2, 4.25, 1.75, "Our choice: YOLO11m", "Chosen for few false alarms (precision 0.90), not for the best per-box score.",
    { fill: NIGHT, headColor: AMBER, bodyColor: WHITE });
}

// ------------------------------------------------------------------ 13. result: per video
{
  const s = newSlide("Result 2: counts per transect on unseen videos", false,
    "This is the end-to-end result for each of the 13 held-out videos. The blue bar is the true number of deer from the annotation, and the orange bar is what the system counted. " +
    "Over all 13 videos there are 83 deer and the system reports 58. On average it is wrong by 2.4 deer per video. " +
    "The errors go in one direction. The system counts too few, and it almost never counts too many. Only two videos are over the truth, GolfDr by two and SIron by one. " +
    "For management that is the safe direction. " +
    "The biggest error is NShelbyRd, with 27 deer in large groups. There we lose animals at every step. " +
    "On three videos the count is exactly right. " +
    "I also want to show the bad cases. On AquaCulture and TouchofNature the system counts zero. Those deer are very far away, about 20 pixels, and the detector does not see them. " +
    "So on a transect with only two or three animals, the system can miss all of them, and the average error hides that.");
  const vids = ["NShelbyRd", "GolfDr", "NWolfCreek", "GiantCityRd", "Robinson", "Melvin", "AquaCulture", "N25thBlue", "NMarseilles", "OikosRd", "TouchofNature", "ChipsRd", "SIron"];
  s.addChart(pres.charts.BAR, [
    { name: "Annotated deer", labels: vids, values: [27, 12, 9, 8, 7, 5, 3, 3, 3, 2, 2, 1, 1] },
    { name: "Counted by the pipeline", labels: vids, values: [20, 14, 5, 3, 7, 1, 0, 3, 1, 1, 0, 1, 2] },
  ], Object.assign({ x: 0.5, y: 1.4, w: 8.4, h: 5.6, barDir: "col", barGrouping: "clustered", barGapWidthPct: 50, chartColors: [PALE, AMBER],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: INK, dataLabelFontFace: BODY,
      catAxisLabelRotate: 315, valAxisMinVal: 0, valAxisMaxVal: 30, valAxisMajorUnit: 10 }, axis, legend,
      frame("13 held-out videos: annotated vs. counted")));
  statCard(s, 9.2, 1.5, 3.55, 1.65, "58 of 83", "deer reported over the 13 videos");
  statCard(s, 9.2, 3.35, 3.55, 1.65, "2.38", "mean absolute error, in deer per video");
  headCard(s, 9.2, 5.2, 3.55, 1.75, "Under-counts", "Only 2 of 13 videos are counted too high. The error is in the safe direction.",
    { fill: NIGHT, headColor: AMBER, bodyColor: WHITE });
}

// ------------------------------------------------------------------ 14. result: pools
{
  const s = newSlide("Result 3: more candidates, fewer deer counted", false,
    "This is the main finding, and it surprised us. The natural idea is: if we lose animals, give the system more candidate tracks, so that fewer animals are missed. " +
    "We tried that in three ways. We lowered the threshold of the detector and let the tracker start tracks more easily. We allowed more overlapping boxes. And we added appearance to the tracker. " +
    "On the left, the groups are ordered by the number of candidate tracks, from 7,000 to 90,000. " +
    "The blue bars go up. More animals are reached, and more animals get their own track. With appearance, 82 of 83 deer are reached. " +
    "But the orange bars go down: 55, 54, 52 and 41. More candidates, fewer deer counted. " +
    "The reason is that the new candidates are mostly second sightings of deer that are already counted. They look like real deer, because they are real deer. " +
    "The right chart shows the same thing with one number. If we lower the confidence threshold, we get missed deer back, but for each deer we get about two duplicates. " +
    "At 0.65 we count 58. At 0.45 we count 81, very close to the truth of 83, but individual videos are less accurate. At zero we count 111. " +
    "We also replaced the rule with machine learning models. All of them were worse than the simple rule.");
  const pools = ["Baseline (7,008)", "+ easier start (18,349)", "+ appearance (27,679)", "+ more overlap (90,239)"];
  s.addChart(pres.charts.BAR, [
    { name: "Reached", labels: pools, values: [74, 79, 82, 79] },
    { name: "Primary", labels: pools, values: [66, 69, 73, 70] },
    { name: "Counted", labels: pools, values: [55, 54, 52, 41] },
  ], Object.assign({ x: 0.5, y: 1.4, w: 6.7, h: 4.35, barDir: "col", barGrouping: "clustered", barGapWidthPct: 55, chartColors: [PALE, STEEL, AMBER],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: INK, dataLabelFontFace: BODY,
      valAxisMinVal: 0, valAxisMaxVal: 90, valAxisMajorUnit: 30 }, axis, legend, frame("Deer (of 83) by number of candidate tracks")));
  const cs = ["0.00", "0.15", "0.25", "0.35", "0.45", "0.55", "0.65", "0.75", "0.85"];
  s.addChart(pres.charts.LINE, [
    { name: "Count reported", labels: cs, values: [111, 111, 108, 99, 81, 70, 58, 24, 0] },
    { name: "True count (83)", labels: cs, values: [83, 83, 83, 83, 83, 83, 83, 83, 83] },
  ], Object.assign({ x: 7.4, y: 1.4, w: 5.4, h: 4.35, chartColors: [AMBER, STEEL], lineSize: 2.5, lineDataSymbolSize: 6,
      showCatAxisTitle: true, catAxisTitle: "confidence threshold of the rule", catAxisTitleFontSize: 10, catAxisTitleColor: SLATE,
      valAxisMinVal: 0, valAxisMaxVal: 120, valAxisMajorUnit: 40 }, axis, legend, frame("Count vs. confidence threshold")));
  const notes = ["More candidates reach more deer but count fewer. The extra tracks are second sightings of deer already counted.",
                 "Lowering the threshold brings back missed deer, with about two duplicates for each one.",
                 "Machine learning in place of the rule was worse: error 2.85 to 3.00 deer per video, against 2.38."];
  notes.forEach((t, i) => { card(s, 0.6 + i * 4.1, 5.95, 3.9, 1.0); txt(s, t, 0.78 + i * 4.1, 5.95, 3.54, 1.0, { fontSize: 12, valign: "middle" }); });
}

// ------------------------------------------------------------------ 15. result: size / range
{
  const s = newSlide("Result 4: chance of being counted, by size in the image", false,
    "This slide is for your field, and it is where I would like your advice. We do not know the distance to each deer. " +
    "But a deer that is far away is small in the image, so the size of the box is a rough measure of distance. " +
    "The blue bars show how often a deer is detected, and the orange bars show how often it is counted, for each size. The number of deer in each group is under the bars. " +
    "Detection stays high. Even under 20 pixels, 78 percent of the deer are detected, and above 30 pixels almost all. " +
    "Counting falls much faster. Between 20 and 30 pixels, where most of our deer are, 57 percent are counted. Under 20 pixels, only 22 percent. " +
    "So distance costs us animals mainly at the confirmation step, not at detection. " +
    "Very large deer, above 100 pixels, are also counted less. The detector never saw deer that large in training. " +
    "I think this curve is close to what you call a detection function in distance sampling. " +
    "But box size is not only distance. A fawn nearby and an adult far away can have the same size. " +
    "There is also a limit in our annotation. Under about 20 pixels we have far fewer annotated deer than the geometry predicts. So our ground truth also misses distant animals, like any visual survey.");
  const bins = ["under 20 px (37)", "20 to 30 px (91)", "30 to 40 px (52)", "40 to 60 px (39)", "60 to 100 px (10)", "over 100 px (6)"];
  s.addChart(pres.charts.BAR, [
    { name: "Detected (%)", labels: bins, values: [78, 96, 100, 100, 100, 83] },
    { name: "Counted (%)", labels: bins, values: [22, 57, 81, 79, 100, 50] },
  ], Object.assign({ x: 0.5, y: 1.4, w: 8.0, h: 5.6, barDir: "col", barGrouping: "clustered", barGapWidthPct: 55, chartColors: [PALE, AMBER],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: INK, dataLabelFontFace: BODY,
      valAxisMinVal: 0, valAxisMaxVal: 110, valAxisMajorUnit: 25 }, axis, legend,
      frame("235 deer by box size (number of deer in brackets)")));
  headCard(s, 8.8, 1.5, 3.95, 1.65, "Distance costs counts", "Small, distant deer are usually detected, but their tracks are too weak to be accepted.");
  headCard(s, 8.8, 3.35, 3.95, 1.65, "Box size is a proxy", "It mixes distance with body size and posture. It needs checking against real distances.");
  headCard(s, 8.8, 5.2, 3.95, 1.75, "A detection function?", "This curve could feed distance sampling. The annotation itself thins out below about 20 px.",
    { fill: NIGHT, headColor: AMBER, bodyColor: WHITE });
}

// ------------------------------------------------------------------ 16. result: sites
{
  const s = newSlide("Result 5: a new site needs its own threshold", false,
    "The last result is about new sites. We trained the detector on three sites and counted on the fourth site, which it had never seen. We did this four times, once for each site. " +
    "Each corner of this chart is one site. The gray and blue lines are reached and primary. They stay large for all four sites. So detection and tracking transfer to a new site quite well. " +
    "The orange line is counted. It is good for SHB, with 69 percent, and then it falls: 40 percent at TON, 29 percent at SHW, and zero at MAS. " +
    "At MAS the system detected 12 of the 15 deer, and it counted none. " +
    "The reason is the threshold of 0.65. When the detector has never seen a site, all its confidences are a little lower. At MAS the best track reaches only 0.585, so nothing passes. " +
    "I have to add a caution. Each site was recorded on one night. So we cannot separate the site from the weather of that night. Temperature and humidity change the thermal contrast. " +
    "The practical message is this: for a new site or a new night, we need a small number of annotated videos from it, to set the threshold again.");
  const sites = ["SHB (132)", "TON (50)", "SHW (38)", "MAS (15)"];
  s.addChart(pres.charts.RADAR, [
    { name: "Reached", labels: sites, values: [90.9, 92.0, 94.7, 80.0] },
    { name: "Primary", labels: sites, values: [77.3, 80.0, 78.9, 80.0] },
    { name: "Counted", labels: sites, values: [68.9, 40.0, 28.9, 0.0] },
  ], Object.assign({ x: 0.5, y: 1.4, w: 6.4, h: 5.6, radarStyle: "marker", chartColors: [PALE, STEEL, AMBER], lineSize: 2.5,
      catAxisLabelFontSize: 12, valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25 }, axis, { catAxisLabelColor: INK }, legend,
      frame("Site held out from training, % of its deer")));
  s.addChart(pres.charts.BAR, [{ name: "Best track score", labels: ["SHB", "TON", "SHW", "MAS"], values: [0.862, 0.782, 0.752, 0.585] }],
    Object.assign({ x: 7.1, y: 1.4, w: 5.7, h: 3.4, barDir: "col", barGapWidthPct: 60, chartColors: [STEEL], showLegend: false,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 11, dataLabelColor: INK,
      dataLabelFontFace: BODY, valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.00" }, axis,
      frame("Highest track score at the unseen site (threshold 0.65)")));
  headCard(s, 7.2, 5.0, 2.7, 1.95, "What transfers", "Detection and tracking: 91% reached and 78% primary over the four sites.");
  headCard(s, 10.05, 5.0, 2.7, 1.95, "What does not", "The fixed threshold. One night per site, so site and weather cannot be separated.",
    { fill: NIGHT, headColor: AMBER, bodyColor: WHITE });
}

// ------------------------------------------------------------------ 17. takeaways + next
{
  const s = newSlide("What we learned, and what we would like to do with you", true,
    "Let me finish with what we learned and what comes next. " +
    "First, detection is not the bottleneck. Almost every deer is found. The animals are lost later, when the system decides whether a track is a new animal. " +
    "Second, giving the system more candidates did not help. It made the count worse. " +
    "Third, the limit is the amount of data, not the model. We have about 200 real tracks to learn from. We think the next step needs a few thousand. " +
    "Fourth, the threshold is a choice. For one transect, the careful setting is better. For a total over many transects, a lower threshold of 0.45 gives an almost unbiased total, 81 of 83. " +
    "On the right are the things where we need your help. We would like to check box size against real distances, so that the size curve can become a detection function for distance sampling. " +
    "We would like repeated nights at the same site, to separate site from weather. And we need a small annotated sample from every new site. " +
    "The dataset and the code will be released with the paper, and there is a small software tool: you give it a video, and it returns the video with boxes, tracks and the count. " +
    "Thank you. I am happy to take questions.");
  txt(s, "What we learned", 0.6, 1.4, 6.0, 0.4, { fontSize: 14, bold: true, color: AMBER, charSpacing: 1 });
  const take = [["Detection is not the bottleneck", "Per animal, almost every deer is found. Confirmation is where deer are lost."],
                ["More candidates did not help", "Better detection, tracking and looser filtering each left the count equal or lower."],
                ["The limit is data, not the model", "About 200 real tracks. Duplicates look like real deer, because they are real deer."],
                ["The threshold is a choice", "0.65 for one transect; 0.45 for an almost unbiased total (81 of 83)."]];
  take.forEach((t, i) => {
    const y = 1.95 + i * 1.22;
    chip(s, i + 1, 0.6, y + 0.05);
    txt(s, t[0], 1.25, y, 5.2, 0.42, { fontSize: 16, bold: true, color: WHITE, valign: "middle" });
    txt(s, t[1], 1.25, y + 0.44, 5.2, 0.7, { fontSize: 12.5, color: SOFT });
  });
  txt(s, "What we would like to do with you", 7.0, 1.4, 5.7, 0.4, { fontSize: 14, bold: true, color: AMBER, charSpacing: 1 });
  const next = [["Box size against real distance", "Validate the proxy, then build a detection function for distance sampling."],
                ["Repeated nights at one site", "To separate the effect of the site from the weather of the night."],
                ["A small annotated sample per new site", "To set the threshold again before counting there."],
                ["Dataset, code and a software tool", "Video in; video with boxes, tracks and the count out."]];
  next.forEach((t, i) => {
    const y = 1.95 + i * 1.22;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.0, y, w: 5.73, h: 1.08, rectRadius: 0.08, fill: { color: "1E2530" }, line: { color: "1E2530" } });
    txt(s, t[0], 7.2, y + 0.08, 5.35, 0.4, { fontSize: 15, bold: true, color: WHITE, valign: "middle" });
    txt(s, t[1], 7.2, y + 0.5, 5.35, 0.5, { fontSize: 12.5, color: SOFT });
  });
}

fs.writeFileSync(SCRIPT, "# TRACT lab talk: speaking script\n\n" +
  "One section per slide. The same text is in the speaker notes of each slide.\n\n" + scriptParts.join("\n"));
pres.writeFile({ fileName: OUT }).then(f => console.log("wrote", f, "slides:", slideNo));
