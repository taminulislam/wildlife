// TRACT project overview deck (6 slides). Run: node build_deck.js <figures_dir> <out.pptx>
// Every number is taken from the manuscript tables (overleaf_springer/sec/*.tex) or from
// results/counting_eval/loso_counting.json; nothing here is a new result.
const pptxgen = require("pptxgenjs");
const path = require("path");

const FIG = process.argv[2];
const OUT = process.argv[3];

const NIGHT = "12161C", INK = "1B2129", SLATE = "5B6775", MIST = "EEF1F4", WHITE = "FFFFFF";
const AMBER = "F2911B", STEEL = "4F6F8F", PALE = "9DB4C8", SOFT = "C9D3DD";
const HEAD = "Cambria", BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "TRACT: counting white-tailed deer in thermal road-transect video";
pres.author = "Taminul Islam";

const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.12 });

function title(slide, text, color) {
  slide.addText(text, { x: 0.6, y: 0.4, w: 12.1, h: 0.9, fontFace: HEAD, fontSize: 30, bold: true,
    color: color || INK, margin: 0, valign: "middle", isTextBox: true });
}
function chip(slide, n, x, y, fill, color) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.46, h: 0.46, fill: { color: fill }, line: { color: fill } });
  slide.addText(String(n), { x, y, w: 0.46, h: 0.46, fontFace: BODY, fontSize: 14, bold: true,
    color: color, align: "center", valign: "middle", margin: 0, isTextBox: true });
}

// ------------------------------------------------------------------ 1. title
{
  const s = pres.addSlide();
  s.background = { color: NIGHT };
  s.addText("TRACT  |  Thermal Road-transect Animal Counting and Tracking", { x: 0.6, y: 0.9, w: 6.6, h: 0.4,
    fontFace: BODY, fontSize: 13, bold: true, color: AMBER, charSpacing: 1, margin: 0, isTextBox: true });
  s.addText("Counting White-Tailed Deer in Vehicle-Mounted Thermal Video", { x: 0.6, y: 1.5, w: 6.6, h: 2.3,
    fontFace: HEAD, fontSize: 38, bold: true, color: WHITE, margin: 0, valign: "top", isTextBox: true });
  s.addText("Detector benchmarking and a staged error decomposition", { x: 0.6, y: 3.95, w: 6.6, h: 0.5,
    fontFace: BODY, fontSize: 18, italic: true, color: SOFT, margin: 0, isTextBox: true });
  s.addText([
    { text: "Taminul Islam, Toqi Tahamid Sarker, Seth J. Morelock,", options: { breakLine: true, color: WHITE } },
    { text: "Guillaume Bastille-Rousseau, Khaled R. Ahmed", options: { breakLine: true, color: WHITE } },
    { text: "School of Computing and Center for Wildlife Sustainability, Southern Illinois University Carbondale",
      options: { color: PALE } },
  ], { x: 0.6, y: 5.2, w: 6.9, h: 1.3, fontFace: BODY, fontSize: 12.5, margin: 0, valign: "top",
       paraSpaceAfter: 6, isTextBox: true });
  s.addImage({ path: path.join(FIG, "dataset/p2_gt.jpg"), x: 7.85, y: 1.25, w: 4.9, h: 3.92,
    altText: "Thermal frame of a white-tailed deer close to the vehicle with its ground-truth box" });
  s.addText("One annotated frame from the corpus: 640 x 512 px thermal, contrast-normalized", {
    x: 7.85, y: 5.27, w: 4.9, h: 0.4, fontFace: BODY, fontSize: 11, color: PALE, margin: 0, isTextBox: true });
  s.addNotes("Project overview. TRACT is a pipeline that counts distinct deer in thermal video recorded from a vehicle " +
    "driving road transects at night, together with a way of measuring where the counting error comes from.");
}

// ------------------------------------------------------------------ 2. problem and data
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "A survey needs animals counted, not boxes placed");
  s.addText([
    { text: "Wildlife pipelines are judged as object detectors, by accuracy per frame (mAP).",
      options: { bullet: true, breakLine: true } },
    { text: "An abundance survey needs the number of distinct animals, which depends on tracking and on deciding which tracks are new animals.",
      options: { bullet: true, breakLine: true } },
    { text: "Our question: does better detection give a better count?", options: { bullet: true, bold: true } },
  ], { x: 0.6, y: 1.5, w: 5.9, h: 2.35, fontFace: BODY, fontSize: 16, color: INK, margin: 0, valign: "top",
       paraSpaceAfter: 10, isTextBox: true });

  const stats = [["32", "thermal road-transect videos, 4 Illinois sites"], ["236", "deer, each annotated as its own track"],
                 ["521,930", "frames at 60 fps, all annotated"], ["29 x 24 px", "median animal size"]];
  stats.forEach((st, i) => {
    const x = 0.6 + (i % 2) * 3.05, y = 4.1 + Math.floor(i / 2) * 1.55;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 2.85, h: 1.35, rectRadius: 0.08, fill: { color: MIST }, line: { color: MIST } });
    s.addText(st[0], { x: x + 0.2, y: y + 0.12, w: 2.45, h: 0.62, fontFace: HEAD, fontSize: 28, bold: true, color: AMBER,
      margin: 0, valign: "middle", isTextBox: true });
    s.addText(st[1], { x: x + 0.2, y: y + 0.74, w: 2.45, h: 0.5, fontFace: BODY, fontSize: 12, color: SLATE,
      margin: 0, valign: "top", isTextBox: true });
  });

  const imgs = [["dataset/p1_raw.jpg", 7.0, 1.5, "Thermal frame as the detector receives it, two deer at different ranges"],
                ["dataset/p3_raw.jpg", 10.0, 1.5, "Thermal frame as the detector receives it, a group of deer"],
                ["dataset/p1_gt.jpg", 7.0, 4.25, "The same frame with ground-truth boxes on two deer"],
                ["dataset/p3_gt.jpg", 10.0, 4.25, "The same frame with ground-truth boxes on the group"]];
  imgs.forEach(im => s.addImage({ path: path.join(FIG, im[0]), x: im[1], y: im[2], w: 2.8, h: 2.24, altText: im[3] }));
  s.addText("What the detector receives", { x: 7.0, y: 3.78, w: 5.8, h: 0.3, fontFace: BODY, fontSize: 11, color: SLATE,
    margin: 0, isTextBox: true });
  s.addText("Ground truth: one track per animal, in every frame", { x: 7.0, y: 6.53, w: 5.8, h: 0.3, fontFace: BODY,
    fontSize: 11, color: SLATE, margin: 0, isTextBox: true });
  s.addNotes("The corpus: 32 nocturnal videos, 8 per site, every deer in every frame annotated in CVAT as an individual track. " +
    "One track is one animal, so the same annotation gives detection, tracking and count ground truth. " +
    "235 of the 236 animals are detectable; one is occluded throughout.");
}

// ------------------------------------------------------------------ 3. method
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Method: a standard pipeline, measured stage by stage");
  s.addImage({ path: path.join(FIG, "wildlife_pipeline.png"), x: 0.6, y: 1.4, w: 12.1, h: 2.757,
    altText: "Pipeline diagram: CLAHE contrast normalization, YOLO11m detection, BoT-SORT tracking with motion compensation, " +
             "three-parameter confirmation rule, count" });

  const cards = [
    ["1", "REACHED", "Did any candidate track touch the animal on at least one frame?", "Measures detection and tracking"],
    ["2", "PRIMARY", "Does the animal have a best-covering track of its own?", "Measures association"],
    ["3", "COUNTED", "Did the confirmation rule accept a track on the animal?", "Measures confirmation"],
  ];
  cards.forEach((c, i) => {
    const x = 0.6 + i * 3.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.45, w: 2.9, h: 2.45, rectRadius: 0.08, fill: { color: MIST },
      line: { color: MIST } });
    chip(s, c[0], x + 0.2, 4.65, AMBER, NIGHT);
    s.addText(c[1], { x: x + 0.8, y: 4.65, w: 1.95, h: 0.46, fontFace: BODY, fontSize: 16, bold: true, color: INK,
      margin: 0, valign: "middle", isTextBox: true });
    s.addText(c[2], { x: x + 0.2, y: 5.25, w: 2.5, h: 0.95, fontFace: BODY, fontSize: 13, color: INK, margin: 0,
      valign: "top", isTextBox: true });
    s.addText(c[3], { x: x + 0.2, y: 6.3, w: 2.5, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: STEEL,
      margin: 0, valign: "middle", isTextBox: true });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.9, y: 4.45, w: 2.8, h: 2.45, rectRadius: 0.08, fill: { color: NIGHT },
    line: { color: NIGHT } });
  s.addText("Held-out protocol", { x: 10.1, y: 4.65, w: 2.4, h: 0.46, fontFace: BODY, fontSize: 16, bold: true, color: AMBER,
    margin: 0, valign: "middle", isTextBox: true });
  s.addText("The rule is fitted by grid search on the 19 training videos, frozen, and reported on 13 videos the detector never saw (83 deer).",
    { x: 10.1, y: 5.25, w: 2.4, h: 1.5, fontFace: BODY, fontSize: 13, color: WHITE, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("Every component is standard: CLAHE normalization, YOLO11m at 640 px, BoT-SORT with global motion compensation and its " +
    "appearance term disabled, orphan recovery, and a confirmation rule with three thresholds (track length of at least 20 frames, " +
    "mean of the five best confidences of at least 0.65). The system does track association within one video; it does not recognise " +
    "individual animals. The contribution is the decomposition: for each ground-truth animal we ask three questions, and the gaps " +
    "between them attribute error to detection, association and confirmation.");
}

// ------------------------------------------------------------------ 4. result 1
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Detection is close to solved; the count is lost afterwards");
  const models = ["YOLOv8m", "ATSS R50", "TOOD R50", "YOLO11m", "RT-DETR-L", "YOLO12m", "YOLOv9m", "YOLOv10m", "DINO R50",
                  "Faster R-CNN", "RTMDet-m"];
  const ap50 = [0.459, 0.380, 0.396, 0.460, 0.434, 0.508, 0.506, 0.510, 0.365, 0.418, 0.466];
  const found = [0.906, 0.962, 0.979, 0.949, 0.983, 0.864, 0.821, 0.885, 0.983, 0.970, 0.996];
  s.addChart(pres.charts.BAR, [
    { name: "AP50 (per box)", labels: models, values: ap50 },
    { name: "Animals found (per animal)", labels: models, values: found },
  ], { x: 0.5, y: 1.4, w: 7.6, h: 5.6, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
       chartColors: [PALE, AMBER], showTitle: true, title: "Eleven detectors, same split and preprocessing",
       titleFontFace: BODY, titleFontSize: 14, titleColor: INK,
       showLegend: true, legendPos: "b", legendFontFace: BODY, legendFontSize: 12, legendColor: INK,
       catAxisLabelColor: SLATE, catAxisLabelFontFace: BODY, catAxisLabelFontSize: 10, catAxisLabelRotate: 315,
       valAxisLabelColor: SLATE, valAxisLabelFontFace: BODY, valAxisLabelFontSize: 10, valAxisMinVal: 0, valAxisMaxVal: 1,
       valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0",
       valGridLine: { color: "E3E7EB", size: 0.5 }, catGridLine: { style: "none" } });

  const st = [["82 to 99.6%", "of animals are found in at least one frame by every detector, although AP50 is only 0.37 to 0.51"],
              ["89 > 80 > 66%", "reached, primary and counted on held-out video: most animals are lost at confirmation"],
              ["MAE 2.38", "animals per video; the pipeline under-counts and almost never over-counts"]];
  st.forEach((r, i) => {
    const y = 1.5 + i * 1.85;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.5, y, w: 4.25, h: 1.65, rectRadius: 0.08, fill: { color: MIST },
      line: { color: MIST } });
    s.addText(r[0], { x: 8.75, y: y + 0.12, w: 3.8, h: 0.62, fontFace: HEAD, fontSize: 26, bold: true, color: AMBER,
      margin: 0, valign: "middle", isTextBox: true });
    s.addText(r[1], { x: 8.75, y: y + 0.76, w: 3.8, h: 0.8, fontFace: BODY, fontSize: 12.5, color: INK, margin: 0,
      valign: "top", isTextBox: true });
  });
  s.addNotes("Per-box metrics look poor because the animals are 27 pixels: a centre error of nine pixels already fails IoU 0.5. " +
    "Scored per animal, every architecture finds 82 to 99.6 percent of the 235 detectable deer. End to end on the 13 held-out " +
    "videos the reported pipeline reaches 89.2 percent of animals, gives 79.5 percent their own track and counts 66.3 percent, " +
    "with a mean absolute error of 2.38 animals per video. The detector chosen for counting is YOLO11m, selected on " +
    "precision-weighted F0.5, not the model with the best AP50.");
}

// ------------------------------------------------------------------ 5. result 2
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Richer candidate pools count fewer animals");
  const pools = ["Baseline (7,008)", "+ lower gate (18,349)", "+ appearance (27,679)", "+ loose NMS (90,239)"];
  s.addChart(pres.charts.BAR, [
    { name: "Reached", labels: pools, values: [74, 79, 82, 79] },
    { name: "Primary", labels: pools, values: [66, 69, 73, 70] },
    { name: "Counted", labels: pools, values: [55, 54, 52, 41] },
  ], { x: 0.5, y: 1.4, w: 6.7, h: 4.35, barDir: "col", barGrouping: "clustered", barGapWidthPct: 55,
       chartColors: [PALE, STEEL, AMBER], showTitle: true, title: "Held-out animals (of 83) by candidate pool size",
       titleFontFace: BODY, titleFontSize: 14, titleColor: INK,
       showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: INK, dataLabelFontFace: BODY,
       showLegend: true, legendPos: "b", legendFontFace: BODY, legendFontSize: 12, legendColor: INK,
       catAxisLabelColor: SLATE, catAxisLabelFontFace: BODY, catAxisLabelFontSize: 10,
       valAxisLabelColor: SLATE, valAxisLabelFontFace: BODY, valAxisLabelFontSize: 10, valAxisMinVal: 0, valAxisMaxVal: 90,
       valAxisMajorUnit: 30, valGridLine: { color: "E3E7EB", size: 0.5 }, catGridLine: { style: "none" } });

  const cs = ["0.00", "0.15", "0.25", "0.35", "0.45", "0.55", "0.65", "0.75", "0.85"];
  s.addChart(pres.charts.LINE, [
    { name: "Predicted count", labels: cs, values: [111, 111, 108, 99, 81, 70, 58, 24, 0] },
    { name: "True count (83)", labels: cs, values: [83, 83, 83, 83, 83, 83, 83, 83, 83] },
  ], { x: 7.4, y: 1.4, w: 5.4, h: 4.35, chartColors: [AMBER, STEEL], lineSize: 2.5, lineDataSymbolSize: 6,
       showTitle: true, title: "Count vs. confidence threshold c", titleFontFace: BODY, titleFontSize: 14, titleColor: INK,
       showLegend: true, legendPos: "b", legendFontFace: BODY, legendFontSize: 12, legendColor: INK,
       showCatAxisTitle: true, catAxisTitle: "confidence threshold c", catAxisTitleFontSize: 10, catAxisTitleColor: SLATE,
       catAxisLabelColor: SLATE, catAxisLabelFontFace: BODY, catAxisLabelFontSize: 10,
       valAxisLabelColor: SLATE, valAxisLabelFontFace: BODY, valAxisLabelFontSize: 10, valAxisMinVal: 0, valAxisMaxVal: 120,
       valAxisMajorUnit: 40, valGridLine: { color: "E3E7EB", size: 0.5 }, catGridLine: { style: "none" } });

  const notes = [
    "Every richer pool reaches more animals than the baseline and counts fewer: the extra candidates are re-sightings of animals already counted.",
    "Lowering c recovers animals but admits about two duplicates for each; the count is unbiased near c = 0.45 (81 of 83).",
    "Learned confirmers (logistic regression, boosting, transformer) lose to the 3-parameter rule: MAE 2.85 to 3.00 vs. 2.38.",
  ];
  notes.forEach((t, i) => {
    const x = 0.6 + i * 4.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.95, w: 3.9, h: 1.1, rectRadius: 0.08, fill: { color: MIST },
      line: { color: MIST } });
    s.addText(t, { x: x + 0.18, y: 6.0, w: 3.54, h: 1.0, fontFace: BODY, fontSize: 12, color: INK, margin: 0,
      valign: "middle", isTextBox: true });
  });
  s.addNotes("The central result. We enriched the candidate pool three ways. The pools are nested, and ordered here by the number of " +
    "candidate tracks they generate. Reached and primary go up, counted goes down: 55, 54, 52, 41 of 83. With 83 animals a difference " +
    "of two or three is not significant on its own; the evidence is the consistent direction. The right chart holds the rule fixed " +
    "and moves only the confidence threshold: the condition acts as a duplicate suppressor, not as a detector threshold. " +
    "Appearance matching is an ablation only; the reported system does not use it.");
}

// ------------------------------------------------------------------ 6. generalization + takeaways
{
  const s = pres.addSlide();
  s.background = { color: NIGHT };
  title(s, "Across sites, and what to take away", WHITE);
  const sites = ["SHB (132)", "TON (50)", "SHW (38)", "MAS (15)"];
  s.addChart(pres.charts.RADAR, [
    { name: "Reached", labels: sites, values: [90.9, 92.0, 94.7, 80.0] },
    { name: "Primary", labels: sites, values: [77.3, 80.0, 78.9, 80.0] },
    { name: "Counted", labels: sites, values: [68.9, 40.0, 28.9, 0.0] },
  ], { x: 0.5, y: 1.4, w: 5.9, h: 4.85, radarStyle: "marker", chartColors: [SOFT, "4F9BE0", AMBER], lineSize: 2.5,
       showTitle: true, title: "Leave-one-site-out, % of each site's animals", titleFontFace: BODY, titleFontSize: 14,
       titleColor: WHITE, showLegend: true, legendPos: "b", legendFontFace: BODY, legendFontSize: 12, legendColor: WHITE,
       catAxisLabelColor: WHITE, catAxisLabelFontFace: BODY, catAxisLabelFontSize: 11,
       valAxisLabelColor: SOFT, valAxisLabelFontFace: BODY, valAxisLabelFontSize: 9, valAxisMinVal: 0, valAxisMaxVal: 100,
       valAxisMajorUnit: 25, valGridLine: { color: "3A4350", size: 0.5 } });
  s.addText("Detection and association transfer to an unseen site; the count does not. An absolute threshold of 0.65 accepts " +
    "nothing at MAS, where no track scores above 0.585.", { x: 0.6, y: 6.3, w: 5.7, h: 0.75, fontFace: BODY, fontSize: 12,
    color: SOFT, margin: 0, valign: "top", isTextBox: true });

  const take = [
    ["Detection is not the bottleneck", "Per animal, detection is near-saturated on this corpus. Confirmation is where animals are lost."],
    ["Improving the candidate pool did not help", "Better detection, association and looser suppression each left the count equal or lower."],
    ["The limit is supervision, not model capacity", "About 200 positive tracks; duplicates sit between true animals and false candidates on every feature."],
    ["Choose the operating point for the purpose", "Minimum error per transect (c = 0.65) or an unbiased total for abundance (c = 0.45); re-fit at a new site."],
  ];
  take.forEach((t, i) => {
    const y = 1.5 + i * 1.3;
    chip(s, i + 1, 6.9, y + 0.05, AMBER, NIGHT);
    s.addText(t[0], { x: 7.55, y, w: 5.2, h: 0.42, fontFace: BODY, fontSize: 16, bold: true, color: WHITE, margin: 0,
      valign: "middle", isTextBox: true });
    s.addText(t[1], { x: 7.55, y: y + 0.44, w: 5.2, h: 0.7, fontFace: BODY, fontSize: 12.5, color: SOFT, margin: 0,
      valign: "top", isTextBox: true });
  });
  s.addText("Dataset, annotations and evaluation code will be released with the paper.", { x: 6.9, y: 6.6,
    w: 5.85, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: AMBER, margin: 0, valign: "middle", isTextBox: true });
  s.addNotes("Leave-one-site-out: train on three sites, count on the fourth, with the confirmation rule unchanged. Pooled over the four " +
    "folds, reached is 91.1 percent, primary 78.3 percent and counted 51.9 percent. Each site was surveyed on a single night, so site " +
    "and night conditions are confounded. Limits of the study: one region, one winter, one species, 236 animals, and a ground truth " +
    "that under-represents distant animals, as any visual survey does.");
}

pres.writeFile({ fileName: OUT }).then(f => console.log("wrote", f));
