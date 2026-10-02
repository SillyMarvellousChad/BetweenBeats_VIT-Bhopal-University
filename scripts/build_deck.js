// Builds docs/Antaral_BetweenBeats_presentation.pptx
const pptxgen = require("pptxgenjs");
const path = require("path");
const A = (f) => path.join(__dirname, "..", "docs", "deck_assets", f);

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "Antaral — Team Between Beats";
pres.author = "Nishant Majumdar";

const C = {
  ink: "0B2B2E", teal: "0F766E", mint: "5EEAD4", tint: "F0FDFA", red: "DC2626",
  redTint: "FEF2F2", purple: "6D28D9", text: "1C1917", muted: "57534E", line: "E7E5E4", white: "FFFFFF",
};
const H = "Cambria", B = "Calibri";
const W = 13.33;

function kicker(s, label, dark = false) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 0.45, w: label.length * 0.095 + 0.5, h: 0.34,
    rectRadius: 0.17, fill: { color: dark ? "134E4A" : C.tint }, line: { color: dark ? "134E4A" : C.tint } });
  s.addText(label.toUpperCase(), { x: 0.6, y: 0.45, w: label.length * 0.095 + 0.5, h: 0.34, align: "center",
    fontFace: B, fontSize: 10.5, bold: true, color: dark ? C.mint : C.teal, charSpacing: 1.5, margin: 0, isTextBox: true });
}
function title(s, t, dark = false, y = 0.95) {
  s.addText(t, { x: 0.6, y, w: W - 1.2, h: 0.85, fontFace: H, fontSize: 34, bold: true,
    color: dark ? C.white : C.text, margin: 0, isTextBox: true });
}
function foot(s, t, dark = false) {
  s.addText(t, { x: 0.6, y: 7.0, w: W - 1.2, h: 0.3, fontFace: B, fontSize: 9.5, color: dark ? "99B5B2" : "8A8580",
    margin: 0, isTextBox: true });
}
function card(s, x, y, w, h, fill = C.white, border = C.line) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill },
    line: { color: border, width: 1 } });
}
function kBadge(s, x, y, size = 0.55, dark = false) {
  s.addShape(pres.shapes.OVAL, { x, y, w: size, h: size, fill: { color: dark ? "134E4A" : C.tint },
    line: { color: dark ? C.mint : C.teal, width: 1.25 } });
  s.addText([{ text: "K", options: { bold: true } }, { text: "+", options: { superscript: true, bold: true } }],
    { x, y, w: size, h: size, align: "center", valign: "middle", fontFace: H, fontSize: size * 26,
      color: dark ? C.mint : C.teal, margin: 0, isTextBox: true });
}

// 1 ─ Title ─────────────────────────────────────────────────────────────────────
let s = pres.addSlide();
s.background = { color: C.ink };
s.addImage({ path: A("hero_trace.png"), x: 0.6, y: 0.7, w: 12.1, h: 2.18, transparency: 10 });
kBadge(s, 0.6, 3.25, 0.7, true);
s.addText("Antaral", { x: 1.5, y: 3.05, w: 8, h: 1.1, fontFace: H, fontSize: 64, bold: true, color: C.white, margin: 0, isTextBox: true });
s.addText("अंतराल  ·  the interval", { x: 1.5, y: 4.1, w: 8, h: 0.5, fontFace: B, fontSize: 20, color: C.mint, margin: 0, isTextBox: true });
s.addText("A digital twin for the gap between dialysis sessions", { x: 0.6, y: 4.85, w: 12, h: 0.6,
  fontFace: H, fontSize: 26, italic: true, color: "E6F4F1", margin: 0, isTextBox: true });
s.addText([
  { text: "Team Between Beats", options: { bold: true, color: C.white, breakLine: true } },
  { text: "Nishant Majumdar  ·  B.Tech CSE, VIT Bhopal University", options: { color: "B8D4D0", breakLine: true } },
  { text: "Digital Twin Challenge 2026  ·  Happiest Health", options: { color: "B8D4D0" } },
], { x: 0.6, y: 5.75, w: 9, h: 1.1, fontFace: B, fontSize: 15, margin: 0, paraSpaceAfter: 3, isTextBox: true });
s.addNotes("Hi, I'm Nishant from VIT Bhopal, and this is Antaral, which means 'the interval' in Hindi. For a person on dialysis, the interval between sessions is where the danger lives. Antaral is a digital twin that watches that interval, predicts dangerous potassium before it happens, and helps the dialysis unit decide who needs a chair first.");

// 2 ─ Problem ───────────────────────────────────────────────────────────────────
s = pres.addSlide();
s.background = { color: C.white };
kicker(s, "The problem");
title(s, "The danger lives in the gap between sessions");
const prob = [
  ["~30%", "of patients at one Bengaluru dialysis centre were on twice-weekly dialysis, with higher weight gains and lower dialysis adequacy [3]", C.teal],
  ["Silent", "Potassium builds up between sessions. Too much can trigger a fatal arrhythmia, often with no warning symptoms", C.red],
  ["Day after", "the long gap: deaths and cardiovascular hospitalisations cluster after the longest interval between sessions [1]", C.red],
  ["Blind", "Clinics check potassium on a monthly lab, and every patient keeps fixed weekdays whatever their potassium is doing", C.muted],
];
prob.forEach((p, k) => {
  const x = 0.6 + k * 3.07;
  card(s, x, 2.15, 2.85, 3.6, k === 3 ? "F5F5F4" : (p[2] === C.red ? C.redTint : C.tint), C.white);
  s.addText(p[0], { x: x + 0.25, y: 2.45, w: 2.4, h: 0.95, fontFace: H, fontSize: 36, bold: true, color: p[2], margin: 0, isTextBox: true });
  s.addText(p[1], { x: x + 0.25, y: 3.55, w: 2.4, h: 2.6, fontFace: B, fontSize: 15, color: C.text, valign: "top", margin: 0, isTextBox: true });
});
foot(s, "[1] Foley et al., NEJM 2011   [3] Mukherjee et al., Indian J Nephrol 2016");
s.addNotes("In India many patients dialyse twice a week instead of three times, because of cost and machine shortages. At one Bengaluru centre it was about thirty percent. Between sessions potassium builds up silently, and high potassium can stop the heart. A large US study showed deaths cluster on the day after the long gap. And yet clinics are nearly blind in between: a monthly lab and fixed weekdays.");

// 3 ─ Lakshmi story ─────────────────────────────────────────────────────────────
s = pres.addSlide();
s.background = { color: C.white };
kicker(s, "Meet Lakshmi (synthetic)");
title(s, "Tuesday, 9 PM. Is she safe until Thursday?");
s.addText([
  { text: "58, diabetic, on an ARB. Dialysis Mon & Thu.", options: { bullet: true, breakLine: true } },
  { text: "Drank tender coconut water on Monday and Tuesday, and logged it.", options: { bullet: true, breakLine: true } },
  { text: "Tonight her smartwatch T wave is far taller than her post-dialysis baseline.", options: { bullet: true, breakLine: true } },
  { text: "Her next potassium lab is weeks away.", options: { bullet: true } },
], { x: 0.6, y: 2.1, w: 5.6, h: 2.8, fontFace: B, fontSize: 18, color: C.text, paraSpaceAfter: 10, margin: 0, valign: "top", isTextBox: true });
card(s, 0.6, 5.15, 5.6, 1.45, C.tint, C.tint);
s.addText([
  { text: "The nephrologist's question tonight", options: { bold: true, color: C.teal, breakLine: true } },
  { text: "Should she come in tomorrow, or wait until Thursday?", options: { color: C.text } },
], { x: 0.85, y: 5.3, w: 5.2, h: 1.2, fontFace: B, fontSize: 17, margin: 0, valign: "middle", isTextBox: true });
s.addImage({ path: A("ecg_beats.png"), x: 6.6, y: 2.0, w: 6.2, h: 3.1 });
s.addText("Smartwatch median beat: her baseline vs tonight", { x: 6.6, y: 5.15, w: 6.2, h: 0.35, fontFace: B, fontSize: 12, color: C.muted, margin: 0, isTextBox: true });
foot(s, "All patients and data in Antaral are synthetic, generated by its own simulator.");
s.addNotes("Let me make this concrete. Lakshmi is one of our synthetic patients: 58, diabetic, on an ARB, dialysis Monday and Thursday. She drank coconut water twice, which many people think is healthy, but it's loaded with potassium. Tonight her smartwatch ECG shows a peaked T wave. Her next lab is weeks away. Should she come in tomorrow?");

// 4 ─ Solution overview ────────────────────────────────────────────────────────
s = pres.addSlide();
s.background = { color: C.white };
kicker(s, "What Antaral is");
title(s, "Three twins, one evening decision");
const steps = [
  ["1", "Patient twin", "A potassium model personalised from the EHR, corrected hour by hour by labs and a smartwatch ECG (Kalman filter)", C.teal],
  ["2", "Risk layer", "A calibrated model on top of the twin: risk of K ≥ 6.0 before the next session, with reasons", "B45309"],
  ["3", "Unit twin", "Every night at 9 PM, re-plans tomorrow's chairs so the patients heading for danger go first", C.teal],
  ["4", "Doctor approves", "Dashboard with suggestions, what-if panel and explanations. Nothing changes without a clinician", C.red],
];
steps.forEach((p, k) => {
  const x = 0.6 + k * 3.12;
  card(s, x, 2.2, 2.85, 3.9);
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: 2.45, w: 0.62, h: 0.62, fill: { color: p[3] }, line: { color: p[3] } });
  s.addText(p[0], { x: x + 0.25, y: 2.45, w: 0.62, h: 0.62, align: "center", valign: "middle", fontFace: H, fontSize: 20, bold: true, color: C.white, margin: 0, isTextBox: true });
  s.addText(p[1], { x: x + 0.25, y: 3.25, w: 2.4, h: 0.5, fontFace: H, fontSize: 20, bold: true, color: C.text, margin: 0, isTextBox: true });
  s.addText(p[2], { x: x + 0.25, y: 3.85, w: 2.4, h: 2.1, fontFace: B, fontSize: 14.5, color: C.muted, valign: "top", margin: 0, isTextBox: true });
  if (k < 3) s.addText("→", { x: x + 2.83, y: 3.7, w: 0.3, h: 0.5, fontFace: B, fontSize: 22, color: "A8A29E", align: "center", margin: 0, isTextBox: true });
});
s.addText("Same machines. Same number of sessions per patient. Smarter timing, earlier warnings.", { x: 0.6, y: 6.35, w: 12, h: 0.45,
  fontFace: H, fontSize: 18, italic: true, color: C.teal, margin: 0, isTextBox: true });
s.addNotes("Antaral works at three levels. A patient twin that tracks each person's potassium. A risk layer that turns it into a calibrated warning with reasons. And a unit twin, which is the twist: every evening it re-plans tomorrow's chairs. Finally the doctor approves. Nothing happens automatically.");

// 5 ─ Data fusion ────────────────────────────────────────────────────────────────
s = pres.addSlide();
s.background = { color: C.white };
kicker(s, "Data fusion");
title(s, "Static record + live signals, fused hourly");
const cols = [
  ["Static EHR", C.muted, ["Age, sex, dry weight", "Diagnosis: diabetic nephropathy, CKD of unknown aetiology…", "ACE-i / ARB, beta-blocker", "Residual urine output, bicarbonate", "Dialysis pattern: 2× or 3× a week"]],
  ["Wearables & home", C.purple, ["Smartwatch single-lead ECG, twice a day", "Smart-scale morning weight", "Steps, resting HR, HRV, sleep, BP", "Food diary: coconut water, banana, dry fruits, aloo sabzi"]],
  ["Dialysis unit", C.teal, ["Session times and length", "Pre-dialysis weight", "Potassium lab, about monthly", "Chairs and shifts available"]],
];
cols.forEach((c, k) => {
  const x = 0.6 + k * 4.1;
  card(s, x, 2.15, 3.85, 3.85);
  s.addText(c[0], { x: x + 0.3, y: 2.35, w: 3.3, h: 0.5, fontFace: H, fontSize: 20, bold: true, color: c[1], margin: 0, isTextBox: true });
  s.addText(c[2].map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < c[2].length - 1 } })),
    { x: x + 0.3, y: 3.0, w: 3.35, h: 2.85, fontFace: B, fontSize: 14.5, color: C.text, paraSpaceAfter: 6, valign: "top", margin: 0, isTextBox: true });
});
card(s, 0.6, 6.2, 12.1, 0.62, C.tint, C.tint);
s.addText("Sandbox rules: no real patient data. Antaral's own simulator generates the EHRs, the hidden physiology, the ECG waveforms and the wearables. Next: validation on MIMIC-IV.",
  { x: 0.85, y: 6.2, w: 11.7, h: 0.62, fontFace: B, fontSize: 13.5, color: C.teal, valign: "middle", margin: 0, isTextBox: true });
s.addNotes("The brief asked for fusion of static and dynamic data. Static: the EHR, including drugs that raise potassium and residual urine output. Dynamic: a smartwatch ECG twice a day, weight, steps, heart rate, sleep, and a food diary with very Indian potassium sources. Plus the unit's own records. Everything is synthetic, generated by our simulator, which also hides physiology the twin never sees, so the evaluation is honest.");

// 6 ─ Patient twin ──────────────────────────────────────────────────────────────
s = pres.addSlide();
s.background = { color: C.white };
kicker(s, "1 · Patient twin");
title(s, "A Kalman filter around real potassium physiology");
card(s, 0.6, 2.0, 4.3, 4.5, C.tint, C.tint);
s.addText([
  { text: "Model", options: { bold: true, color: C.teal, breakLine: true } },
  { text: "dK/dt = (absorbed intake − renal + tissue release) / V", options: { fontFace: "Courier New", fontSize: 12, breakLine: true } },
  { text: " ", options: { fontSize: 6, breakLine: true } },
  { text: "Hidden state", options: { bold: true, color: C.teal, breakLine: true } },
  { text: "potassium K and a personal drift b, learned over days", options: { breakLine: true } },
  { text: " ", options: { fontSize: 6, breakLine: true } },
  { text: "Corrected by", options: { bold: true, color: C.teal, breakLine: true } },
  { text: "monthly labs (precise) and smartwatch-ECG estimates (noisy)", options: { breakLine: true } },
  { text: " ", options: { fontSize: 6, breakLine: true } },
  { text: "What-if", options: { bold: true, color: C.teal, breakLine: true } },
  { text: "Monte Carlo rollouts for any session time, food, binder or diet call" },
], { x: 0.85, y: 2.2, w: 3.85, h: 4.1, fontFace: B, fontSize: 14, color: C.text, valign: "top", margin: 0, isTextBox: true });
s.addImage({ path: A("lakshmi_twin.png"), x: 5.15, y: 2.05, w: 7.6, h: 3.04 });
s.addText("Lakshmi's twin: estimate tracks the simulated truth (dotted); forecast crosses 6.0 before Thursday and again before Monday",
  { x: 5.15, y: 5.25, w: 7.6, h: 0.6, fontFace: B, fontSize: 12, color: C.muted, margin: 0, isTextBox: true });
s.addNotes("The patient twin is a mechanistic model of potassium: what you eat, minus what your gut and any remaining kidney function remove, spread over a buffering volume. Dialysis pulls it down toward the bath and it rebounds after. A Kalman filter corrects the model every hour with labs and ECG estimates, and learns a personal drift. Here's Lakshmi: the twin, solid line, tracks the hidden truth, dotted, and the forecast shows danger before Thursday.");

// 7 ─ ECG sensor ────────────────────────────────────────────────────────────────
s = pres.addSlide();
s.background = { color: C.white };
kicker(s, "Smartwatch as a potassium sensor");
title(s, "Reading potassium from a peaked T wave");
s.addImage({ path: A("ecg_beats.png"), x: 0.6, y: 2.0, w: 6.4, h: 3.2 });
s.addText("High potassium: taller, narrower T wave; longer PR; wider QRS. Each beat is compared with the patient's own post-dialysis beat.",
  { x: 0.6, y: 5.35, w: 6.4, h: 0.9, fontFace: B, fontSize: 13.5, color: C.muted, margin: 0, isTextBox: true });
const ecgStats = [["0.50", "mEq/L mean absolute error on held-out patients"], ["0.82", "correlation with true potassium"], ["0.70 → 0.62", "error (RMSE) after personalising to the patient's own baseline"]];
ecgStats.forEach((p, k) => {
  const y = 2.0 + k * 1.35;
  s.addText(p[0], { x: 7.5, y, w: 2.6, h: 0.9, fontFace: H, fontSize: k === 2 ? 28 : 40, bold: true, color: C.purple, valign: "middle", margin: 0, isTextBox: true });
  s.addText(p[1], { x: 10.15, y, w: 2.6, h: 0.9, fontFace: B, fontSize: 14, color: C.text, valign: "middle", margin: 0, isTextBox: true });
});
s.addText("Deliberately noisy, like a real watch: the twin trusts it less than a lab. Deep learning on 2-lead ECGs has detected hyperkalaemia with AUROC 0.85–0.88 in large CKD cohorts [2].",
  { x: 7.5, y: 6.05, w: 5.25, h: 0.8, fontFace: B, fontSize: 12, italic: true, color: C.muted, margin: 0, isTextBox: true });
foot(s, "[2] Galloway et al., JAMA Cardiology 2019");
s.addNotes("This is the bloodless potassium sensor. High potassium makes the T wave taller and narrower. We measure features from the smartwatch's median beat and compare them with the patient's own beat right after dialysis, when potassium is known to be low. Mean error is half a mEq per litre, which is noisy on purpose, so the twin treats it as a weak sensor. Mayo Clinic research showed deep learning can detect hyperkalaemia from ECGs, so this is grounded in real evidence.");

// 8 ─ Prediction results ───────────────────────────────────────────────────────
s = pres.addSlide();
s.background = { color: C.white };
kicker(s, "Outcome 1 · Prediction");
title(s, "Fusion beats every single data stream");
s.addChart(pres.charts.BAR, [{
  name: "AUROC", labels: ["Last monthly lab (today)", "Smartwatch ECG alone", "Physics-only twin", "Twin with wearables", "Antaral final"],
  values: [0.899, 0.840, 0.912, 0.937, 0.966],
}], {
  x: 0.6, y: 2.0, w: 7.2, h: 4.6, barDir: "bar", chartColors: ["A8A29E", "A78BFA", "F59E0B", "14B8A6", "0F766E"],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 12, dataLabelColor: C.text,
  valAxisMinVal: 0.8, valAxisMaxVal: 1.0, valAxisLabelFontSize: 11, catAxisLabelFontSize: 13, catAxisLabelColor: C.text,
  valAxisLabelColor: C.muted, valGridLine: { color: "EEEEEE", size: 0.5 }, catGridLine: { style: "none" },
  showTitle: true, title: "AUROC: K ≥ 6.0 before next session (200 unseen patients)", titleFontSize: 13, titleColor: C.muted,
  barGapWidthPct: 45, varyColors: true,
});
const stats = [["89%", "of dangerous nights caught at 90% specificity", "vs 70% with the last monthly lab"], ["0.85", "within-patient AUROC: spots her dangerous weeks", "vs 0.56 for the lab"], ["13,000", "patient-nights on held-out patients", "95% CI for AUROC: 0.955–0.975"]];
stats.forEach((p, k) => {
  const y = 2.0 + k * 1.58;
  card(s, 8.2, y, 4.55, 1.4, C.tint, C.tint);
  s.addText(p[0], { x: 8.4, y: y + 0.12, w: 1.9, h: 1.15, fontFace: H, fontSize: 34, bold: true, color: C.teal, valign: "middle", margin: 0, isTextBox: true });
  s.addText([{ text: p[1], options: { color: C.text, breakLine: true } }, { text: p[2], options: { color: C.muted, fontSize: 11.5 } }],
    { x: 10.3, y: y + 0.12, w: 2.35, h: 1.15, fontFace: B, fontSize: 13, valign: "middle", margin: 0, isTextBox: true });
});
s.addNotes("Results on two hundred synthetic patients the models never saw. Today's practice, the last monthly lab, scores an AUROC of 0.899. The ECG alone is noisy at 0.84. Our physics-only twin is 0.91. Fusing it with wearables gets 0.937, and the final model 0.966. At ninety percent specificity we catch 89 percent of dangerous nights versus 70. And the key number is 0.85 within a patient: the lab knows who runs high; Antaral knows when.");

// 9 ─ Scheduler ────────────────────────────────────────────────────────────────
s = pres.addSlide();
s.background = { color: C.white };
kicker(s, "Outcome 2 · Unit twin");
title(s, "Re-timing chairs, not adding sessions");
s.addText([
  { text: "Every night at 9 PM an integer programme picks tomorrow's chairs:", options: { breakLine: true } },
  { text: "Maximise danger-hours avoided (K ≥ 6, weighted up at 6.5 and 7.0)", options: { bullet: true, breakLine: true } },
  { text: "Same chairs, same sessions per patient, safe minimum and maximum gaps", options: { bullet: true, breakLine: true } },
  { text: "Looks at both upcoming gaps: moving a session earlier makes the next gap longer", options: { bullet: true, breakLine: true } },
  { text: "Riskiest patients get the earliest shift", options: { bullet: true } },
], { x: 0.6, y: 2.05, w: 5.4, h: 3.6, fontFace: B, fontSize: 16, color: C.text, paraSpaceAfter: 8, valign: "top", margin: 0, isTextBox: true });
s.addChart(pres.charts.BAR, [{ name: "Reduction vs fixed schedule (%)", labels: ["Arrive at dialysis with K ≥ 6", "Episodes of K ≥ 6", "Hours at K ≥ 6", "Hours at K ≥ 7"], values: [14.8, 11.2, 5.4, -0.7] }], {
  x: 6.3, y: 2.0, w: 6.45, h: 4.3, barDir: "col", chartColors: ["0F766E", "14B8A6", "5EEAD4", "D6D3D1"], varyColors: true,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"", dataLabelFontSize: 13, dataLabelColor: C.text,
  valAxisMinVal: -5, valAxisMaxVal: 20, valAxisLabelFontSize: 11, catAxisLabelFontSize: 12, catAxisLabelColor: C.text, valAxisLabelColor: C.muted,
  valGridLine: { color: "EEEEEE", size: 0.5 }, catGridLine: { style: "none" }, showTitle: true,
  title: "Reduction vs today's fixed schedule (60-patient unit)", titleFontSize: 13, titleColor: C.muted, barGapWidthPct: 50,
});
s.addText("6 replicate units × 8 weeks, identical patients, meals, illnesses and session counts. Fewer arrivals with K ≥ 6 in all 6 units.",
  { x: 6.3, y: 6.35, w: 6.45, h: 0.55, fontFace: B, fontSize: 12, color: C.muted, margin: 0, isTextBox: true });
s.addNotes("Now the twist. Each evening an integer programme decides tomorrow's chairs, using the twin's what-if forecasts. It never adds sessions or machines. It looks at both upcoming gaps, because pulling someone in early makes their next gap longer. In simulation, fifteen percent fewer patients arrive at dialysis with dangerous potassium, consistent across all six replicate units. Notice the last bar: it does not reduce time above 7. I'll come back to that honestly.");

// 10 ─ Knowing when not to move ─────────────────────────────────────────────────
s = pres.addSlide();
s.background = { color: C.white };
kicker(s, "Back to Lakshmi");
title(s, "An AI that knows when NOT to move a patient");
card(s, 0.6, 2.1, 5.85, 2.55, C.tint, C.tint);
s.addText("Keep Thursday", { x: 0.9, y: 2.3, w: 5.3, h: 0.5, fontFace: B, fontSize: 22, bold: true, color: C.teal, margin: 0, isTextBox: true });
s.addText("48", { x: 0.9, y: 2.85, w: 1.9, h: 1.3, fontFace: H, fontSize: 64, bold: true, color: C.teal, valign: "middle", margin: 0, isTextBox: true });
s.addText("expected danger-hours over her next two gaps", { x: 2.85, y: 2.95, w: 3.3, h: 1.1, fontFace: B, fontSize: 16, color: C.text, valign: "middle", margin: 0, isTextBox: true });
card(s, 6.9, 2.1, 5.85, 2.55, C.redTint, C.redTint);
s.addText("Move to Wednesday", { x: 7.2, y: 2.3, w: 5.3, h: 0.5, fontFace: B, fontSize: 22, bold: true, color: C.red, margin: 0, isTextBox: true });
s.addText("55", { x: 7.2, y: 2.85, w: 1.9, h: 1.3, fontFace: H, fontSize: 64, bold: true, color: C.red, valign: "middle", margin: 0, isTextBox: true });
s.addText("danger-hours: her next gap would stretch to 5 days, until Monday", { x: 9.15, y: 2.95, w: 3.4, h: 1.1, fontFace: B, fontSize: 16, color: C.text, valign: "middle", margin: 0, isTextBox: true });
s.addText([
  { text: "So Antaral keeps her on Thursday, uses Wednesday's spare chairs for two patients it can actually help, and flags Lakshmi under ", options: {} },
  { text: "\"high risk, schedule kept: consider other action\"", options: { bold: true } },
  { text: ". In the what-if panel the doctor compares a potassium binder or a dietitian call about the coconut water.", options: {} },
], { x: 0.6, y: 5.05, w: 12.1, h: 1.3, fontFace: B, fontSize: 16, color: C.text, margin: 0, valign: "top", isTextBox: true });
s.addNotes("Back to Lakshmi. The obvious answer is to bring her in tomorrow. The twin disagrees: moving her to Wednesday stretches her next gap to five days and makes things worse overall. So it keeps Thursday, gives Wednesday's chairs to two patients it can really help, and flags Lakshmi for a different action, like a binder or a call about the coconut water. That's the value of a twin over a classifier: it reasons about consequences.");

// 11 ─ Dashboard ────────────────────────────────────────────────────────────────
s = pres.addSlide();
s.background = { color: "F5F5F4" };
kicker(s, "The doctor's dashboard");
title(s, "How a doctor works with the virtual patient");
s.addImage({ path: A("unit_tonight_crop.png"), x: 0.6, y: 1.95, w: 5.95, h: 4.28 });
s.addImage({ path: A("patient_twin_crop.png"), x: 6.8, y: 1.95, w: 5.95, h: 4.28 });
s.addText("Unit tonight: suggestions to approve, tomorrow's chair board, risk for every patient", { x: 0.6, y: 6.35, w: 5.95, h: 0.5, fontFace: B, fontSize: 12.5, color: C.muted, margin: 0, isTextBox: true });
s.addText("Patient twin: trajectory, forecast, ECG vs baseline, reasons, what-if panel", { x: 6.8, y: 6.35, w: 5.95, h: 0.5, fontFace: B, fontSize: 12.5, color: C.muted, margin: 0, isTextBox: true });
s.addNotes("This is the working prototype, built in Streamlit. On the left, tonight's review for the whole unit: suggested moves the doctor approves, a chair board for tomorrow, and a risk table. On the right, Lakshmi's twin with her potassium history, the forecast, her ECG against her baseline, and a what-if panel. I'll demo it live in the video.");

// 12 ─ Honest limits & next ────────────────────────────────────────────────────
s = pres.addSlide();
s.background = { color: C.white };
kicker(s, "Honest limits · next steps");
title(s, "What the simulation taught us, and what comes next");
card(s, 0.6, 2.05, 5.9, 4.6, C.redTint, C.redTint);
s.addText("Limits we found", { x: 0.9, y: 2.25, w: 5.4, h: 0.5, fontFace: H, fontSize: 20, bold: true, color: C.red, margin: 0, isTextBox: true });
s.addText([
  "Re-timing does not cut hours above 7.0: those patients need more dialysis, not different timing",
  "Too many schedule changes for a real unit today; the stability setting needs tuning with clinicians",
  "All results are on synthetic data, not clinical validation",
  "A tuned ML model without the twin comes close on accuracy; the twin's edge is what-if reasoning",
].map((t, j, a) => ({ text: t, options: { bullet: true, breakLine: j < a.length - 1 } })),
  { x: 0.9, y: 2.9, w: 5.4, h: 3.6, fontFace: B, fontSize: 14.5, color: C.text, paraSpaceAfter: 8, valign: "top", margin: 0, isTextBox: true });
card(s, 6.85, 2.05, 5.9, 4.6, C.tint, C.tint);
s.addText("Next", { x: 7.15, y: 2.25, w: 5.4, h: 0.5, fontFace: H, fontSize: 20, bold: true, color: C.teal, margin: 0, isTextBox: true });
s.addText([
  "Validate on MIMIC-IV: dialysis patients with potassium labs and linked ECGs",
  "Real smartwatch ECG exports in place of synthetic beats",
  "Dose alerts for chronically high patients: extra session or prescription review",
  "Pilot with an Indian dialysis unit, DPDP-compliant, clinician approval on every change",
].map((t, j, a) => ({ text: t, options: { bullet: true, breakLine: j < a.length - 1 } })),
  { x: 7.15, y: 2.9, w: 5.4, h: 3.6, fontFace: B, fontSize: 14.5, color: C.text, paraSpaceAfter: 8, valign: "top", margin: 0, isTextBox: true });
s.addNotes("I want to be honest about limits. Re-timing doesn't reduce the most extreme hours, because those patients need more dialysis. The scheduler still makes more changes than a real unit would accept. And everything is synthetic. Next steps: validate on MIMIC-IV, use real smartwatch ECG exports, add dose alerts, and pilot with an Indian dialysis unit.");

// 13 ─ Close ───────────────────────────────────────────────────────────────────
s = pres.addSlide();
s.background = { color: C.ink };
s.addImage({ path: A("hero_trace.png"), x: 1.6, y: 5.35, w: 10.6, h: 1.91, transparency: 55 });
kBadge(s, 0.6, 0.9, 0.7, true);
s.addText("Same chairs. Same sessions.\nFewer people arriving in danger.", { x: 0.6, y: 1.85, w: 12, h: 1.9,
  fontFace: H, fontSize: 40, bold: true, color: C.white, margin: 0, isTextBox: true });
s.addText([
  { text: "Antaral  ·  Team Between Beats  ·  Nishant Majumdar, VIT Bhopal University", options: { color: C.mint, breakLine: true } },
  { text: "github.com/SillyMarvellousChad/BetweenBeats_VIT-Bhopal-University", options: { color: "B8D4D0", breakLine: true } },
  { text: "Live demo: betweenbeatsvit-bhopal-university-cmf2hewrxvj5wfdtksbfho.streamlit.app", options: { color: "B8D4D0", breakLine: true } },
  { text: "Research prototype for clinician decision support. Synthetic data only. Not a medical device.", options: { color: "8FB0AC", fontSize: 12 } },
], { x: 0.6, y: 3.85, w: 12, h: 1.2, fontFace: B, fontSize: 15, margin: 0, paraSpaceAfter: 4, isTextBox: true });
s.addNotes("Antaral: same chairs, same sessions, fewer people arriving in danger. Everything is open source on GitHub. Thank you.");

pres.writeFile({ fileName: path.join(__dirname, "..", "docs", "Antaral_BetweenBeats_presentation.pptx") })
  .then((f) => console.log("wrote", f));
