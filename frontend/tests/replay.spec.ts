// Portable synthetic data tests the offline viewer, never model accuracy.
import {test, expect} from "@playwright/test";
import {execFileSync} from "node:child_process";
import {createHash, randomUUID} from "node:crypto";
import {mkdirSync, readFileSync, writeFileSync} from "node:fs";
import path from "node:path";
import {fileURLToPath, pathToFileURL} from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
const dir = path.join(root, ".artifacts", "replay-tests", randomUUID());
const output = path.join(dir, "replay.html");
const manifestFile = path.join(dir, "manifest.json");
const recordFile = path.join(dir, "record.json");
const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a0s8AAAAASUVORK5CYII=", "base64");
const source = {source_sha256: createHash("sha256").update(png).digest("hex"), raster_sha256: "synthetic", width: 1, height: 1, page_index: 0};
const region = {image_id: "crop-a", canonical_bbox: [0, 0, 1, 1], page_index: 0, sent_size: [1, 1], private_path: "DO_NOT_COPY"};
const events = [
  {sequence: 1, stage: "get_algorithms", state: "running", elapsed: 1, classes: ["door"]},
  {sequence: 2, stage: "get_algorithms", state: "completed", elapsed: 3, seconds: 2, classes: ["door"], delivered_ids: ["ml-1", "rule-1"]},
  {sequence: 3, stage: "inspect_nemotron", state: "running", elapsed: 4, seconds: null, images: [region, {...region, image_id: "crop-b"}], question: "Compare crop-a and crop-b.", inspection_id: "worker"},
  {sequence: 4, stage: "inspect_nemotron", state: "completed", elapsed: 5, seconds: 1, inspection_id: "worker", advisory: {findings: ["A symbol needs review."], verdict: "uncertain", private_reasoning: "DO_NOT_COPY"}},
  {sequence: 5, stage: "answer", state: "completed", elapsed: 6},
];
const record = {request_id: "synthetic-replay", status: "completed", source, events,
  question: "<script>window.replayInjection=true</script>", raw_provider: "DO_NOT_COPY",
  evidence: {version: 1, status: "available", source, shapes: [
    {id: "ml-1", class: "door", layer: "ml", bbox: [0, 0, 1, 1]},
    {id: "rule-1", class: "door", layer: "deterministic", bbox: [0, 0, 1, 1]},
  ], comparisons: []},
  result: {status: "ok", source, answer: {status: "answered", answer: 1, unit: "door", explanation: "Synthetic recorded answer.", coverage: "Fixture only.", uncertainties: ["A symbol needs review."]}, canonical_citations: [region], raw_provider: "DO_NOT_COPY"},
};

test.beforeAll(() => {
  mkdirSync(dir, {recursive: true});
  const image = path.join(dir, "synthetic.png");
  writeFileSync(image, png);
  writeFileSync(recordFile, JSON.stringify(record));
  writeFileSync(manifestFile, JSON.stringify({image, sourceRevision: "synthetic test", examples: [{record: recordFile, title: "Synthetic replay", note: "Explicit test fixture."}]}));
  execFileSync(process.execPath, [path.join(root, "tools/build_visual_replay.mjs"), manifestFile, output], {cwd: root, timeout: 60000, stdio: "pipe"});
});

test("offline replay preserves tool states, crop identity and controls without requests", async ({page}) => {
  const errors: string[] = [], requests: string[] = [];
  page.on("pageerror", e => errors.push(e.message));
  page.on("request", r => {if (/^https?:/.test(r.url())) requests.push(r.url());});
  await page.clock.install();
  await page.goto(pathToFileURL(output).href);
  await expect(page.getByTestId("replay-position")).toContainText("Event 1/5");
  await page.getByRole("button", {name: "Play", exact: true}).click();
  await page.clock.runFor(2100);
  await page.getByRole("button", {name: "Pause", exact: true}).click();
  await expect(page.getByTestId("replay-position")).toContainText("Event 2/5");
  await page.getByRole("button", {name: "Explore raw ML + rule candidates"}).click();
  await page.getByLabel("ML detections", {exact: true}).check();
  await page.getByLabel("Deterministic evidence", {exact: true}).check();
  await expect(page.getByTestId("algorithm-overlay")).toHaveAttribute("data-shapes", "2");
  await page.getByRole("button", {name: "Back to conversation", exact: true}).click();
  await page.getByRole("button", {name: "Nemotron", exact: true}).click();
  await expect(page.locator(".live-region-label")).toHaveText(["Nemotron · Detail 1 · View 1", "Nemotron · Detail 1 · View 2"]);
  await expect(page.locator(".work-crops button small")).toHaveText(["crop-a · Sent 1 × 1 px · page 1", "crop-b · Sent 1 × 1 px · page 1"]);
  await page.getByRole("button", {name: "Recorded result", exact: true}).click();
  await expect(page.getByTestId("answer-value")).toHaveText("1 door");
  await page.locator(".work-recap > summary").click();
  await expect(page.locator(".work-recap")).toContainText("uncertainty remains");
  await page.getByRole("button", {name: "Restart replay", exact: true}).click();
  await expect(page.getByTestId("agent-answer")).toHaveCount(0);
  await expect(page.getByTestId("algorithm-overlay")).toHaveAttribute("data-shapes", "0");
  expect(await page.evaluate(() => (window as unknown as {replayInjection?: boolean}).replayInjection)).toBeUndefined();
  expect(readFileSync(output, "utf8")).not.toContain("DO_NOT_COPY");
  expect(requests).toEqual([]);
  expect(errors).toEqual([]);
});

test("builder refuses records from a different original", () => {
  const mismatch = path.join(dir, "mismatch.json");
  writeFileSync(mismatch, JSON.stringify({...record, source: {...source, source_sha256: "wrong-image"}}));
  const manifest = JSON.parse(readFileSync(manifestFile, "utf8"));
  manifest.examples[0].record = mismatch;
  const badManifest = path.join(dir, "bad-manifest.json");
  writeFileSync(badManifest, JSON.stringify(manifest));
  expect(() => execFileSync(process.execPath, [path.join(root, "tools/build_visual_replay.mjs"), badManifest, path.join(dir, "bad.html")], {cwd: root, stdio: "pipe"})).toThrow(/does not match/);
});
