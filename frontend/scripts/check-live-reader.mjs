import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { chromium, expect } from "@playwright/test";
import { PDFDocument } from "pdf-lib";

const origin = "https://planproof-silk.vercel.app";
const browser = await chromium.launch();
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  page.setDefaultTimeout(60_000);
  await page.goto(origin, {waitUntil: "domcontentloaded"});
  for (const [id, sha] of [
    ["A-801", "0c6bc07e4e413738cb8db95cffd3356664b25719fa1f2648f99599a3db92f36c"],
    ["A-701", "a635bb2c0d8d3535cfa47759b339b7d0a844836bf983c4afd18add4c5fb01703"],
  ]) {
    const [download] = await Promise.all([
      page.waitForResponse(response => new URL(response.url()).hostname === "raw.githubusercontent.com" &&
        decodeURIComponent(new URL(response.url()).pathname).includes("/" + id + " - ")),
      page.getByRole("button", {name: new RegExp(id)}).click(),
    ]);
    assert.equal(download.status(), 200, id + " publisher response");
    assert.equal(createHash("sha256").update(await download.body()).digest("hex"), sha, id + " publisher bytes");
    const image = page.getByTestId("source-image");
    await expect(image).toHaveAttribute("alt", new RegExp(id + ".*page 1"), {timeout: 60_000});
    await image.evaluate(async node => {
      await node.decode();
      if (node.naturalWidth < 100 || node.naturalHeight < 100) throw new Error("Page preview has no drawing dimensions.");
    });
    await expect(page.locator("header .filename")).toContainText(id);
    await expect(page.getByLabel("PDF page")).toHaveValue("1");
    console.log(id + " opened from its verified public source.");
    await page.getByRole("button", {name: "Planproof home"}).click();
  }

  const doc = await PDFDocument.create();
  const sheet = doc.addPage([500, 300]);
  sheet.drawText("D6 certified fire-resistance rating: 60 minutes", {x: 24, y: 260, size: 12});
  const bytes = Buffer.from(await doc.save());
  await page.getByLabel("Upload drawing", {exact: true}).setInputFiles({
    name: "synthetic-rating.pdf", mimeType: "application/pdf", buffer: bytes,
  });
  await expect(page.getByTestId("source-image")).toHaveAttribute("alt", /synthetic-rating.pdf/);
  await page.getByLabel("Question about this drawing").fill("What is the certified fire-resistance rating of door D6 in minutes?");
  const [inference] = await Promise.all([
    page.waitForResponse(response => response.url() === origin + "/api/reader" && response.request().method() === "POST",
      {timeout: 150_000}),
    page.getByRole("button", {name: "Ask Planproof", exact: true}).click(),
  ]);
  assert.equal(inference.status(), 200, "Production reader status");
  const answer = await inference.json();
  assert.equal(answer.source_sha256, createHash("sha256").update(bytes).digest("hex"));
  assert.deepEqual([answer.status, answer.value, answer.unit, answer.page], ["answered", 60, "min", 1]);
  await expect(page.getByTestId("answer-value")).toHaveText("60 min");
  await page.getByRole("button", {name: "View evidence 1, page 1", exact: true}).click();
  const box = page.getByTestId("evidence-box").first();
  await expect(box).toBeVisible();
  const [x0, y0, x1, y1] = answer.evidence[0].bbox;
  for (const [attribute, expected] of Object.entries({x:x0, y:y0, width:x1-x0, height:y1-y0})) {
    assert.ok(Math.abs(Number(await box.getAttribute(attribute)) - expected) < .01, "Source-coordinate " + attribute);
  }
  console.log("Synthetic native PDF returned 60 min with a matching evidence overlay.");
} finally {
  await browser.close();
}
