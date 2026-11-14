// API responses here are explicit fixtures, never model-accuracy evidence.
import { test, expect, type Page } from "@playwright/test";
import {
  isAnswered,
  resultText,
  visibleCitations,
  type AgentResult,
} from "../src/lib/api";

const png = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a0s8AAAAASUVORK5CYII=",
  "base64",
);
const drawing = {
  document_id: "test-document",
  page: 1,
  image_url: "/api/documents/test-document/image",
  width: 1000,
  height: 600,
};
const result: AgentResult = {
  status: "ok",
  answer: {
    status: "answered",
    answer: 0,
    unit: "doors",
    explanation: "No doors appear in this inspected region.",
    coverage: "One region inspected.",
    uncertainties: ["Whole-page coverage is not established."],
  },
  source: { width: 1000, height: 600, page_index: 0 },
  canonical_citations: [
    { canonical_bbox: [100, 120, 400, 300], page_index: 0 },
  ],
  algorithm_used: false,
  nemotron_calls: 1,
  reflection_calls: 1,
  seconds: 12.5,
};
async function setup(page: Page) {
  await page.route("https://**/*", (r) => r.abort());
  await page.route("**/api/v1/agent", (r) =>
    r.fulfill({
      json: {
        configured: true,
        model: "configured-model",
        connection_verified: false,
      },
    }),
  );
  await page.route("**/api/documents", (r) => r.fulfill({ json: drawing }));
  await page.route("**/api/documents/*/image", (r) =>
    r.fulfill({
      contentType: "image/svg+xml",
      body: '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="600"><rect width="1000" height="600" fill="white"/></svg>',
    }),
  );
  await page.goto("/");
}
async function upload(page: Page, name = "plan.png", mimeType = "image/png") {
  await page
    .getByLabel("Upload drawing", { exact: true })
    .setInputFiles({ name, mimeType, buffer: png });
  await expect(page.getByTestId("source-image")).toBeAttached();
}
async function ask(page: Page) {
  await page.getByLabel("Question about this drawing").fill("How many doors?");
  await page
    .getByRole("button", { name: "Ask Planproof", exact: true })
    .click();
}
test("one assistant with upload first and no model controls", async ({
  page,
}) => {
  await setup(page);
  await expect(
    page.getByRole("heading", { name: "Your plans. Clearly understood." }),
  ).toBeVisible();
  await expect(page.getByTestId("agent-answer")).toHaveCount(0);
  await expect(page.getByRole("checkbox")).toHaveCount(0);
  await expect(page.getByLabel("Question about this drawing")).toHaveCount(0);
  await upload(page);
  await expect(
    page.getByRole("button", { name: "Ask Planproof", exact: true }),
  ).toBeDisabled();
  await page.getByLabel("Question about this drawing").fill("How many doors?");
  await expect(
    page.getByRole("button", { name: "Ask Planproof", exact: true }),
  ).toBeEnabled();
  await expect(page.getByText(/Sending lets Planproof/)).toBeVisible();
  await expect(page.locator("body")).not.toContainText(
    /Gemini|Nemotron|Algorithms|configured-model/,
  );
});
test("sends unchanged agent contract, prevents duplicates, and draws canonical evidence", async ({
  page,
}) => {
  await setup(page);
  await upload(page);
  const bodies: unknown[] = [];
  let release: () => void = () => {};
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/api/v1/documents/*/ask", async (route) => {
    bodies.push(route.request().postDataJSON());
    await gate;
    await route.fulfill({ json: result });
  });
  await ask(page);
  await expect(
    page.getByRole("button", { name: "Ask Planproof", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "New drawing", exact: true }),
  ).toBeDisabled();
  await expect(page.getByTestId("actual-tools")).toHaveCount(0);
  release();
  await expect(page.getByTestId("answer-value")).toHaveText("0 doors");
  expect(bodies).toEqual([
    {
      question: "How many doors?",
      mode: "auto",
      allow_gemini: true,
      allow_nemotron: true,
    },
  ]);
  await expect(page.getByTestId("actual-tools")).toHaveCount(0);
  await page
    .getByRole("button", { name: "View evidence 1, page 1", exact: true })
    .click();
  await expect(page.getByTestId("evidence-overlay")).toHaveAttribute(
    "viewBox",
    "0 0 1000 600",
  );
  await expect(page.getByTestId("evidence-box")).toHaveAttribute("x", "100");
  await expect(page.getByTestId("evidence-box")).toHaveAttribute(
    "width",
    "300",
  );
  await expect(page.getByTestId("evidence-box")).toHaveAttribute(
    "stroke",
    "#005bb5",
  );
  const imageBox = await page.getByTestId("source-image").boundingBox();
  const evidenceBox = await page.getByTestId("evidence-box").boundingBox();
  expect(imageBox!.height / imageBox!.width).toBeCloseTo(0.6, 2);
  expect((evidenceBox!.x - imageBox!.x) / imageBox!.width).toBeCloseTo(0.1, 2);
  expect(evidenceBox!.width / imageBox!.width).toBeCloseTo(0.3, 2);
  await page.getByText("About this answer", { exact: true }).click();
  await expect(
    page.getByText("Whole-page coverage is not established.", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Clear conversation", exact: true })
    .click();
  await expect(page.getByTestId("evidence-box")).toHaveCount(0);
});
test("abstention and outages are not displayed as zero, with no automatic retries", async ({
  page,
}) => {
  await setup(page);
  await upload(page);
  let calls = 0;
  await page.route("**/api/v1/documents/*/ask", (r) => {
    calls++;
    return r.fulfill({
      json: {
        status: "unavailable",
        answer: { ...result.answer, status: "abstain", answer: null },
        reliability: { retry_after_seconds: 20 },
      },
    });
  });
  await ask(page);
  await expect(page.getByTestId("answer-value")).toHaveText(
    "Not enough evidence to answer",
  );
  await expect(
    page.getByText("Please wait 20 seconds before trying again."),
  ).toBeVisible();
  await page.waitForTimeout(250);
  expect(calls).toBe(1);
});
test("a recovered answer stays usable without exposing model machinery", async ({
  page,
}) => {
  await setup(page);
  await upload(page);
  await page.route("**/api/v1/documents/*/ask", (r) =>
    r.fulfill({
      json: { ...result, reliability: { token_repair_succeeded: true } },
    }),
  );
  await ask(page);
  await expect(page.getByTestId("answer-value")).toHaveText("0 doors");
  await expect(page.locator("body")).not.toContainText(
    /provider|token_repair|Nemotron/,
  );
});
test("expired document clears source and requires explicit re-upload", async ({
  page,
}) => {
  await setup(page);
  await upload(page);
  await page.route("**/api/v1/documents/*/ask", (r) =>
    r.fulfill({ status: 404, json: { detail: "missing" } }),
  );
  await ask(page);
  await expect(page.getByRole("alert")).toContainText("expired");
  await expect(page.getByTestId("source-image")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: /Try opening this drawing again/ }),
  ).toBeVisible();
});
test("network failure never restores stale evidence for the new question", async ({
  page,
}) => {
  await setup(page);
  await upload(page);
  await page.route("**/api/v1/documents/*/ask", (r) =>
    r.fulfill({ json: result }),
  );
  await ask(page);
  await expect(page.getByTestId("evidence-box")).toHaveCount(1);
  await page.route("**/api/v1/documents/*/ask", (r) => r.abort("failed"));
  await ask(page);
  await expect(page.getByRole("alert")).toBeVisible();
  await expect(page.getByRole("alert")).toContainText(
    "question may still be processing",
  );
  await expect(page.getByTestId("answer-value").last()).toHaveText(
    "We couldn’t complete this question",
  );
  await expect(page.getByTestId("evidence-box")).toHaveCount(0);
});
test("page upload uses selected PDF page and clears previous answer", async ({
  page,
}) => {
  await setup(page);
  await upload(page, "plan.pdf", "application/pdf");
  await page.route("**/api/v1/documents/*/ask", (r) =>
    r.fulfill({ json: result }),
  );
  await ask(page);
  await expect(page.getByTestId("agent-answer")).toHaveCount(1);
  let body = "";
  await page.route("**/api/documents", (r) => {
    body = r.request().postData() || "";
    return r.fulfill({ json: { ...drawing, page: 2 } });
  });
  await page.getByLabel("PDF page", { exact: true }).fill("2");
  await page.getByRole("button", { name: "Load page", exact: true }).click();
  await expect(page.getByTestId("source-image")).toHaveAttribute(
    "alt",
    /page 2/,
  );
  expect(body).toContain('name="page"\r\n\r\n2');
  await expect(page.getByTestId("agent-answer")).toHaveCount(0);
  await expect(page.getByTestId("evidence-box")).toHaveCount(0);
});
test("untrusted answer text is inert and mobile stays within viewport", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await setup(page);
  await upload(page);
  await page.route("**/api/v1/documents/*/ask", (r) =>
    r.fulfill({
      json: {
        ...result,
        answer: {
          ...result.answer,
          explanation: '<img src=x onerror="window.hacked=true">',
        },
      },
    }),
  );
  await ask(page);
  await expect(page.getByTestId("agent-answer")).toContainText("<img src=x");
  expect(await page.evaluate(() => (window as any).hacked)).toBeUndefined();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});
test("rejects unsupported and oversized originals without upload", async ({
  page,
}) => {
  await setup(page);
  let uploads = 0;
  await page.route("**/api/documents", (r) => {
    uploads++;
    return r.fulfill({ json: drawing });
  });
  await page
    .getByLabel("Upload drawing", { exact: true })
    .setInputFiles({ name: "plan.webp", mimeType: "image/webp", buffer: png });
  await expect(page.getByRole("alert")).toContainText("PDF, PNG or JPEG");
  await page.getByLabel("Upload drawing", { exact: true }).setInputFiles({
    name: "large.png",
    mimeType: "image/png",
    buffer: Buffer.alloc(10 * 1024 * 1024 + 1),
  });
  await expect(page.getByRole("alert")).toContainText("10 MiB");
  expect(uploads).toBe(0);
});
test("canonical filtering excludes local, wrong-page and mismatched-frame boxes", () => {
  expect(
    visibleCitations(
      {
        ...result,
        canonical_citations: [
          ...result.canonical_citations!,
          { page_index: 1, canonical_bbox: [1, 1, 10, 10] },
          { page_index: 0, canonical_bbox: [-1, 0, 10, 10] },
          { page_index: 0, canonical_bbox: [1, 1, 9999, 10] },
        ],
      },
      drawing,
    ),
  ).toHaveLength(1);
  expect(
    visibleCitations(
      { ...result, source: { width: 20, height: 10, page_index: 0 } },
      drawing,
    ),
  ).toEqual([]);
  expect(isAnswered({ ...result, status: "unavailable" })).toBe(false);
  expect(resultText({ status: "not_configured", answer: null })).toContain(
    "available",
  );
});

test("suggestions prepare a question without sending, Enter sends and Shift+Enter adds a line", async ({
  page,
}) => {
  await setup(page);
  await upload(page);
  let calls = 0;
  await page.route("**/api/v1/documents/*/ask", (r) => {
    calls++;
    return r.fulfill({ json: result });
  });
  await page
    .getByRole("button", { name: "What is the drawing title?", exact: true })
    .click();
  const composer = page.getByLabel("Question about this drawing");
  await expect(composer).toHaveValue("What is the drawing title?");
  expect(calls).toBe(0);
  await composer.press("End");
  await composer.press("Shift+Enter");
  await expect(composer).toHaveValue("What is the drawing title?\n");
  expect(calls).toBe(0);
  await composer.press("Enter");
  await expect(page.getByTestId("answer-value")).toHaveText("0 doors");
  expect(calls).toBe(1);
  await expect(composer).toHaveValue("");
});

test("mobile evidence opens the drawing and returns to the conversation", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await setup(page);
  await upload(page);
  await page.route("**/api/v1/documents/*/ask", (r) =>
    r.fulfill({ json: result }),
  );
  await ask(page);
  await page
    .getByRole("button", { name: "View evidence 1, page 1", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Show drawing", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("source-image")).toBeVisible();
  await expect(page.getByTestId("evidence-box")).toBeVisible();
  await page
    .getByRole("button", { name: "Show conversation", exact: true })
    .click();
  await expect(page.getByTestId("answer-value")).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});

test("drag and drop uploads exactly once without a hidden inference call", async ({
  page,
}) => {
  await setup(page);
  let uploads = 0,
    asks = 0;
  await page.route("**/api/documents", (r) => {
    uploads++;
    return r.fulfill({ json: drawing });
  });
  await page.route("**/api/v1/documents/*/ask", (r) => {
    asks++;
    return r.fulfill({ json: result });
  });
  const dataTransfer = await page.evaluateHandle((bytes) => {
    const dt = new DataTransfer();
    dt.items.add(
      new File([new Uint8Array(bytes)], "drop.png", { type: "image/png" }),
    );
    return dt;
  }, Array.from(png));
  await page.locator(".product").dispatchEvent("dragover", { dataTransfer });
  await expect(
    page.getByText("Drop your drawing", { exact: true }),
  ).toBeVisible();
  await page.locator(".product").dispatchEvent("drop", { dataTransfer });
  await expect(page.getByTestId("source-image")).toBeVisible();
  expect(uploads).toBe(1);
  expect(asks).toBe(0);
});

test("narrow welcome and keyboard focus remain usable", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 740 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await setup(page);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.getByRole("button", { name: /Add your drawing/ }).focus();
  await expect(
    page.getByRole("button", { name: /Add your drawing/ }),
  ).toBeFocused();
  await upload(page);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await expect(page.getByLabel("Question about this drawing")).toBeVisible();
});
