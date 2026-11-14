// Explicit simulated events test presentation and transport, not model accuracy.
import { test, expect, type Page } from "@playwright/test";
const source = {
  raster_sha256: "fixture-raster",
  page_index: 0,
  width: 1000,
  height: 600,
};
const drawing = {
  document_id: "drawing",
  page: 1,
  image_url: "/api/documents/drawing/image",
  width: 1000,
  height: 600,
};
const region = {
  image_id: "crop-1",
  canonical_bbox: [100, 100, 400, 350],
  page_index: 0,
};
const result = {
  status: "ok",
  answer: {
    status: "answered",
    answer: 2,
    unit: "doors",
    explanation: "Two candidates are visible in this crop.",
    coverage: "Crop only; remaining page not established.",
    uncertainties: ["Some symbols remain ambiguous."],
  },
  source,
  canonical_citations: [region],
};
const events = [
  {
    sequence: 1,
    stage: "get_algorithms",
    state: "completed",
    classes: ["room", "door"],
    delivered_ids: ["m1"],
    images: [],
    citations: [],
    elapsed: 1,
    seconds: 0.4,
  },
  {
    sequence: 2,
    stage: "crop_region",
    state: "completed",
    images: [region],
    citations: [],
    elapsed: 2,
    seconds: 0.2,
  },
  {
    sequence: 3,
    stage: "inspect_nemotron",
    state: "completed",
    images: [region],
    citations: [region],
    elapsed: 3,
    seconds: 0.5,
    advisory: {
      findings: ["Two door-like symbols, with a partial edge symbol."],
      verdict: "uncertain",
    },
  },
  {
    sequence: 4,
    stage: "reflect",
    state: "completed",
    checks: ["Check overlapping crops for duplicates."],
    unresolved: ["Remaining page unknown."],
    images: [],
    citations: [],
    elapsed: 4,
  },
];
const bundle = {
  version: 1,
  status: "available",
  source,
  shapes: [
    { id: "m1", class: "door", layer: "ml", bbox: [100, 100, 140, 140] },
    {
      id: "d1",
      class: "door",
      layer: "deterministic",
      bbox: [102, 100, 142, 140],
      method: "quarter-arc-leaf",
      cues: [{ name: "arc_support", value: 0.75 }],
    },
    {
      id: "boundary:r1",
      class: "room",
      layer: "hybrid",
      bbox: [80, 80, 500, 400],
      status: "ml_assisted",
      polygons: [
        {
          outer: [
            [80, 80],
            [500, 80],
            [500, 400],
            [80, 400],
            [80, 80],
          ],
          holes: [],
        },
      ],
    },
  ],
  comparisons: [
    {
      relation: "agreement",
      source_ids: ["m1", "d1"],
      bbox: [100, 100, 142, 140],
      iou: 0.8,
      semantics: "spatial/class agreement, not proof of correctness",
    },
  ],
  warnings: ["Candidates are not verified physical counts."],
};
async function setup(page: Page) {
  await page.route("**/api/**", (r) =>
    r.fulfill({ status: 404, json: { detail: "Fixture resource unavailable." } }),
  );
  await page.route("**/api/documents", (r) => r.fulfill({ json: drawing }));
  await page.route("**/api/v1/documents/*/frame", (r) =>
    r.fulfill({ json: source }),
  );
  await page.route("**/api/documents/*/image", (r) =>
    r.fulfill({
      contentType: "image/svg+xml",
      body: '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="600"><rect width="1000" height="600" fill="white"/></svg>',
    }),
  );
  await page.goto("/");
  await page.getByLabel("Upload drawing", { exact: true }).setInputFiles({
    name: "test.png",
    mimeType: "image/png",
    buffer: Buffer.from("synthetic fixture"),
  });
  await expect(page.getByTestId("source-image")).toBeAttached();
}
async function ask(page: Page) {
  await page.getByLabel("Question about this drawing").fill("How many doors?");
  await page
    .getByRole("button", { name: "Ask Planproof", exact: true })
    .click();
}

test("incomplete door inventory keeps supported and uncertain candidates visible after refresh", async ({ page }) => {
  await setup(page);
  let submissions = 0;
  const counted = {
    ...result,
    completion_reason: "inspection_incomplete",
    answer: { ...result.answer, status: "abstain", answer: null,
      explanation: "Located one supported door opening and one uncertain candidate. This is a partial inventory." },
    canonical_citations: [],
    count_evidence: { unit: "door_opening", supported: 1, uncertain: 1, complete: false,
      objects: [
        { ...region, id: "crop-1-D1", canonical_bbox: [100, 100, 140, 140], status: "supported", reason: "Visible leaf and arc" },
        { ...region, id: "crop-2-D1", canonical_bbox: [250, 100, 285, 145], status: "uncertain", reason: "Partly cut off" },
        { ...region, id: "outside", canonical_bbox: [900, 100, 1200, 200], status: "uncertain", reason: "Invalid frame fixture" },
      ],
    },
  };
  await page.route("**/api/v1/documents/*/runs", r => {
    submissions++;
    return r.fulfill({ json: { request_id: r.request().postDataJSON().request_id,
      status: "completed", source, events: [], last_sequence: 0, result: counted } });
  });
  await ask(page);
  await expect(page.getByTestId("answer-value")).toHaveText("Inspection incomplete");
  await expect(page.getByTestId("count-inventory")).toContainText("Provisional door inventory");
  await expect(page.getByTestId("count-candidate")).toHaveCount(2);
  await expect(page.locator('[data-testid="count-candidate"][data-status="uncertain"]')).toHaveAttribute("stroke-dasharray", "5 3");
  await expect(page.locator('[data-testid="count-candidate"][data-status="supported"]')).toHaveAttribute("stroke", "#42634a");
  await page.getByRole("button", { name: "Show door candidates", exact: true }).click();
  await expect(page.getByTestId("count-candidate")).toHaveCount(0);
  await page.getByRole("button", { name: "Show door candidates", exact: true }).click();
  await expect(page.getByTestId("count-candidate")).toHaveCount(2);
  await page.reload();
  await expect(page.getByTestId("count-candidate")).toHaveCount(2);
  await expect(page.getByTestId("answer-value")).toHaveText("Inspection incomplete");
  expect(submissions).toBe(1);
});

test("door inventory from another source cannot draw boxes on this page", async ({ page }) => {
  await setup(page);
  await page.route("**/api/v1/documents/*/runs", r => r.fulfill({ json: {
    request_id: r.request().postDataJSON().request_id, status: "completed", source,
    events: [], last_sequence: 0,
    result: { ...result, source: { ...source, raster_sha256: "different-source" },
      count_evidence: { unit: "door_opening", supported: 1, uncertain: 0, complete: false,
        objects: [{ ...region, id: "D1", status: "supported", reason: "Different drawing" }] } },
  } }));
  await ask(page);
  await expect(page.getByTestId("agent-answer")).toBeVisible();
  await expect(page.getByTestId("count-candidate")).toHaveCount(0);
});

test("latest task downloads a PDF without rerunning or confusing older citations and drafts", async ({page}) => {
  await setup(page);
  await expect(page.getByRole("button", {name: "Task details", exact: true})).toBeDisabled();
  let submissions = 0, lastId = "", downloads = 0;
  let body: Record<string, string> = {};
  await page.route("**/api/v1/documents/*/runs", r => {
    submissions++;
    lastId = r.request().postDataJSON().request_id;
    return r.fulfill({json: {request_id: lastId, status: "completed", source, events, last_sequence: 4, result}});
  });
  await page.route("**/api/v1/runs/*/report", async r => {
    downloads++;
    body = r.request().postDataJSON();
    expect(r.request().url()).toContain(lastId + "/report");
    await r.fulfill({contentType: "application/pdf", body: "%PDF-1.7\n% Simulated download transport fixture"});
  });
  await ask(page);
  await page.getByLabel("Question about this drawing").fill("Where are the windows?");
  await page.getByRole("button", {name: "Ask Planproof", exact: true}).click();
  await expect(page.getByTestId("answer-value")).toHaveCount(2);
  await page.getByLabel("Question about this drawing").fill("Unsent next question");
  await page.getByRole("button", {name: "View evidence 1, page 1", exact: true}).first().click();
  const button = page.getByRole("button", {name: "Download task report PDF", exact: true});
  await expect(button).toHaveCount(1);
  await expect(page.getByRole("button", {name: "How this answer was found", exact: true})).toHaveCount(0);
  const waiting = page.waitForEvent("download");
  await button.click();
  const file = await waiting;
  expect(file.suggestedFilename()).toBe(`planproof-task-${lastId.slice(0,8)}.pdf`);
  expect(body).toEqual({document_id: "drawing", question: "Where are the windows?", filename: "test.png"});
  expect(submissions).toBe(2);
  expect(downloads).toBe(1);
  await expect(page.getByLabel("Question about this drawing")).toHaveValue("Unsent next question");
});

test("download failure preserves the answer, blocks duplicate requests and allows explicit retry", async ({page}) => {
  await setup(page);
  let fail = true, requests = 0;
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route("**/api/v1/documents/*/runs", r => r.fulfill({json: {request_id: r.request().postDataJSON().request_id, status: "completed", source, events, last_sequence: 4, result}}));
  await page.route("**/api/v1/runs/*/report", async r => {
    requests++;
    if (fail) { await gate; return r.fulfill({status: 503, json: {detail: "Report service unavailable. Try again shortly."}}); }
    return r.fulfill({contentType: "application/pdf", body: "%PDF-1.7\n% Fixture"});
  });
  await ask(page);
  await page.getByRole("button", {name: "Download task report PDF", exact: true}).click();
  await expect(page.getByRole("button", {name: "Preparing report… PDF", exact: true})).toBeDisabled();
  expect(requests).toBe(1);
  release();
  await expect(page.getByRole("alert")).toContainText("Report service unavailable");
  await expect(page.getByTestId("answer-value")).toHaveText("2 doors");
  fail = false;
  const waiting = page.waitForEvent("download");
  await page.getByRole("button", {name: "Download task report PDF", exact: true}).click();
  await waiting;
  expect(requests).toBe(2);
  await expect(page.locator(".report-error")).toHaveCount(0);
});

test("a running latest task hides the previous report; a failed task exports its own record", async ({page}) => {
  await setup(page);
  let submissions = 0, finished = false, id = "";
  await page.route("**/api/v1/documents/*/runs", r => {
    submissions++;
    id = r.request().postDataJSON().request_id;
    return r.fulfill({json: {request_id: id, status: submissions === 1 ? "completed" : "running", source, events, last_sequence: 4, result: submissions === 1 ? result : null}});
  });
  await page.route("**/api/v1/runs/*?*", r => r.fulfill({json: {request_id: id, status: finished ? "failed" : "running", source, events, last_sequence: 4, message: finished ? "The service was unavailable." : undefined}}));
  await ask(page);
  await expect(page.getByRole("button", {name: "Download task report PDF", exact: true})).toBeVisible();
  await ask(page);
  await expect(page.getByRole("button", {name: "Download task report PDF", exact: true})).toHaveCount(0);
  await expect(page.getByTestId("live-work-log")).toBeVisible();
  finished = true;
  await expect(page.getByText("The service was unavailable.", {exact: true})).toBeVisible();
  await expect(page.getByRole("button", {name: "Download task report PDF", exact: true})).toHaveCount(1);
});

test("inspector links replay, layers, comparisons and Follow without extra submissions", async ({
  page,
}) => {
  await setup(page);
  let submissions = 0;
  await page.route("**/api/v1/documents/*/runs", (r) => {
    submissions++;
    return r.fulfill({
      json: {
        request_id: r.request().postDataJSON().request_id,
        status: "completed",
        source,
        events,
        last_sequence: 4,
        result,
      },
    });
  });
  await page.route("**/api/v1/runs/*/evidence", (r) =>
    r.fulfill({ json: bundle }),
  );
  await ask(page);
  await page
    .getByRole("button", { name: "Task details", exact: true })
    .click();
  await expect(page.getByLabel("Follow live regions")).not.toBeChecked();
  await page
    .getByRole("button", { name: "Inspect step 2 region 1", exact: true })
    .click();
  await expect(page.getByTestId("activity-region")).toHaveAttribute("x", "100");
  await expect(
    page.getByText("Remaining page unknown.", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("Follow live regions").check();
  await page.getByTestId("plan-viewport").dispatchEvent("wheel");
  await expect(page.getByLabel("Follow live regions")).not.toBeChecked();
  await page.getByRole("button", { name: "Diagnostics", exact: true }).click();
  await page.getByRole("button", { name: "Show raw algorithm output", exact: true }).click();
  await page.getByLabel("ML detections", { exact: true }).check();
  await page.getByLabel("Deterministic evidence", { exact: true }).check();
  await expect(page.getByTestId("algorithm-overlay")).toHaveAttribute(
    "data-shapes",
    "2",
  );
  await page.locator(".object-list button").first().click();
  await expect(page.locator(".object-detail")).toContainText("IoU 0.80");
  await expect(page.locator(".object-detail")).toContainText(
    "Supplied to the agent",
  );
  await page.locator(".object-list button").nth(1).click();
  await expect(page.locator(".object-detail")).toContainText("quarter-circle arc");
  await expect(page.locator(".object-detail")).toContainText("Swing arc support");
  await expect(page.locator(".object-detail")).toContainText("0.750");
  await expect(page.locator(".object-detail")).toContainText(
    "not recorded as supplied to the agent",
  );
  await page.getByLabel("ML-assisted boundaries", { exact: true }).check();
  await page.getByLabel("Evidence class").selectOption("room");
  await expect(page.getByTestId("algorithm-overlay")).toHaveAttribute(
    "data-shapes",
    "1",
  );
  await page
    .getByRole("button", { name: "Compare ML / rules", exact: true })
    .click();
  await expect(page.getByTestId("algorithm-overlay")).toHaveAttribute(
    "data-split",
    "50",
  );
  await page.getByLabel("Comparison split", { exact: true }).fill("70");
  await expect(page.getByTestId("algorithm-overlay")).toHaveAttribute(
    "data-split",
    "70",
  );
  await page
    .getByRole("button", { name: "Compare ML / rules", exact: true })
    .click();
  await expect(page.getByTestId("algorithm-overlay")).toHaveAttribute(
    "data-split",
    "off",
  );
  expect(submissions).toBe(1);
  await page
    .getByRole("button", { name: "Back to conversation", exact: true })
    .click();
  await expect(page.getByTestId("algorithm-overlay")).toHaveAttribute(
    "data-shapes",
    "0",
  );
  await expect(page.getByTestId("answer-value")).toHaveText("2 doors");
  await page.reload();
  await page
    .getByRole("button", { name: "Task details", exact: true })
    .click();
  await page.route("**/api/v1/runs/*/evidence", (r) => r.abort());
  await page.getByRole("button", { name: "Diagnostics", exact: true }).click();
  await page.getByRole("button", { name: "Show raw algorithm output", exact: true }).click();
  await expect(page.getByLabel("ML detections", { exact: true })).toBeVisible();
  expect(submissions).toBe(1);
});

test("live polling reconnects after refresh to the same run and keeps sequence unique", async ({
  page,
}) => {
  await setup(page);
  let submissions = 0,
    polls = 0,
    runId = "",
    completed = false,
    disconnected = false;
  await page.route("**/api/v1/documents/*/runs", (r) => {
    submissions++;
    runId = r.request().postDataJSON().request_id;
    return r.fulfill({
      json: {
        request_id: runId,
        status: "running",
        source,
        events: events.slice(0, 1),
        last_sequence: 1,
      },
    });
  });
  await page.route("**/api/v1/runs/*?*", (r) => {
    polls++;
    if (disconnected) return r.abort();
    return r.fulfill({
      json: {
        request_id: runId,
        status: completed ? "completed" : "running",
        source,
        events: events.slice(0, completed ? 4 : 2),
        last_sequence: completed ? 4 : 2,
        result: completed ? result : null,
      },
    });
  });
  await ask(page);
  await page
    .getByRole("button", { name: "Task details", exact: true })
    .click();
  await expect(page.locator(".event-list > li")).toHaveCount(2);
  await page.route("**/api/v1/runs/*/evidence", (r) =>
    r.fulfill({ json: bundle }),
  );
  await page.getByRole("button", { name: "Diagnostics", exact: true }).click();
  await page.getByRole("button", { name: "Show raw algorithm output", exact: true }).click();
  await page.getByLabel("ML detections", { exact: true }).check();
  const previousPolls = polls;
  await expect.poll(() => polls).toBeGreaterThan(previousPolls);
  await expect(page.getByTestId("algorithm-overlay")).toHaveAttribute(
    "data-shapes", "1",
  );
  await page.getByRole("button", { name: "Steps", exact: true }).click();
  disconnected = true;
  await expect(
    page.getByLabel("Evidence inspector").getByText("Reconnecting to activity", { exact: true }),
  ).toBeVisible({ timeout: 6000 });
  disconnected = false;
  await page.reload();
  await page
    .getByRole("button", { name: "Task details", exact: true })
    .click();
  await expect(page.locator(".event-list > li")).toHaveCount(2);
  completed = true;
  await expect(page.locator(".event-list > li")).toHaveCount(4);
  await expect(page.getByText(/Recorded activity/)).toBeVisible();
  expect(submissions).toBe(1);
  await page
    .getByRole("button", { name: "Back to conversation", exact: true })
    .click();
  await expect(page.getByTestId("answer-value")).toHaveText("2 doors");
});

test("direct answer and wrong-source evidence never pretend algorithms ran", async ({
  page,
}) => {
  await setup(page);
  await page.route("**/api/v1/documents/*/runs", (r) =>
    r.fulfill({
      json: {
        request_id: r.request().postDataJSON().request_id,
        status: "completed",
        source,
        events: [],
        last_sequence: 0,
        result,
      },
    }),
  );
  await page.route("**/api/v1/runs/*/evidence", (r) =>
    r.fulfill({
      json: { ...bundle, status: "not_run", shapes: [], comparisons: [] },
    }),
  );
  await ask(page);
  await page
    .getByRole("button", { name: "Task details", exact: true })
    .click();
  await page.getByRole("button", { name: "Diagnostics", exact: true }).click();
  await page.getByRole("button", { name: "Show raw algorithm output", exact: true }).click();
  await expect(
    page.getByText("Algorithms not used for this answer."),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Back to conversation", exact: true })
    .click();
  await page.route("**/api/v1/documents/*/frame", (r) =>
    r.fulfill({ json: { ...source, raster_sha256: "another-raster" } }),
  );
  await page.reload();
  await page
    .getByRole("button", { name: "Task details", exact: true })
    .click();
  await expect(
    page.getByText(/Source identity could not be matched/),
  ).toBeVisible();
  await expect(page.getByTestId("activity-region")).toHaveCount(0);
});

test("mobile inspector navigates to crops and back with keyboard and no overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await setup(page);
  await page.route("**/api/v1/documents/*/runs", (r) =>
    r.fulfill({
      json: {
        request_id: r.request().postDataJSON().request_id,
        status: "completed",
        source,
        events,
        last_sequence: 4,
        result,
      },
    }),
  );
  await ask(page);
  const inspect = page.getByRole("button", {
    name: "Task details",
    exact: true,
  });
  await page.getByRole("button", { name: "Show drawing", exact: true }).click();
  await inspect.focus();
  await inspect.press("Enter");
  await page
    .getByRole("button", { name: "Inspect step 2 region 1", exact: true })
    .click();
  await expect(page.getByTestId("activity-region")).toBeVisible();
  await page
    .getByRole("button", { name: "Show conversation", exact: true })
    .click();
  await expect(
    page.getByRole("region", { name: "Evidence inspector" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});


test("actual work and requested regions appear on the plan without opening diagnostics", async ({page}) => {
  await setup(page);
  let done = false, evidenceReads = 0;
  const activeEvents = [events[1], {...events[2], state: "running", advisory: undefined, question: "Are these symbols bathtubs or sinks?"}];
  let id = "";
  await page.route("**/api/v1/documents/*/runs", r => {
    id = r.request().postDataJSON().request_id;
    return r.fulfill({json: {request_id: id, status: "running", source, events: activeEvents, last_sequence: 3}});
  });
  await page.route("**/api/v1/runs/*?*", r => r.fulfill({json: {request_id: id, status: done ? "completed" : "running", source, events: done ? events : activeEvents, last_sequence: done ? 4 : 3, result: done ? result : null}}));
  await page.route("**/api/v1/runs/*/evidence", r => { evidenceReads++; return r.fulfill({json: bundle}); });
  await ask(page);
  await expect(page.getByTestId("live-work-log").getByRole("status")).toContainText("Calling Nemotron");
  await expect(page.getByText("Are these symbols bathtubs or sinks?", {exact: true})).not.toBeVisible();
  await expect(page.locator(".work-log > li")).toHaveCount(0);
  await page.screenshot({path: "../.artifacts/compact-activity/compact-desktop.png", fullPage: true});
  await page.locator(".activity-details > summary").click();
  await expect(page.getByTestId("plan-live-activity")).toHaveCount(0);
  await expect(page.getByTestId("live-work-log")).toContainText("Are these symbols bathtubs or sinks?");
  await expect(page.getByTestId("live-work-log")).toContainText("Checking this detail with Nemotron");
  await expect(page.getByTestId("live-inspection-region")).toHaveAttribute("x", "100");
  await expect(page.getByTestId("live-inspection-region")).toHaveAttribute("width", "300");
  await expect(page.getByLabel("Evidence inspector")).toHaveCount(0);
  expect(evidenceReads).toBe(0);
  await expect(page.getByTestId("algorithm-overlay")).toHaveAttribute("data-shapes", "0");
  const follow = page.getByLabel("Drawing controls").getByRole("button", {name: "Follow view", exact: true});
  await expect(follow).toHaveAttribute("aria-pressed", "false");
  await follow.focus();
  await follow.press("Enter");
  await expect(follow).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("zoom-level")).not.toHaveText("100%");
  await follow.click();
  await expect(follow).toHaveAttribute("aria-pressed", "false");
  await follow.click();
  await page.getByTestId("plan-viewport").dispatchEvent("wheel");
  await expect(page.getByRole("button", {name: "Follow view", exact: true})).toHaveAttribute("aria-pressed", "false");
  done = true;
  await expect(page.getByTestId("answer-value")).toHaveText("2 doors");
  await expect(page.getByTestId("plan-live-activity")).toHaveCount(0);
  await expect(page.getByTestId("live-inspection-region")).toHaveCount(0);
  await expect(page.locator(".work-recap")).toHaveCount(0);
  await expect(page.getByRole("button", {name: "Download task report PDF", exact: true})).toBeVisible();
  await page.getByRole("button", {name: "Task details", exact: true}).click();
  await expect(page.locator(".event-list")).toContainText("Remaining page unknown.");
});

test("pending algorithms remain honest while source inspection proceeds", async ({page}) => {
  await setup(page);
  const pending = [{...events[0], state: "pending", summary: "Algorithm evidence is not ready."}, {...events[1], state: "running"}];
  await page.route("**/api/v1/runs/*?*", r => r.fulfill({json: {request_id: "pending-fixture", status: "running", source, events: pending, last_sequence: 2}}));
  await page.route("**/api/v1/documents/*/runs", r => r.fulfill({json: {request_id: r.request().postDataJSON().request_id, status: "running", source, events: pending, last_sequence: 2}}));
  await ask(page);
  await page.locator(".activity-details > summary").click();
  const log = page.getByTestId("live-work-log");
  await expect(log).toContainText("Structural analysis continues in the background");
  await expect(log).toContainText("No algorithm evidence was supplied by this request");
  await expect(log).toContainText("Opening a closer view");
  await expect(page.getByTestId("algorithm-overlay")).toHaveAttribute("data-shapes", "0");
});

for (const timing of [
  { name: "null", seconds: null, suffix: "" },
  { name: "omitted", seconds: undefined, suffix: "" },
  { name: "zero", seconds: 0, suffix: " · 0.0s" },
  { name: "measured", seconds: 2.5, suffix: " · 2.5s" },
  { name: "non-numeric", seconds: "pending", suffix: "" },
  { name: "negative", seconds: -1, suffix: "" },
]) {
  test(`inspection duration ${timing.name} keeps the live job and task history usable`, async ({page}) => {
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    await setup(page);
    let done = false, id = "", submissions = 0;
    const started = {...events[2], sequence: 1, state: "running", inspection_id: "worker-1",
      seconds: timing.seconds, advisory: undefined, citations: [], question: "Inspect this entrance."};
    const returned = {...started, sequence: 2, state: "completed", seconds: 1.25,
      advisory: {findings: ["The source check returned."]}};
    const snapshot = () => ({request_id: id, status: done ? "completed" : "running", source,
      events: done ? [started, returned] : [started], last_sequence: done ? 2 : 1,
      result: done ? result : null});
    await page.route("**/api/v1/documents/*/runs", r => {
      submissions++;
      id = r.request().postDataJSON().request_id;
      return r.fulfill({json: snapshot()});
    });
    await page.route("**/api/v1/runs/*?*", r => r.fulfill({json: snapshot()}));
    await ask(page);
    await page.locator(".activity-details > summary").click();
    await expect(page.getByTestId("live-work-log").locator(".work-content > small"))
      .toHaveText(`In progress${timing.suffix}`);
    await expect(page.getByTestId("source-image")).toBeVisible();
    await page.getByRole("button", {name: "Task details", exact: true}).click();
    await expect(page.locator(".event-title small")).toHaveText(`running${timing.suffix}`);
    done = true;
    // The raw start event remains in history after its worker finishes.
    await expect(page.locator(".event-title small")).toHaveText([
      `running${timing.suffix}`, "completed · 1.3s",
    ]);
    await page.getByRole("button", {name: "Back to conversation", exact: true}).click();
    await expect(page.getByTestId("answer-value")).toHaveText("2 doors");
    await expect(page.getByLabel("Question about this drawing")).toBeEnabled();
    expect(errors).toEqual([]);
    expect(submissions).toBe(1);
  });
}

test("parallel inspections keep each region and completion associated with its worker", async ({page}) => {
  await setup(page);
  const second = {...region, image_id: "crop-2", canonical_bbox: [500, 100, 800, 350]};
  const workers = [
    {...events[2], sequence: 1, state: "running", inspection_id: "worker-a", question: "Check the left apartment entrance", images: [region], citations: [], advisory: undefined},
    {...events[2], sequence: 2, state: "running", inspection_id: "worker-b", question: "Check the right apartment entrance", images: [second], citations: [], advisory: undefined},
    {...events[2], sequence: 3, state: "completed", inspection_id: "worker-a", question: "Check the left apartment entrance", images: [region], citations: [region], advisory: {findings: ["Left entry observed."]}},
  ];
  await page.route("**/api/v1/runs/*?*", r => r.fulfill({json: {request_id: "parallel-fixture", status: "running", source, events: workers, last_sequence: 3}}));
  await page.route("**/api/v1/documents/*/runs", r => r.fulfill({json: {request_id: r.request().postDataJSON().request_id, status: "running", source, events: workers, last_sequence: 3}}));
  await ask(page);
  await page.locator(".activity-details > summary").click();
  const log = page.getByTestId("live-work-log");
  await expect(log.locator("li")).toHaveCount(2);
  const right = log.locator("li").filter({hasText: "Check the right apartment entrance"});
  await expect(right).toHaveClass(/work-current/);
  await expect(right).toContainText("In progress");
  const left = log.locator("li").filter({hasText: "Check the left apartment entrance"});
  await expect(left).toContainText("Left entry observed");
  await expect(left).not.toHaveClass(/work-current/);
  await expect(page.getByTestId("live-inspection-region")).toHaveAttribute("x", "500");
});

test("tool cards retain active inspections, original timing and uncertainty through completion", async ({page}) => {
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(e.message));
  await setup(page);
  let phase = 0, id = "", submissions = 0;
  const requested = {...region, sent_size: [1536, 1024]};
  const second = {...region, canonical_bbox: [500, 100, 800, 350]};
  const first = {...events[2], sequence: 1, elapsed: 3, state: "running", seconds: null,
    inspection_id: "first", advisory: undefined, citations: [], images: [requested], question: "Check the left entrance."};
  const other = {...first, sequence: 2, elapsed: 4, inspection_id: "other", images: [second], question: "Check the right entrance."};
  const later = Array.from({length: 6}, (_, i) => ({...events[1], sequence: i + 3, elapsed: i + 5}));
  const returned = {...first, sequence: 9, elapsed: 12, state: "completed", seconds: 9,
    advisory: {verdict: "uncertain", findings: ["The entrance symbol is ambiguous.", "A nearby mark may be furniture.", "The lower corner is obscured."]}};
  const failed = {...other, sequence: 10, elapsed: 14, state: "unavailable", seconds: 10};
  const snapshot = () => ({request_id: id, source, status: phase === 2 ? "completed" : "running",
    events: [first, other, ...later, ...(phase > 0 ? [returned] : []), ...(phase === 2 ? [failed] : [])],
    last_sequence: phase === 2 ? 10 : phase === 1 ? 9 : 8, result: phase === 2 ? result : null});
  await page.route("**/api/v1/documents/*/runs", r => {submissions++; id = r.request().postDataJSON().request_id; return r.fulfill({json: snapshot()});});
  await page.route("**/api/v1/runs/*?*", r => r.fulfill({json: snapshot()}));
  await ask(page);
  await expect(page.locator(".work-log > li")).toHaveCount(0);
  const colors = await page.getByTestId("live-inspection-region").evaluateAll(nodes => nodes.map(n => getComputedStyle(n).stroke));
  expect(new Set(colors).size).toBe(2);
  expect(colors).not.toContain("rgb(66, 99, 74)"); // Final evidence green.
  await page.locator(".activity-details > summary").click();
  const log = page.getByTestId("live-work-log");
  await expect(log.locator('[data-tool="nemotron"]')).toHaveCount(2);
  await expect(log.getByRole("status")).toContainText("Calling Nemotron");
  const left = log.locator(".work-log > li").filter({hasText: "Check the left entrance."});
  await expect(left).toContainText("Nemotron · Detail 1");
  await expect(left).toContainText("Started +0:03");
  await expect(left).toContainText("Sent 1536 × 1024 px");
  await expect(left).not.toContainText("0.0s");
  await expect(page.locator(".live-region-label")).toHaveText(["Nemotron 1", "Nemotron 2"]);
  await left.getByRole("button", {name: "Inspect work step 1 source 1"}).click();
  await expect(page.getByTestId("activity-region")).toHaveAttribute("x", "100");
  await page.getByRole("button", {name: "Follow view", exact: true}).click();
  phase = 1;
  await expect(left).toContainText("Returned +0:12");
  await expect(left).toContainText("Started +0:03");
  await expect(left).toContainText("uncertainty remains");
  await expect(page.locator(".live-region-label")).toHaveText("Nemotron 2");
  await expect(page.getByTestId("activity-region")).toHaveAttribute("x", "500");
  await left.getByText("Show 1 more findings", {exact: true}).click();
  await expect(left.getByText("The lower corner is obscured.", {exact: true})).toBeVisible();
  await expect(log.locator(".work-log > li")).toHaveCount(8);
  phase = 2;
  await expect(page.getByTestId("answer-value")).toHaveText("2 doors");
  await expect(page.locator(".work-recap")).toHaveCount(0);
  await page.getByRole("button", {name: "Task details", exact: true}).click();
  await expect(page.locator(".event-list")).toContainText("unavailable");
  await expect(page.locator(".event-list")).toContainText("The entrance symbol is ambiguous.");
  await page.reload();
  await expect(page.getByTestId("answer-value")).toHaveText("2 doors");
  await expect(page.locator(".work-recap")).toHaveCount(0);
  expect(submissions).toBe(1);
  expect(errors).toEqual([]);
});

test("algorithm card exposes recorded ML and rules only on explicit request", async ({page}) => {
  await setup(page);
  let reads = 0, submissions = 0;
  const algorithm = {...events[0], cached: true, delivered_ids: ["m1", "d1", "m1"]};
  await page.route("**/api/v1/documents/*/runs", r => {submissions++; return r.fulfill({json: {
    request_id: r.request().postDataJSON().request_id, source, status: "running", events: [algorithm], last_sequence: 1,
  }});});
  await page.route("**/api/v1/runs/*?*", r => r.fulfill({json: {source, status: "running", events: [algorithm], last_sequence: 1}}));
  await page.route("**/api/v1/runs/*/evidence", r => {reads++; return r.fulfill({json: bundle});});
  await ask(page);
  await page.locator(".activity-details > summary").click();
  const card = page.locator('.work-log > li[data-tool="algorithms"]');
  await expect(card).toContainText("ML detection");
  await expect(card).toContainText("Deterministic rules");
  await expect(card).toContainText("2 evidence items supplied by this request");
  await expect(card.locator(".work-status")).toHaveText("Reused");
  expect(reads).toBe(0);
  await card.getByRole("button", {name: "Explore raw ML + rule candidates"}).click();
  await expect(page.getByLabel("ML detections", {exact: true})).toBeVisible();
  await expect(page.getByTestId("algorithm-overlay")).toHaveAttribute("data-shapes", "0");
  await page.getByLabel("ML detections", {exact: true}).check();
  await expect(page.getByTestId("algorithm-overlay")).toHaveAttribute("data-shapes", "1");
  expect(reads).toBe(1);
  expect(submissions).toBe(1);
});

test("four active detail views and tool cards fit a narrow screen", async ({page}) => {
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(e.message));
  await page.setViewportSize({width: 320, height: 844});
  await page.emulateMedia({reducedMotion: "reduce"});
  await setup(page);
  const workers = Array.from({length: 4}, (_, i) => ({...events[2], state: "running", sequence: i + 1,
    elapsed: i, seconds: null, inspection_id: `worker-${i}`, advisory: undefined, citations: [],
    images: [{...region, canonical_bbox: [50 + i * 220, 100, 200 + i * 220, 350]}], question: `Inspect entrance ${i + 1}.`}));
  await page.route("**/api/v1/documents/*/runs", r => r.fulfill({json: {request_id: r.request().postDataJSON().request_id, source, status: "running", events: workers, last_sequence: 4}}));
  await page.route("**/api/v1/runs/*?*", r => r.fulfill({json: {request_id: "fixture", source, status: "running", events: workers, last_sequence: 4}}));
  await ask(page);
  await page.locator(".activity-details > summary").click();
  await expect(page.getByTestId("live-work-log").locator(".work-log > li")).toHaveCount(4);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole("button", {name: "Inspect work step 1 source 1"}).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("live-inspection-region")).toHaveCount(4);
  await expect(page.locator(".live-region-label")).toHaveCount(4);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole("button", {name: "Show conversation", exact: true}).click();
  await expect(page.getByTestId("live-work-log")).toBeVisible();
  expect(errors).toEqual([]);
});

test("briefing attaches citations to claims and only detection previews follow the text", async ({page}) => {
  await setup(page);
  const rich = {...result, answer: {...result.answer, answer: 12, unit: "m2", briefing: [
    {section: "answer", text: "The gym is approximately 12 square metres.", citations: [1]},
    {section: "basis", text: "The boundary follows the inside wall faces.", citations: [2]}
  ], citation_details: [
    {citation: 1, kind: "measurement", label: "Printed room area"},
    {citation: 2, kind: "detection", label: "Gym boundary"}
  ]}, measurement: {approximate: true}, canonical_citations: [region, {...region, canonical_bbox: [400, 100, 600, 300]}]};
  await page.route("**/api/v1/documents/*/runs", r => r.fulfill({json: {request_id: r.request().postDataJSON().request_id, status: "completed", source, events, last_sequence: 4, result: rich}}));
  await ask(page);
  await expect(page.getByTestId("answer-value")).toHaveText("≈ 12 m²");
  const paragraphs = page.locator(".briefing-prose p");
  await expect(paragraphs.nth(0).getByRole("button")).toHaveText("1");
  await expect(paragraphs.nth(1).getByRole("button")).toHaveText("2");
  await expect(page.locator(".detection-card")).toHaveCount(1);
  await expect(page.locator(".detection-card")).toContainText("Gym boundary");
  await expect(page.getByTestId("evidence-box")).toHaveCount(1);
  await expect(page.getByTestId("evidence-box")).toHaveAttribute("x", "400");
  await page.getByRole("button", {name: "View evidence 1, page 1", exact: true}).click();
  await expect(page.getByTestId("evidence-box").first()).toHaveAttribute("x", "100");
  await expect(page.locator(".detection-card")).toBeVisible();
  await expect(page.getByText("Some symbols remain ambiguous.", {exact: true})).not.toBeVisible();
  await page.getByText("More information", {exact: true}).click();
  await expect(page.getByText("Some symbols remain ambiguous.", {exact: true})).toBeVisible();
});

test("clean answer keeps every preview at the end and combines notes into one disclosure", async ({page}) => {
  await setup(page);
  const rich = {...result, answer: {...result.answer, answer: 4, unit: "staircases", briefing: [
    {section: "answer", text: "Four staircases connect the wings and central common area.", citations: [1, 2, 3, 4]},
    {section: "basis", text: "Three are outside the wings; one is in the common area.", citations: []},
    {section: "limitation", text: "The remaining details were not inspected.", citations: []},
  ], citation_details: ["North wing", "West wing", "East wing", "Common area"].map((label, i) => ({citation: i + 1, kind: "detection", label}))},
  canonical_citations: Array.from({length: 4}, (_, i) => ({...region, canonical_bbox: [50 + i * 220, 100, 200 + i * 220, 350]}))};
  await page.route("**/api/v1/documents/*/runs", r => r.fulfill({json: {request_id: r.request().postDataJSON().request_id, status: "completed", source, events, last_sequence: 4, result: rich}}));
  await ask(page);
  const answer = page.getByTestId("agent-answer");
  await expect(answer.locator("details")).toHaveCount(1);
  await expect(answer.getByText("More information", {exact: true})).toBeVisible();
  await expect(answer.getByText(/Verification notes|Scope of this answer/)).toHaveCount(0);
  await expect(page.locator(".work-recap")).toHaveCount(0);
  await expect(answer.locator(":scope > :last-child")).toHaveClass("detection-evidence");
  await expect(answer.locator(".detection-card")).toHaveCount(4);
  for (const card of await answer.locator(".detection-card").all()) await expect(card).toBeVisible();
  await answer.getByText("More information", {exact: true}).click();
  await expect(answer.getByText("The remaining details were not inspected.", {exact: true})).toBeVisible();
  await expect(answer.getByText("Some symbols remain ambiguous.", {exact: true})).toBeVisible();
  await expect(answer.getByText("Crop only; remaining page not established.", {exact: true})).toBeVisible();
  await answer.getByText("More information", {exact: true}).click();
  await page.screenshot({path: "../.artifacts/answer-polish/desktop.png", fullPage: true});
  await page.setViewportSize({width: 390, height: 844});
  await page.getByRole("button", {name: "Show conversation", exact: true}).click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({path: "../.artifacts/answer-polish/mobile.png", fullPage: true});
});

test("raw diagnostics require explicit opt-in and never load automatically", async ({page}) => {
  await setup(page);
  let reads = 0;
  await page.route("**/api/v1/documents/*/runs", r => r.fulfill({json:{request_id:r.request().postDataJSON().request_id,status:"completed",source,events,last_sequence:4,result}}));
  await page.route("**/api/v1/runs/*/evidence", r => { reads++; return r.fulfill({json:bundle}); });
  await ask(page);
  await page.getByRole("button", {name:"Task details",exact:true}).click();
  await page.getByRole("button", {name:"Diagnostics",exact:true}).click();
  await expect(page.getByText("Raw algorithm proposals",{exact:true})).toBeVisible();
  await expect(page.getByLabel("ML detections",{exact:true})).toHaveCount(0);
  expect(reads).toBe(0);
  await page.getByRole("button", {name:"Show raw algorithm output",exact:true}).click();
  await expect(page.getByLabel("ML detections",{exact:true})).toBeVisible();
  expect(reads).toBe(1);
});
