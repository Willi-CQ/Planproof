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
  await expect(page.getByTestId("plan-live-activity")).toContainText("Are these symbols bathtubs or sinks?");
  await expect(page.getByTestId("live-work-log")).toContainText("Checking this detail with Nemotron");
  await expect(page.getByTestId("live-inspection-region")).toHaveAttribute("x", "100");
  await expect(page.getByTestId("live-inspection-region")).toHaveAttribute("width", "300");
  await expect(page.getByLabel("Evidence inspector")).toHaveCount(0);
  expect(evidenceReads).toBe(0);
  await expect(page.getByTestId("algorithm-overlay")).toHaveAttribute("data-shapes", "0");
  await page.getByRole("button", {name: "Follow view", exact: true}).click();
  await expect(page.getByRole("button", {name: "Following", exact: true})).toHaveAttribute("aria-pressed", "true");
  await page.getByTestId("plan-viewport").dispatchEvent("wheel");
  await expect(page.getByRole("button", {name: "Follow view", exact: true})).toHaveAttribute("aria-pressed", "false");
  done = true;
  await expect(page.getByTestId("answer-value")).toHaveText("2 doors");
  await expect(page.getByTestId("plan-live-activity")).toHaveCount(0);
  await expect(page.getByTestId("live-inspection-region")).toHaveCount(0);
  await expect(page.locator(".work-recap")).toHaveCount(0);
  await page.getByRole("button", {name: "Task details", exact: true}).click();
  await expect(page.locator(".event-list")).toContainText("Remaining page unknown.");
});

test("pending algorithms remain honest while source inspection proceeds", async ({page}) => {
  await setup(page);
  const pending = [{...events[0], state: "pending", summary: "Algorithm evidence is not ready."}, {...events[1], state: "running"}];
  await page.route("**/api/v1/runs/*?*", r => r.fulfill({json: {request_id: "pending-fixture", status: "running", source, events: pending, last_sequence: 2}}));
  await page.route("**/api/v1/documents/*/runs", r => r.fulfill({json: {request_id: r.request().postDataJSON().request_id, status: "running", source, events: pending, last_sequence: 2}}));
  await ask(page);
  const log = page.getByTestId("live-work-log");
  await expect(log).toContainText("Structural analysis continues in the background");
  await expect(log).toContainText("No algorithm evidence has been supplied yet");
  await expect(log).toContainText("Opening a closer view");
  await expect(page.getByTestId("algorithm-overlay")).toHaveAttribute("data-shapes", "0");
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
  await expect(page.locator(".briefing-limits")).toContainText("Some symbols remain ambiguous.");
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
