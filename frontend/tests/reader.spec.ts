// Browser transport and evidence checks with synthetic files and a fixture API.
import { expect, test, type Page, type Route } from "@playwright/test";
import { createHash, randomBytes } from "node:crypto";
import { crc32, deflateSync } from "node:zlib";
import { PDFDocument } from "pdf-lib";

const hash = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
async function submitted(route: Route) {
  const request = route.request();
  const body = await new Response(new Uint8Array(request.postDataBuffer()!), {
    headers: {"content-type": request.headers()["content-type"]},
  }).formData();
  const file = body.get("file") as File;
  const bytes = Buffer.from(await file.arrayBuffer());
  return {file, bytes, question: body.get("question"), page: Number(body.get("page"))};
}
async function ask(page: Page, question = "Where is door D-01?") {
  await page.getByLabel("Question about this drawing").fill(question);
  await page.getByRole("button", {name: "Ask Planproof", exact: true}).click();
}
async function imageFile(page: Page, type: string) {
  const encoded = await page.evaluate(mime => {
    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 180;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, 320, 180);
    ctx.fillStyle = "black";
    ctx.fillText("D-01", 82, 64);
    return canvas.toDataURL(mime).split(",")[1];
  }, type);
  return Buffer.from(encoded, "base64");
}
function answer(bytes: Buffer, page: number, type: string, width: number, height: number) {
  return {source_sha256: hash(bytes), page, width, height,
    coordinates: type === "application/pdf" ? "displayed_pdf_points" : "image_pixels",
    status: "answered", value: "D-01", unit: null, reason: "",
    calculation: "Locate the printed door identifier.", evidence: [
      {id: "mark1", kind: "door_mark", text: "D-01", bbox: [80,50,100,70]},
    ]};
}
function largeBilevelPng(width: number, height: number) {
  const chunk = (name: string, data: Buffer) => {
    const type = Buffer.from(name, "ascii"), size = Buffer.alloc(4), checksum = Buffer.alloc(4);
    size.writeUInt32BE(data.length);
    checksum.writeUInt32BE(crc32(data, crc32(type)));
    return Buffer.concat([size, type, data, checksum]);
  };
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width);
  header.writeUInt32BE(height, 4);
  header[8] = 1;
  const stride = 1 + Math.ceil(width / 8);
  const rows = Buffer.alloc(stride * height, 255);
  for (let y = 0; y < height; y++) {
    rows[y * stride] = 0;
    if (y >= 50 && y < 70) rows.fill(0, y * stride + 11, y * stride + 14);
  }
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]), chunk("IHDR", header),
    chunk("IDAT", deflateSync(rows)), chunk("IEND", Buffer.alloc(0))]);
}
test("a 67 MP original keeps native bytes and citation coordinates independently of its preview", async ({page}) => {
  const source = largeBilevelPng(10200, 6600);
  await page.route("**/api/reader", async route => {
    const sent = await submitted(route);
    expect(sent.bytes.equals(source)).toBeTruthy();
    await route.fulfill({json: answer(sent.bytes, 1, sent.file.type, 10200, 6600)});
  });
  await page.goto("/");
  await page.getByLabel("Upload drawing", {exact:true}).setInputFiles({name:"bilevel.png", mimeType:"image/png", buffer:source});
  await expect(page.getByText(/original 10200 × 6600 image/)).toBeVisible();
  await expect(page.getByRole("checkbox", {name:/Use the displayed/})).toHaveCount(0);
  const preview = page.getByTestId("source-image");
  await expect(preview).toHaveAttribute("width", "10200");
  expect(await preview.evaluate(async node => {
    const image = node as HTMLImageElement;
    await image.decode();
    return image.naturalWidth;
  })).toBe(3600);
  await ask(page);
  await page.getByRole("button", {name:"View evidence 1, page 1"}).click();
  await expect(page.getByTestId("evidence-box")).toHaveAttribute("x", "80");
  const detail = page.getByRole("img", {name:"Selected evidence from original pixels"});
  await expect.poll(() => detail.evaluate(node => {
    const el = node as HTMLCanvasElement;
    const ctx = el.getContext("2d")!;
    const center = ctx.getImageData(Math.floor(el.width / 2), Math.floor(el.height / 2), 1, 1).data;
    return Array.from(center);
  })).toEqual([0,0,0,255]);
});

test("unsupported questions, text not located and incomplete counts are different results", async ({page}) => {
  await page.goto("/");
  const bytes = await imageFile(page, "image/png");
  let attempt = 0;
  await page.route("**/api/reader", async route => {
    const data = answer(bytes, 1, "image/png", 320, 180);
    const results = [
      {status:"unresolved", value:null, evidence:[], reason_code:"unsupported_question", reason:"The operation is unsupported."},
      {status:"unresolved", value:null, evidence:[], reason_code:"text_not_found", reason:"No matching text was found."},
      {status:"partial", value:2, unit:"count", reason_code:"incomplete_coverage", reason:"Only some labels could be read."},
    ];
    await route.fulfill({json:{...data, ...results[attempt++]}});
  });
  await page.getByLabel("Upload drawing", {exact:true}).setInputFiles({name:"labels.png", mimeType:"image/png", buffer:bytes});
  await page.getByRole("button", {name:'Wo steht „North Hall“?'}).click();
  await expect(page.getByLabel("Question about this drawing")).toHaveValue('Wo steht „North Hall“?');
  await ask(page);
  await expect(page.getByTestId("answer-value").last()).toHaveText("This question needs a different reading method");
  await ask(page);
  await expect(page.getByTestId("answer-value").last()).toHaveText("No matching label found in the text read");
  await ask(page);
  await expect(page.getByTestId("answer-value").last()).toHaveText("At least 2");
});
test("public Stoa references download on selection and reject changed publisher files", async ({page}) => {
  let downloads = 0;
  await page.route("https://raw.githubusercontent.com/**", async route => {
    downloads++;
    await route.fulfill({headers:{"access-control-allow-origin":"*"}, contentType:"application/pdf", body:"changed"});
  });
  await page.goto("/");
  await expect(page.getByRole("heading", {name:"Try a public Stoa reference sheet"})).toBeVisible();
  await expect(page.getByRole("button", {name:/A-701/})).toBeVisible();
  expect(downloads).toBe(0);
  await page.getByRole("button", {name:/A-801/}).click();
  await expect(page.getByRole("alert")).toContainText("does not match the verified Stoa sheet");
  await expect(page.getByTestId("source-image")).toHaveCount(0);
  expect(downloads).toBe(1);
});
test("an oversized PDF sends only the selected vector page, with page-bound history and evidence", async ({page}) => {
  const doc = await PDFDocument.create();
  doc.addPage([240,120]);
  doc.addPage([300,180]).drawText("Second sheet D-01", {x: 50, y: 50, size: 12});
  await doc.attach(randomBytes(4_100_000), "synthetic.bin", {mimeType: "application/octet-stream"});
  const bytes = Buffer.from(await doc.save());
  expect(bytes.byteLength).toBeGreaterThan(4_000_000);
  let requests = 0;
  await page.route("**/api/reader", async route => {
    requests++;
    const sent = await submitted(route);
    expect(sent.page).toBe(1);
    expect(sent.file.type).toBe("application/pdf");
    expect(sent.question).toBe("Where is door D-01?");
    const derivative = await PDFDocument.load(sent.bytes);
    expect(derivative.getPageCount()).toBe(1);
    expect(derivative.getPages()[0].getSize()).toEqual({width: 300, height: 180});
    await route.fulfill({json: answer(sent.bytes, 1, sent.file.type, 300, 180)});
  });
  await page.goto("/");
  await page.getByLabel("Upload drawing", {exact: true}).setInputFiles({name:"two.pdf", mimeType:"application/pdf", buffer:bytes});
  await page.getByRole("button", {name:"Next page"}).click();
  await expect(page.getByLabel("PDF page")).toHaveValue("2");
  await expect(page.getByRole("button", {name:"Next page"})).toBeDisabled();
  await expect(page.getByText("Page 2 will be sent as a single-page PDF", {exact:false})).toBeVisible();
  expect(requests).toBe(0);
  await ask(page);
  await expect(page.getByTestId("answer-value")).toHaveText("D-01");
  await page.getByRole("button", {name:"View evidence 1, page 2"}).click();
  await expect(page.getByTestId("evidence-box")).toHaveAttribute("x", "80");
  await page.getByRole("button", {name:"Previous page"}).click();
  await expect(page.getByTestId("answer-value")).toHaveCount(0);
  await page.getByRole("button", {name:"Next page"}).click();
  await expect(page.getByTestId("answer-value")).toHaveText("D-01");
  await page.reload();
  await expect(page.getByTestId("answer-value")).toHaveText("D-01");
  expect(requests).toBe(1);
});

test("an oversized raster needs consent and a partial answer retains its value and limitation", async ({page}) => {
  await page.goto("/");
  const source = await imageFile(page, "image/png");
  const oversize = Buffer.concat([source, Buffer.alloc(4_000_001)]);
  let requests = 0;
  await page.route("**/api/reader", async route => {
    requests++;
    const sent = await submitted(route);
    expect(sent.bytes.length).toBeLessThan(4_000_000);
    await route.fulfill({json:{...answer(sent.bytes, 1, sent.file.type, 320, 180),
      status:"partial", value:1, unit:"count", reason:"Raster text coverage is incomplete."}});
  });
  await page.getByLabel("Upload drawing", {exact: true}).setInputFiles({name:"large.png", mimeType:"image/png", buffer:oversize});
  await page.getByLabel("Question about this drawing").fill("How many distinct door marks are printed on this sheet?");
  const send = page.getByRole("button", {name:"Ask Planproof", exact:true});
  await expect(send).toBeDisabled();
  expect(requests).toBe(0);
  await page.getByRole("checkbox", {name:"Use the displayed", exact:false}).check();
  await send.click();
  await expect(page.getByText("Partial result", {exact:true})).toBeVisible();
  await expect(page.getByTestId("answer-value")).toHaveText("1");
  await expect(page.getByText("Raster text coverage is incomplete.")).toBeVisible();
});

test("JPEG originals are sent intact, mismatched sources and reader errors cannot produce evidence", async ({page}) => {
  await page.goto("/");
  const source = await imageFile(page, "image/jpeg");
  let attempt = 0;
  await page.route("**/api/reader", async route => {
    const sent = await submitted(route);
    expect(sent.bytes.equals(source)).toBeTruthy();
    if (++attempt === 1) await route.fulfill({json:{...answer(sent.bytes, 1, sent.file.type, 320, 180), source_sha256:"wrong"}});
    else if (attempt === 2) await route.fulfill({status:422, json:{detail:"The reader could not open this sheet."}});
    else await route.fulfill({json:{...answer(sent.bytes, 1, sent.file.type, 320, 180), status:"unresolved",
      value:null, calculation:null, reason:"The printed door mark is ambiguous.", evidence:[]}});
  });
  await page.getByLabel("Upload drawing", {exact: true}).setInputFiles({name:"sheet.jpg", mimeType:"image/jpeg", buffer:source});
  await ask(page);
  await expect(page.getByRole("alert")).toContainText("did not match this sheet");
  await ask(page);
  await expect(page.getByRole("alert")).toContainText("could not open this sheet");
  await ask(page);
  await expect(page.getByTestId("answer-value")).toHaveText("Not enough evidence to answer");
  await expect(page.getByText("The printed door mark is ambiguous.")).toBeVisible();
  await expect(page.getByTestId("evidence-box")).toHaveCount(0);
});
