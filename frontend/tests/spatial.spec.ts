import { createHash } from "node:crypto";
import { expect, test, type Page } from "@playwright/test";
import { PDFDocument } from "pdf-lib";

async function openSample(page: Page) {
  await page.goto("/");
  await page.getByRole("button", {name:"Explore the 3D sample"}).click();
  await expect(page.getByTestId("spatial-stage")).toBeVisible();
}

test("the bundled sample opens in 3D and supports orbit, top view, and adjustable sketch depth", async ({page}) => {
  let readerCalls = 0;
  await page.route("**/api/reader", route => {
    readerCalls++;
    return route.fulfill({status:500});
  });
  await openSample(page);
  const stage = page.getByTestId("spatial-stage");
  await expect(stage).toHaveAttribute("data-yaw","-28");
  await expect(page.getByTestId("sketch-volume")).toHaveCount(3);
  await page.getByRole("button", {name:"Rotate right",exact:true}).click();
  await expect(stage).toHaveAttribute("data-yaw","2");
  await stage.focus();
  await page.keyboard.press("ArrowUp");
  await expect(stage).toHaveAttribute("data-tilt","57");
  const matrix = await page.getByTestId("spatial-projection").getAttribute("transform");
  await page.getByRole("button", {name:"3D zoom in",exact:true}).click();
  await expect(page.getByTestId("spatial-projection")).not.toHaveAttribute("transform", matrix!);
  const roof = page.getByTestId("sketch-volume").first().locator(".spatial-roof");
  const points = await roof.getAttribute("points");
  await page.getByRole("slider").fill("85");
  await expect(roof).not.toHaveAttribute("points", points!);
  await page.getByRole("button", {name:"Show sketch volumes",exact:true}).click();
  await expect(page.getByTestId("sketch-volume")).toHaveCount(0);
  await page.getByRole("button", {name:"Top view",exact:true}).click();
  await expect(stage).toHaveAttribute("data-tilt","0");
  await page.getByRole("button", {name:"Return to 2D"}).click();
  await expect(page.getByTestId("source-image")).toBeVisible();
  await page.getByRole("button", {name:"Explore in 3D"}).click();
  await expect(stage).toHaveAttribute("data-tilt","0");
  expect(readerCalls).toBe(0);
});

test("an uploaded PDF keeps its evidence anchored and sketches stay on their page", async ({page}) => {
  const pdf = await PDFDocument.create();
  pdf.addPage([320,220]).drawText("Room 1", {x:20,y:195});
  pdf.addPage([320,220]).drawText("Room 2", {x:20,y:195});
  const bytes = Buffer.from(await pdf.save());
  await page.route("**/api/reader", route => route.fulfill({json:{
    source_sha256:createHash("sha256").update(bytes).digest("hex"),
    page:1, width:320, height:220, coordinates:"displayed_pdf_points",
    status:"answered", value:"Room 1", unit:null, reason:"",
    evidence:[{id:"room",kind:"printed_label",text:"Room 1",bbox:[20,13,69,25]}],
  }}));
  await page.goto("/");
  await page.getByLabel("Upload drawing", {exact:true}).setInputFiles({name:"drawing.pdf",mimeType:"application/pdf",buffer:bytes});
  await page.getByLabel("Question about this drawing").fill('Find the label "Room 1"');
  await page.getByRole("button",{name:"Ask Planproof",exact:true}).click();
  await expect(page.getByTestId("answer-value")).toHaveText("Room 1");
  await page.getByRole("button",{name:"Explore in 3D"}).click();
  const pin = page.getByRole("button",{name:"Show 3D evidence 1",exact:true});
  await pin.click();
  await expect(pin).toHaveAttribute("aria-pressed","true");
  await expect(page.locator(".spatial-caption")).toContainText("Room 1");
  await page.getByRole("button",{name:"Focus selected evidence"}).click();
  const model = page.getByTestId("spatial-projection");
  const projectedAnchor = () => model.evaluate((node) => {
    const group = node as SVGGElement;
    const matrix = group.transform.baseVal.consolidate()!.matrix;
    const expected = new DOMPoint(44.5,19).matrixTransform(matrix);
    const base = group.ownerSVGElement!.querySelector(".spatial-pin-base")!;
    return [Number(base.getAttribute("cx"))-expected.x, Number(base.getAttribute("cy"))-expected.y];
  });
  for (const delta of await projectedAnchor()) expect(Math.abs(delta)).toBeLessThan(.001);
  await page.getByRole("button",{name:"Rotate left",exact:true}).click();
  for (const delta of await projectedAnchor()) expect(Math.abs(delta)).toBeLessThan(.001);
  await expect(page.getByTestId("sketch-volume")).toHaveCount(0);
  await page.getByRole("button",{name:"Sketch",exact:true}).click();
  for (const p of [[84,92],[162,92],[162,148],[84,148]]) {
    const position = await model.evaluate((node, [x,y]) => {
      const mapped = new DOMPoint(x,y).matrixTransform((node as unknown as SVGGElement).getScreenCTM()!);
      return [mapped.x,mapped.y];
    }, p);
    await page.mouse.click(position[0], position[1]);
  }
  await page.getByRole("button",{name:"Raise sketch"}).click();
  await expect(page.getByTestId("sketch-volume")).toHaveCount(1);
  await page.getByRole("button",{name:"Next page"}).click();
  await expect(page.getByLabel("PDF page")).toHaveValue("2");
  await page.getByRole("button",{name:"Explore in 3D"}).click();
  await expect(page.getByTestId("sketch-volume")).toHaveCount(0);
  await expect(page.getByTestId("spatial-pin")).toHaveCount(0);
});

test("the sample can be rotated and switched back to 2D on a touch viewport", async ({browser}) => {
  const page = await browser.newPage({viewport:{width:390,height:844}, isMobile:true, hasTouch:true});
  await openSample(page);
  await expect(page.getByRole("button",{name:"Measure",exact:true})).toBeVisible();
  const presets = await page.locator(".spatial-top").boundingBox();
  const modes = await page.locator(".plan-view-modes").boundingBox();
  expect(presets!.x+presets!.width).toBeLessThan(modes!.x);
  await page.getByRole("button",{name:"Rotate left",exact:true}).tap();
  await expect(page.getByTestId("spatial-stage")).toHaveAttribute("data-yaw","-58");
  await page.getByRole("button",{name:"Return to 2D"}).tap();
  await expect(page.getByTestId("source-image")).toBeVisible();
  await page.close();
});
