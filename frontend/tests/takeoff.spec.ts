import { expect, test, type Page } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { PDFDocument } from "pdf-lib";

async function openSample(page:Page) {
  await page.goto("/");
  await page.getByRole("button",{name:"Explore the 3D sample"}).click();
  await page.getByRole("button",{name:"Measure",exact:true}).waitFor();
}

async function tapSource(page:Page,x:number,y:number) {
  const rectangle=await page.getByTestId("source-image").boundingBox();
  if (!rectangle) throw new Error("drawing is not visible");
  await page.mouse.click(rectangle.x+x*rectangle.width,rectangle.y+y*rectangle.height);
}

test("manual scale converts a length and a drawn area and supports count markers",async({page})=>{
  let readerCalls=0;
  await page.route("**/api/reader",route=>{readerCalls++;return route.fulfill({status:500});});
  await openSample(page);
  await page.getByRole("button",{name:"Measure",exact:true}).click();
  await page.getByRole("button",{name:"Scale",exact:true}).click();
  await tapSource(page,160/1120,669/740);
  await tapSource(page,360/1120,669/740);
  await page.getByLabel("Reference distance").fill("200");
  await page.getByLabel("Distance unit").selectOption("cm");
  await page.getByRole("button",{name:"Save scale"}).click();
  await expect(page.getByText("2 m reference · Set on this sheet")).toBeVisible();
  await page.getByRole("button",{name:"Length",exact:true}).click();
  await tapSource(page,160/1120,520/740);
  await tapSource(page,560/1120,520/740);
  await page.getByRole("button",{name:"Save measurement"}).click();
  await expect(page.getByRole("button",{name:"Show Length 1 on plan"})).toContainText(/(?:3\.9[5-9]|4(?:\.0[0-5])?) m/);
  await page.getByRole("button",{name:"Area",exact:true}).click();
  await page.getByRole("button",{name:"Fit drawing",exact:true}).click();
  for(const [x,y] of [[160,420],[960,420],[960,580],[160,580]]) await tapSource(page,x/1120,y/740);
  await page.getByRole("button",{name:"Save measurement"}).click();
  await expect(page.getByRole("button",{name:"Show Area 1 on plan"})).toContainText(/12\.[5-9].*m²/);
  await page.getByRole("button",{name:"Count",exact:true}).click();
  await page.getByRole("button",{name:"Fit drawing",exact:true}).click();
  await tapSource(page,.21,.4);
  await tapSource(page,.6,.4);
  await tapSource(page,.7,.45);
  await page.getByRole("button",{name:"Undo last measurement point"}).click();
  await page.getByRole("button",{name:"Save measurement"}).click();
  await expect(page.getByRole("button",{name:"Show Count 1 on plan"})).toContainText("2 pcs");
  await expect(page.getByRole("img",{name:"Measurement marks"}).locator("[data-takeoff-item]")).toHaveCount(3);
  expect(readerCalls).toBe(0);
});

test("imported answer evidence is saved, priced and exported with its calibration",async({page})=>{
  await page.route("**/api/reader",async route=>{
    const request=route.request();
    const data=await new Response(new Uint8Array(request.postDataBuffer()!),{
      headers:{"content-type":request.headers()["content-type"]},
    }).formData();
    const payload=Buffer.from(await (data.get("file") as File).arrayBuffer());
    await route.fulfill({json:{
      status:"answered",value:12.8,unit:"m²",approximate:true,calculation:"Measured floor and graphic scale",reason:"",
      source_sha256:createHash("sha256").update(payload).digest("hex"),page:1,width:1120,height:740,coordinates:"image_pixels",
      evidence:[
        {id:"floor",kind:"room_polygon",text:"Hallway",bbox:[160,420,960,580],polygon:[[160,420],[960,420],[960,580],[160,580]]},
        {id:"scale",kind:"graphic_scale",text:"2 m",bbox:[150,640,390,700],line:[[160,669],[360,669]]},
      ],
    }});
  });
  await openSample(page);
  await page.getByLabel("Question about this drawing").fill("Mark me the hallway, how big is it?");
  await page.getByRole("button",{name:"Ask Planproof",exact:true}).click();
  await expect(page.getByTestId("answer-value")).toHaveText("≈ 12.8 m²");
  await page.getByRole("button",{name:"Measure",exact:true}).click();
  await page.getByRole("button",{name:"Add outline from answer"}).click();
  await expect(page.getByRole("button",{name:"Show Hallway on plan"})).toContainText("12.8 m²");
  await expect(page.getByText("2 m reference · From a plan answer")).toBeVisible();
  await page.getByRole("button",{name:"Edit Hallway"}).click();
  await page.getByLabel("Item name").fill("=Hallway");
  await page.getByLabel("Price per m² (€)").fill("25,50");
  await page.getByRole("button",{name:"Save item"}).click();
  await expect(page.getByRole("region",{name:"Estimate"})).toContainText("€326.40");
  await page.reload();
  await page.getByRole("button",{name:"Measure",exact:true}).click();
  await expect(page.getByRole("button",{name:"Show =Hallway on plan"})).toContainText("12.8 m²");
  const getting=page.waitForEvent("download");
  await page.getByRole("button",{name:"Export CSV"}).click();
  const file=await getting;
  expect(file.suggestedFilename()).toMatch(/-page-1-takeoff\.csv$/);
  const csv=await readFile((await file.path())!,"utf8");
  expect(csv).toContain(`"'=Hallway","Area","12.8","m²","25.5","326.4","answer","2","answer"`);
});

test("takeoff stays bound to the PDF page on a phone",async({browser})=>{
  const page=await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
  const doc=await PDFDocument.create();
  doc.addPage([330,310]);
  doc.addPage([330,310]);
  await page.goto("/");
  await page.getByLabel("Upload drawing",{exact:true}).setInputFiles({name:"pages.pdf",mimeType:"application/pdf",buffer:Buffer.from(await doc.save())});
  await page.getByRole("button",{name:"Show drawing",exact:true}).click();
  await page.getByRole("button",{name:"Measure",exact:true}).click();
  await page.getByRole("button",{name:"Count",exact:true}).click();
  await tapSource(page,.4,.52);
  await page.getByRole("button",{name:"Save measurement"}).click();
  await page.getByRole("button",{name:"Show takeoff",exact:true}).click();
  await expect(page.getByRole("button",{name:"Show Count 1 on plan"})).toContainText("1 pcs");
  await page.getByRole("button",{name:"Show drawing",exact:true}).click();
  await page.getByRole("button",{name:"Next page",exact:true}).click();
  await expect(page.getByLabel("PDF page")).toHaveValue("2");
  await page.getByRole("button",{name:"Measure",exact:true}).click();
  await page.getByRole("button",{name:"Show takeoff",exact:true}).click();
  await expect(page.getByRole("button",{name:"Export CSV"})).toBeDisabled();
  await page.getByRole("button",{name:"Show drawing",exact:true}).click();
  await page.getByRole("button",{name:"Previous page",exact:true}).click();
  await expect(page.getByLabel("PDF page")).toHaveValue("1");
  await page.getByRole("button",{name:"Measure",exact:true}).click();
  await page.getByRole("button",{name:"Show takeoff",exact:true}).click();
  await expect(page.getByRole("button",{name:"Show Count 1 on plan"})).toContainText("1 pcs");
  await page.close();
});
