// Offline browser validation of the generated private before/after artifact.
import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs';
import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { writeFile } from 'node:fs/promises';

const [input, receipt] = process.argv.slice(2);
if (!input || !receipt) throw new Error('Usage: node tools/check_production_audit.mjs report.html receipt.json');
const browser = await chromium.launch({ headless: true, ...(process.env.PLANPROOF_BROWSER ? { executablePath: process.env.PLANPROOF_BROWSER } : {}) });
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
const external = [], errors = [];
await context.route('**/*', route => {
  if (/^https?:/.test(route.request().url())) {
    external.push(route.request().url());
    return route.abort();
  }
  return route.continue();
});
const page = await context.newPage();
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(pathToFileURL(resolve(input)).href);
  const data = await page.locator('#data').textContent().then(JSON.parse);
  assert.equal(await page.locator('#cases button').count(), data.cases.length);
  for (let i = 0; i < data.cases.length; i++) {
    await page.locator('#cases button').nth(i).click();
    assert.equal(await page.locator('#question').textContent(), data.cases[i].english);
    assert.equal(await page.locator('#answers .answer').count(), 3);
    assert.ok((await page.locator('#drawing image').getAttribute('href')).startsWith('data:image/png;base64,'));
    const text = await page.locator('#answers').textContent();
    assert.ok(!/undefined|NaN/.test(text));
  }
  for (const [filter, expected] of [
    ['regression', data.summary.regressions.length],
    ['gain', data.summary.gains.length],
    ['unit', data.cases.filter(c => c.count_value_score.correct && !c.score.correct).length],
    ['failed', data.cases.filter(c => c.answer.status !== 'answered').length],
  ]) {
    await page.locator('#filter').selectOption(filter);
    assert.equal(await page.locator('#cases button').count(), expected);
  }
  await page.locator('#filter').selectOption('all');
  await page.locator('#cases button').first().click();
  await page.locator('#old-boxes').check();
  await page.locator('#zoom-in').click();
  assert.equal(await page.locator('#drawing').evaluate(n => n.style.width), '150%');
  await page.locator('#fit').click();
  await page.screenshot({ path: `${receipt}.desktop.png`, fullPage: false });
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.screenshot({ path: `${receipt}.mobile.png`, fullPage: false });
  assert.deepEqual(external, []);
  assert.deepEqual(errors, []);
  const result = { cases: data.cases.length, externalRequests: external.length, browserErrors: errors.length, mobileWidth: 390, passed: true };
  await writeFile(receipt, JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
}
