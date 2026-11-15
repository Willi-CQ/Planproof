// Export the editable pitch SVG without a server or external network requests.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from '../frontend/node_modules/@playwright/test/index.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dir = path.join(root, 'docs/pitch');
const svg = await fs.readFile(path.join(dir, 'architecture.svg'), 'utf8');
const browser = await chromium.launch({headless: true,
  ...(process.env.PLANPROOF_CHROMIUM_PATH ? {executablePath: process.env.PLANPROOF_CHROMIUM_PATH} : {})});
try {
  const page = await browser.newPage({viewport: {width: 1920, height: 1080}, deviceScaleFactor: 2});
  await page.route('**/*', route => route.abort());
  await page.setContent(`<html><head><meta charset="utf-8"><title>Planproof architecture</title><style>@page{size:20in 11.25in;margin:0}html,body{margin:0;width:1920px;height:1080px}svg{display:block}*{print-color-adjust:exact;-webkit-print-color-adjust:exact}</style></head><body>${svg}</body></html>`);
  await page.evaluate(() => document.fonts.ready);
  const overflow = await page.evaluate(() => [...document.querySelectorAll('svg text')].flatMap(node => {
    const box = node.getBoundingClientRect();
    return box.left < 0 || box.top < 0 || box.right > 1920 || box.bottom > 1080 ? [node.textContent] : [];
  }));
  if (overflow.length) throw new Error(`Text outside slide: ${overflow.join(', ')}`);
  const cardOverflow = await page.evaluate(() => {
    const cards = [...document.querySelectorAll('svg rect')].filter(rect => +rect.getAttribute('rx') >= 15);
    return [...document.querySelectorAll('svg text')].flatMap(node => {
      const x = +node.getAttribute('x'), y = +node.getAttribute('y');
      const containing = cards.map(card => card.getBoundingClientRect())
        .filter(box => x > box.left && x < box.right && y > box.top && y < box.bottom)
        .sort((a, b) => a.width * a.height - b.width * b.height)[0];
      const box = node.getBoundingClientRect();
      return containing && (box.left < containing.left + 12 || box.right > containing.right - 12)
        ? [node.textContent] : [];
    });
  });
  if (cardOverflow.length) throw new Error(`Text exceeds card padding: ${cardOverflow.join(', ')}`);
  await page.screenshot({path: path.join(dir, 'architecture.png')});
  await page.pdf({path: path.join(dir, 'architecture.pdf'), printBackground: true, preferCSSPageSize: true});
  const pdf = await fs.readFile(path.join(dir, 'architecture.pdf'));
  const pages = (pdf.toString('latin1').match(/\/Type\s*\/Page\b/g) || []).length;
  if (pages !== 1) throw new Error(`Expected one PDF page, found ${pages}`);
  console.log(JSON.stringify({png: '3840 x 2160', pdf: '20 x 11.25 in', pages, textWithinSlideAndCards: true, network: 'blocked'}));
} finally {
  await browser.close();
}
