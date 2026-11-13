import { pathToFileURL } from "node:url";
import { readFileSync } from "node:fs";

const [, , tsPfad, csvPfad] = process.argv;
const m = await import(pathToFileURL(tsPfad).href);
const text = readFileSync(csvPfad, "utf8");
const rows = m.parseQuantityCsv(text);
console.log(JSON.stringify({ rows, summary: m.summariseByCategory(rows) }));
