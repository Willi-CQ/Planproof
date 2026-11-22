const repository = 'https://github.com/OpeningDesign/Sports_Complex';
const revision = 'c0e96c19005d942e2c42b7099c7074fd51aa7ccd';

export interface PublicSheet {
  id: string;
  name: string;
  sourceUrl: string;
  attribution: string;
  licenseUrl: string;
  fileName: string;
  downloadUrl: string;
  sha256: string;
}

function sheet(id: string, title: string, sha256: string): PublicSheet {
  const fileName = `${id} - ${title}.pdf`;
  const path = `Sheets%20-%20Current%20Set/${encodeURIComponent(fileName)}`;
  return {
    id, name: `${id} · ${title}`, fileName, sha256,
    attribution: 'OpeningDesign and the full Xcel Sports Complex project team credited on the sheet',
    licenseUrl: 'https://creativecommons.org/licenses/by-sa/4.0/',
    sourceUrl: `${repository}/blob/${revision}/${path}`,
    downloadUrl: `https://raw.githubusercontent.com/OpeningDesign/Sports_Complex/${revision}/${path}`,
  };
}

export const publicSheets = [
  sheet('A-801', 'Door and Finish Schedules & Wall Types',
    '0c6bc07e4e413738cb8db95cffd3356664b25719fa1f2648f99599a3db92f36c'),
  sheet('A-701', 'Restrooms',
    'a635bb2c0d8d3535cfa47759b339b7d0a844836bf983c4afd18add4c5fb01703'),
];

export async function loadPublicSheet(sheet: PublicSheet): Promise<File> {
  const response = await fetch(sheet.downloadUrl, { credentials: 'omit', referrerPolicy: 'no-referrer' });
  if (!response.ok) throw new Error(`Could not download ${sheet.id} from OpeningDesign (HTTP ${response.status}).`);
  const bytes = await response.arrayBuffer();
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  const hex = Array.from(digest, (byte) => byte.toString(16).padStart(2, '0')).join('');
  if (hex !== sheet.sha256) throw new Error(`The downloaded ${sheet.id} does not match the verified Stoa sheet.`);
  return new File([bytes], sheet.fileName, { type: 'application/pdf' });
}
