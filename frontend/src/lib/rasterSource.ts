export const maxImagePixels = 150_000_000;
const maxImageEdge = 32_768;

function checkedFrame(width: number, height: number, blob: Blob) {
  if (!width || !height) throw new Error("This image has invalid dimensions.");
  if (width * height > maxImagePixels || Math.max(width, height) > maxImageEdge)
    throw new Error("Open an image up to 150 megapixels and 32,768 pixels per side, or export a detail at its original resolution.");
  return {width, height, blob};
}

/** Read the encoded pixel frame before asking the browser to allocate a bitmap. */
export async function rasterSource(file: File) {
  if (file.type === "image/png") {
    const bytes = new Uint8Array(await file.slice(0, 24).arrayBuffer());
    if (bytes.length !== 24 || ![137,80,78,71,13,10,26,10].every((v,i) => bytes[i] === v)
      || String.fromCharCode(...bytes.subarray(12,16)) !== "IHDR")
      throw new Error("This PNG has an invalid image header.");
    const header = new DataView(bytes.buffer);
    return checkedFrame(header.getUint32(16), header.getUint32(20), file);
  }
  // A JPEG segment is at most 65,535 bytes. Skip unrelated metadata without
  // loading the compressed image into a second JS buffer.
  let offset = 2, start = 0, keptFrom = 0, width = 0, height = 0;
  let chunk = new Uint8Array(await file.slice(0, 65_544).arrayBuffer());
  if (chunk[0] !== 255 || chunk[1] !== 216) throw new Error("This JPEG has an invalid image header.");
  const parts: Blob[] = [];
  while (offset < file.size) {
    if (offset + 10 > start + chunk.length) {
      start = offset;
      chunk = new Uint8Array(await file.slice(start, start + 65_544).arrayBuffer());
    }
    const i = offset - start, marker = chunk[i + 1];
    if (chunk[i] !== 255 || marker === undefined) break;
    if (marker === 255) {offset++; continue;}
    if (marker === 0xda || marker === 0xd9) break; // Start of scan / end
    const length = (chunk[i + 2] << 8) + chunk[i + 3];
    const end = offset + 2 + length;
    if (length < 2 || end > file.size) break;
    if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) {
      height = (chunk[i + 5] << 8) + chunk[i + 6];
      width = (chunk[i + 7] << 8) + chunk[i + 8];
    }
    // Strip EXIF from the display copy so all browsers match the backend's
    // stored pixel coordinates, including for a rotated camera JPEG.
    if (marker === 0xe1 && String.fromCharCode(...chunk.subarray(i+4, i+10)) === "Exif\0\0") {
      parts.push(file.slice(keptFrom, offset));
      keptFrom = end;
    }
    offset = end;
  }
  const blob = parts.length ? new Blob([...parts, file.slice(keptFrom)], {type: file.type}) : file;
  return checkedFrame(width, height, blob);
}
