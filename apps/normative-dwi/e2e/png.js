// Decode the PNGs Playwright writes (8-bit RGB or RGBA, not interlaced) to RGBA pixels.
import { unzlibSync } from "fflate";

/** { width, height, rgba } of a PNG; rgba[4 * (y * width + x) + c]. */
export function decodePng(buffer) {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let width = 0, height = 0, channels = 0;
  const data = [];
  for (let at = 8; at < bytes.length;) {
    const length = view.getUint32(at);
    const type = String.fromCharCode(...bytes.subarray(at + 4, at + 8));
    const chunk = bytes.subarray(at + 8, at + 8 + length);
    if (type === "IHDR") {
      width = view.getUint32(at + 8);
      height = view.getUint32(at + 12);
      const [depth, colour, , , interlace] = chunk.subarray(8, 13);
      channels = { 2: 3, 6: 4 }[colour];
      if (depth !== 8 || !channels || interlace) throw new Error(`unsupported PNG: depth ${depth}, colour type ${colour}, interlace ${interlace}`);
    } else if (type === "IDAT") data.push(chunk);
    at += 12 + length;
  }
  const joined = new Uint8Array(data.reduce((n, c) => n + c.length, 0));
  data.reduce((offset, c) => (joined.set(c, offset), offset + c.length), 0);
  const raw = unzlibSync(joined);
  const stride = width * channels;
  const pixels = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let i = 0; i < stride; i++) {
      const left = i >= channels ? pixels[y * stride + i - channels] : 0;
      const up = y > 0 ? pixels[(y - 1) * stride + i] : 0;
      const corner = y > 0 && i >= channels ? pixels[(y - 1) * stride + i - channels] : 0;
      const p = left + up - corner;
      const paeth = Math.abs(p - left) <= Math.abs(p - up) && Math.abs(p - left) <= Math.abs(p - corner) ? left : Math.abs(p - up) <= Math.abs(p - corner) ? up : corner;
      const predictor = [0, left, up, (left + up) >> 1, paeth][filter];
      if (predictor === undefined) throw new Error(`unknown PNG filter ${filter}`);
      pixels[y * stride + i] = (line[i] + predictor) & 255;
    }
  }
  const rgba = new Uint8Array(width * height * 4);
  for (let p = 0; p < width * height; p++) {
    rgba.set(pixels.subarray(p * channels, p * channels + 3), 4 * p);
    rgba[4 * p + 3] = channels === 4 ? pixels[p * channels + 3] : 255;
  }
  return { width, height, rgba };
}
