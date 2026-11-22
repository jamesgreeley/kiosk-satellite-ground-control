// Spectrograms of a session's audio, drawn in the browser from the WAVs the
// page already fetches, so nothing extra is kept on the server.

const SR = 16000;
const NFFT = 512; // 32 ms
const ROWS = 64;
const MAX_COLS = 720;
const RANGE_DB = 70;

// Magma, from silence to the loudest sound.
const STOPS = [
  [0, 0x00, 0x00, 0x04], [0.13, 0x1c, 0x10, 0x44], [0.25, 0x4f, 0x12, 0x7b], [0.38, 0x81, 0x25, 0x81],
  [0.5, 0xb5, 0x36, 0x7a], [0.63, 0xe5, 0x50, 0x64], [0.75, 0xfb, 0x87, 0x61], [0.88, 0xfe, 0xc2, 0x87],
  [1, 0xfc, 0xfd, 0xbf],
];
const LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const v = i / 255;
    let k = 0;
    while (k < STOPS.length - 2 && v > STOPS[k + 1][0]) k++;
    const [a, ...ca] = STOPS[k];
    const [b, ...cb] = STOPS[k + 1];
    const f = (v - a) / (b - a);
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = ca[c] + (cb[c] - ca[c]) * f;
  }
  return lut;
})();

const WINDOW = Float32Array.from({ length: NFFT }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / NFFT));

// FFT bins averaged into each row, on the mel scale up to 8 kHz like the ear.
const BANDS = (() => {
  const mel = (f) => 2595 * Math.log10(1 + f / 700);
  const hz = (m) => 700 * (10 ** (m / 2595) - 1);
  const top = mel(SR / 2);
  const bins = NFFT / 2;
  const out = [];
  for (let r = 0; r < ROWS; r++) {
    const lo = Math.floor((hz((r / ROWS) * top) / (SR / 2)) * bins);
    const hi = Math.max(lo + 1, Math.floor((hz(((r + 1) / ROWS) * top) / (SR / 2)) * bins));
    out.push([Math.min(lo, bins - 1), Math.min(hi, bins)]);
  }
  return out;
})();

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = nr;
      }
    }
  }
}

/** Levels in dB, one column per time step and ROWS rows from low to high. */
export function analyze(samples) {
  const cols = Math.max(1, Math.min(MAX_COLS, Math.floor(samples.length / 128)));
  const hop = samples.length / cols;
  const db = new Float32Array(cols * ROWS);
  const re = new Float32Array(NFFT), im = new Float32Array(NFFT);
  let max = -Infinity;
  for (let c = 0; c < cols; c++) {
    const start = Math.round((c + 0.5) * hop - NFFT / 2);
    for (let i = 0; i < NFFT; i++) {
      const j = start + i;
      re[i] = j >= 0 && j < samples.length ? (samples[j] / 32768) * WINDOW[i] : 0;
      im[i] = 0;
    }
    fft(re, im);
    for (let r = 0; r < ROWS; r++) {
      const [lo, hi] = BANDS[r];
      let p = 0;
      for (let k = lo; k < hi; k++) p += re[k] * re[k] + im[k] * im[k];
      const v = 10 * Math.log10(p / (hi - lo) + 1e-12);
      db[c * ROWS + r] = v;
      if (v > max) max = v;
    }
  }
  return { cols, db, max };
}

/** Draws ``levels`` on ``canvas``, with ``top`` dB as the brightest, so rows
 * drawn against the same top compare: removed sound is dark. */
export function draw(canvas, { cols, db }, top) {
  canvas.width = cols;
  canvas.height = ROWS;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(cols, ROWS);
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < ROWS; r++) {
      const v = Math.max(0, Math.min(1, (db[c * ROWS + r] - (top - RANGE_DB)) / RANGE_DB));
      const i = Math.round(v * 255) * 3;
      const o = ((ROWS - 1 - r) * cols + c) * 4;
      img.data[o] = LUT[i];
      img.data[o + 1] = LUT[i + 1];
      img.data[o + 2] = LUT[i + 2];
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}
