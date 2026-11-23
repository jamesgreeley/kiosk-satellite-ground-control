// How loud a session's audio is, from the same WAVs as the spectrograms.
// Levels are dBFS: 0 is the loudest a 16-bit sample can be.

const FRAME = 320; // 20 ms at 16 kHz
const SILENT = -100; // a frame this quiet is silence put in, not sound heard
const CLIPPED = -0.5;

const dbfs = (x) => (x > 0 ? 20 * Math.log10(x) : -Infinity);

/** Peak, speech, average and noise floor of ``samples`` (PCM16), or null
 * when nothing in it was heard. Speech is the 10 loudest 20 ms frames, the
 * floor the frame 20% of the way up from the quietest. Silence put in for
 * what was left out doesn't count. */
export function loudness(samples) {
  const frames = [];
  let peak = 0, energy = 0, heard = 0;
  for (let f = 0; f + FRAME <= samples.length; f += FRAME) {
    let sum = 0;
    for (let i = f; i < f + FRAME; i++) {
      const s = samples[i] / 32768;
      sum += s * s;
      if (Math.abs(s) > peak) peak = Math.abs(s);
    }
    const db = dbfs(Math.sqrt(sum / FRAME));
    if (db < SILENT) continue;
    frames.push(db);
    energy += sum;
    heard += FRAME;
  }
  if (frames.length < 10) return null;
  frames.sort((a, b) => b - a);
  const speech = frames.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
  const floor = Math.max(frames[frames.length - 1 - Math.floor(frames.length * 0.2)], -75);
  return {
    peak: dbfs(peak),
    speech,
    average: dbfs(Math.sqrt(energy / heard)),
    floor,
    over: speech - floor,
    clipped: dbfs(peak) >= CLIPPED,
  };
}
