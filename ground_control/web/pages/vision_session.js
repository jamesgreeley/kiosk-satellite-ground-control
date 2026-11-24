import { api, audioUrl } from "../api.js";
import { shell } from "../app.js";
import { ask, chip, clock, h, icon, secs, section, toast } from "../ui.js";

const ENDS = {
  idle: "Ended when nothing was seen for 10 s",
  closed: "Ended when the kiosk's camera stopped",
  long: "Went on in a new session after 30 minutes",
};

const REASONS = {
  first: "First look", facing: "Looking at the kiosk", reading: "New finger reading", last: "Last look", look: "Look",
};

const ms = (v) => (v === null || v === undefined ? "" : `${Number(v).toFixed(v < 10 ? 1 : 0)} ms`);
const fingers = (n) => `${n} finger${n === 1 ? "" : "s"}`;

/** One lane of the timeline: a mark at each look's time. */
function lane(label, looks, total, mark) {
  return h("div", { class: "vlane" },
    h("span", { class: "vll" }, label),
    h("div", { class: "vlt" }, looks.map((lk) => {
      const m = mark(lk);
      if (!m) return null;
      return h("span", { class: `vm ${m.cls || ""}`, title: m.title, style: { left: `${(lk.t / total) * 100}%`, ...m.style } }, m.text || null);
    })));
}

function timeline(record, total) {
  const looks = record.looks || [];
  const gc = (lk) => Object.values(lk.ms || {}).reduce((a, b) => a + b, 0);
  const top = Math.max(1, ...looks.map((lk) => Math.max(gc(lk), lk.rtt || 0)));
  const outcomes = (record.events || []).filter((e) => ["screensaver_dismissed", "screensaver_postponed", "gesture"].includes(e.type));
  return h("div", { class: "card tl" },
    h("div", { class: "legend" },
      h("span", {}, h("i", { class: "swb", style: { background: "var(--primary)" } }), "Face looking at the kiosk"),
      h("span", {}, h("i", { class: "swb", style: { background: "var(--other)" } }), "Face turned away"),
      h("span", {}, h("i", { class: "swb", style: { background: "var(--warn)" } }), "Hand, with its finger count"),
      h("span", {}, h("i", { class: "swb", style: { background: "var(--primary)", opacity: 0.5 } }), "Ground Control's time, and the round trip outlined")),
    h("div", { class: "vlanes" },
      outcomes.map((e) => h("div", { class: "mk", style: { left: `calc(84px + (100% - 84px) * ${e.t / total})` } },
        h("span", { class: "mklbl" }, e.type === "gesture" ? fingers(e.fingers ?? 0) : e.type === "screensaver_dismissed" ? "Dismissed" : "Postponed"))),
      lane("Face", looks, total, (lk) => (lk.face
        ? { cls: lk.face.frontal ? "on" : "away", title: `${lk.t} s: ${lk.face.count} face${lk.face.count === 1 ? "" : "s"}, ${lk.face.frontal ? "looking at the kiosk" : "turned away"}` }
        : lk.want.includes("f") ? { cls: "none", title: `${lk.t} s: no face` } : null)),
      lane("Hand", looks, total, (lk) => (lk.hand
        ? { cls: "hand", text: lk.hand.fingers ?? "", title: `${lk.t} s: ${lk.hand.gesture}${lk.hand.fingers != null ? `, ${fingers(lk.hand.fingers)}` : ""}` }
        : lk.want.includes("h") ? { cls: "none", title: `${lk.t} s: no hand` } : null)),
      lane("Time", looks, total, (lk) => ({
        cls: "lat", title: `${lk.t} s: Ground Control ${ms(gc(lk))}${lk.rtt ? `, round trip ${ms(lk.rtt)}` : ""}`,
        style: { height: `${Math.max(2, (gc(lk) / top) * 40)}px`, "--rtt": `${lk.rtt ? Math.max(2, (lk.rtt / top) * 40) : 0}px` },
      }))),
    h("div", { class: "axis", style: { marginLeft: "84px" } }, h("span", {}, "0 s"), h("span", {}, secs(total / 2)), h("span", {}, secs(total))));
}

function steps(record) {
  const out = [];
  const events = record.events || [];
  const handAt = new Set(events.filter((e) => e.type === "hand").map((e) => e.t));
  for (const e of events) {
    // A first face already looking at the kiosk, or a first hand already
    // with a count, is one step.
    if (e.type === "face" && e.frontal) continue;
    if (e.type === "fingers" && handAt.has(e.t)) continue;
    if (e.type === "face") out.push(["Face seen", e.frontal ? "Looking at the kiosk." : "Turned away from the kiosk.", e.t]);
    if (e.type === "facing") out.push(["Looking at the kiosk", `The face took ${Math.round(e.width * 100)}% of the frame's width.`, e.t]);
    if (e.type === "hand") out.push(["Hand seen", `Read as ${e.gesture}${e.fingers != null ? `, ${fingers(e.fingers)}` : ""}.`, e.t]);
    if (e.type === "fingers") out.push([`${fingers(e.fingers)} read`, `Read as ${e.gesture}.`, e.t]);
    if (e.type === "screensaver_dismissed") out.push(["Screensaver dismissed", "The kiosk woke the screen for the face looking at it.", e.t]);
    if (e.type === "screensaver_postponed") out.push(["Screensaver postponed", "Someone was still in front of the kiosk.", e.t]);
    if (e.type === "gesture") out.push([`${fingers(e.fingers ?? 0)} gesture`, e.action ? `${e.action}.` : "The kiosk ran the gesture's action.", e.t]);
    if (e.type === "end") out.push(["End", `${ENDS[e.reason] || "Ended"}.`, e.t]);
  }
  return out.sort((a, b) => a[2] - b[2]).map(([title, desc, t]) => h("div", { class: "row" },
    h("div", { class: "grow" }, h("div", { class: "rt" }, title), h("div", { class: "rd" }, desc)),
    h("span", { class: "t" }, `${Number(t).toFixed(2)} s`)));
}

const TIMES = [
  ["ground_control", "Ground Control", "The whole look on Ground Control: reading the frame and running the models."],
  ["decode", "Reading the frame", "Decoding the JPEG and turning it upright."],
  ["face", "Face Detection", "The face model."],
  ["hands", "Hand Gestures", "The hand model."],
  ["encode", "Kiosk encoding", "The kiosk turning the camera frame into a JPEG."],
  ["round_trip", "Round trip", "From the kiosk sending a frame to its result coming back, network included."],
];

function performance(stats) {
  const t = stats.ms || {};
  const cell = (v, label) => h("span", { class: "pv" }, h("b", {}, ms(v)), h("small", {}, label));
  const rows = TIMES.filter(([k]) => t[k]?.p50 != null).map(([k, label, about]) => h("div", { class: "row" },
    h("div", { class: "grow" }, h("div", { class: "rt" }, label), h("div", { class: "rd" }, about)),
    h("span", { class: "pvs" }, cell(t[k].p50, "median"), cell(t[k].p95, "95%"), cell(t[k].max, "slowest"))));
  const lost = [];
  if (stats.dropped) lost.push(`${stats.dropped} frame${stats.dropped === 1 ? "" : "s"} dropped for a newer one`);
  if (stats.failed) lost.push(`${stats.failed} frame${stats.failed === 1 ? "" : "s"} that couldn't be read`);
  return h("div", { class: "card" },
    rows.length ? rows : h("div", { class: "empty" }, "No times for this session."),
    h("div", { class: "row" }, h("div", { class: "grow" }, h("div", { class: "rd" },
      `${stats.looks} looks: ${stats.face_looks} for faces and ${stats.hand_looks} for hands. ${lost.length ? `${lost.join(" and ")}.` : "No frames lost."}`))));
}

/** A kept frame, upright, with the faces and the hand drawn on it. */
function frame(id, f, styles) {
  const canvas = h("canvas", { class: "vf", role: "img", "aria-label": `${REASONS[f.reason] || "Look"} at ${f.t} s` });
  audioUrl(`api/sessions/${id}/frames/${f.n}`).then((url) => {
    const img = new Image();
    img.onload = () => {
      const turn = ((f.rotation || 0) % 360 + 360) % 360;
      const side = turn === 90 || turn === 270;
      const w = side ? img.height : img.width;
      const ht = side ? img.width : img.height;
      canvas.width = w;
      canvas.height = ht;
      const c = canvas.getContext("2d");
      c.save();
      c.translate(w / 2, ht / 2);
      c.rotate((turn * Math.PI) / 180);
      c.drawImage(img, -img.width / 2, -img.height / 2);
      c.restore();
      c.lineWidth = Math.max(2, w / 220);
      c.font = `600 ${Math.max(14, w / 30)}px system-ui, sans-serif`;
      for (const [x, y, bw, bh, frontal] of f.faces || []) {
        c.strokeStyle = frontal ? styles.primary : styles.other;
        c.strokeRect(x * w, y * ht, bw * w, bh * ht);
      }
      if (f.hand) {
        const [x, y, bw, bh, n] = f.hand;
        c.strokeStyle = styles.warn;
        c.strokeRect(x * w, y * ht, bw * w, bh * ht);
        if (n != null) {
          c.fillStyle = styles.warn;
          c.fillText(String(n), x * w, Math.max(16, y * ht - 6));
        }
      }
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }).catch(() => canvas.replaceWith(h("div", { class: "empty" }, "This frame is gone.")));
  return h("figure", { class: "vfig" }, canvas,
    h("figcaption", {}, h("span", {}, REASONS[f.reason] || "Look"), h("span", { class: "sub" }, `${Number(f.t).toFixed(2)} s`)));
}

export async function renderVision(record) {
  const id = record.id;
  const stats = record.stats || {};
  const total = Math.max(0.5, stats.duration_s || 0);
  const where = record.device_name || record.device_id || "Unknown kiosk";
  const css = getComputedStyle(document.documentElement);
  const styles = { primary: css.getPropertyValue("--primary").trim(), other: css.getPropertyValue("--other").trim(),
    warn: css.getPropertyValue("--warn").trim() };

  const remove = async () => {
    if (!await ask({ title: "Delete this session?", text: "Its frames are deleted too.", ok: "Delete", danger: true })) return;
    await api(`api/sessions/${id}`, { method: "DELETE" });
    location.hash = "#/sessions";
  };
  const download = async () => {
    try {
      const a = h("a", { href: await audioUrl(`api/sessions/${id}/download`), download: `ground-control-session-${id}.zip` });
      a.click();
    } catch (err) { toast(`Couldn't download the session: ${err.message}`); }
  };

  const frames = record.frames || [];
  return {
    nodes: [
      shell("sessions", {
        back: "#/sessions", title: `${where}, ${clock(record.started)}`,
        actions: [
          h("button", { class: "ib", "aria-label": "Download the session", title: "Download the session: its frames, details and log", onclick: download }, icon("download")),
          h("button", { class: "ib", "aria-label": "Delete session", onclick: remove }, icon("trash")),
        ],
      }),
      h("main", { class: "page" },
        h("div", { class: "chips" },
          (record.summary || []).map((w) => chip(w, `big ${/dismissed|→/.test(w) ? "c-ok" : "c-info"}`)),
          chip(`Vision Match, ${secs(stats.duration_s)}`, "big", icon("eye", 16)),
          chip(`${record.model === "accurate" ? "Accurate" : "Light"} model, ${record.on_gpu ? "GPU" : "CPU"}`, "big"),
          record.continues ? h("a", { class: "chip big", href: `#/sessions/${record.continues}` }, "Goes on from an earlier session") : null),
        section("Timeline"),
        timeline(record, total),
        h("div", { class: "cols" },
          h("section", {}, section("Steps"), h("div", { class: "card" }, steps(record))),
          h("section", {}, section("Performance"), performance(stats))),
        section("Frames", frames.length ? "Faces looking at the kiosk are outlined in teal, faces turned away in gray and the hand in amber with its finger count." : null),
        frames.length
          ? h("div", { class: "vframes" }, frames.map((f) => frame(id, f, styles)))
          : h("div", { class: "card" }, h("div", { class: "empty" }, "No frames were kept for this session. Keep frames is in Settings, Vision Match."))),
    ],
  };
}
