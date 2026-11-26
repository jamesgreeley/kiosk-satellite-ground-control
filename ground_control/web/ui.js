// Small DOM helpers. Text always goes in as text nodes, never as HTML.
import { audioUrl } from "./api.js";

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "value") el.value = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

const PATHS = {
  mark: '<path d="M4 12h1M8 8v8M12 4v16M16 7v10M20 11v2"/>',
  back: '<path d="M19 12H5M11 18l-6-6 6-6"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  edit: '<path d="M4 20h4L19 9l-4-4L4 16z"/>',
  play: '<path d="M8 5v14l11-7z" fill="currentColor" stroke="none"/>',
  pause: '<path d="M7 5h4v14H7zM13 5h4v14h-4z" fill="currentColor" stroke="none"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
  download: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
  copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
  warn: '<path d="M12 4 2.5 20h19z"/><path d="M12 10v4M12 17h.01"/>',
  mic: '<path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z"/><path d="M19 11a7 7 0 0 1-14 0M12 18v3"/>',
  history: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5M12 7v5l3 3"/>',
  screen: '<rect x="3" y="5" width="18" height="13" rx="2"/><path d="M9 21h6"/>',
  chip: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  prev: '<path d="m15 6-6 6 6 6"/>',
  next: '<path d="m9 6 6 6-6 6"/>',
  list: '<path d="M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01"/>',
  calendar: '<rect x="4" y="5" width="16" height="15" rx="2"/><path d="M4 10h16M9 3v4M15 3v4"/>',
  music: '<path d="M9 18V5l11-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="17" cy="16" r="3"/>',
  person: '<circle cx="12" cy="8" r="4"/><path d="M4 20a8 8 0 0 1 16 0"/>',
  walk: '<circle cx="12" cy="5" r="2"/><path d="M10 22l1-7-2-3 3-4 3 4-2 3 1 7"/>',
  turn: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
  upload: '<path d="M12 20V9M7 14l5-5 5 5M5 4h14"/>',
  tune: '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
  cut: '<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M20 4 8.12 15.88M14.47 14.48 20 20M8.12 8.12 12 12"/>',
  github: '<path fill="currentColor" stroke="none" d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.53-1.33-1.28-1.69-1.28-1.69-1.05-.71.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.8 1.19 1.83 1.19 3.09 0 4.42-2.7 5.4-5.26 5.68.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5z"/>',
  coffee: '<path d="M10 2v2M14 2v2M6 2v2M16 8a1 1 0 0 1 1 1v8a4 4 0 0 1-4 4H7a4 4 0 0 1-4-4V9a1 1 0 0 1 1-1h14a4 4 0 1 1 0 8h-1"/>',
  eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  face: '<path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M8 14s1.5 2 4 2 4-2 4-2M9 9h.01M15 9h.01"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  down: '<path d="m6 9 6 6 6-6"/>',
};

export function icon(name, size = 22) {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("width", size);
  s.setAttribute("height", size);
  s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("fill", "none");
  s.setAttribute("stroke", "currentColor");
  s.setAttribute("stroke-width", "1.8");
  s.setAttribute("stroke-linecap", "round");
  s.setAttribute("stroke-linejoin", "round");
  s.setAttribute("aria-hidden", "true");
  s.innerHTML = PATHS[name] || "";
  return s;
}

// Each person's Home Assistant picture, fetched once per page load.
const pictures = new Map();

/** The picture's address names the linked Home Assistant person too, so a new
 * link never shows the picture the browser kept for the old one. */
function picture(person) {
  const entity = person.things?.person || "";
  const key = `${person.id}:${entity}`;
  if (!pictures.has(key)) {
    pictures.set(key, audioUrl(`api/people/${encodeURIComponent(person.id)}/picture?entity=${encodeURIComponent(entity)}`)
      .catch(() => null));
  }
  return pictures.get(key);
}

function showPicture(el, person) {
  if (!person.picture || !person.id) return;
  picture(person).then((url) => {
    if (!url) return;
    el.classList.add("pic");
    el.replaceChildren(h("img", { src: url, alt: "" }));
  });
}

/** A person's circle: their Home Assistant picture in a rim of their color, or their initial. */
export function disc(person, cls = "") {
  if (!person || !person.name) return h("span", { class: `disc unk ${cls}` }, "?");
  const el = h("span", { class: `disc ${cls}`, "data-person": person.id || null, style: { background: person.color || "var(--fill)" } },
    person.name.slice(0, 1).toUpperCase());
  showPicture(el, person);
  return el;
}

/** After a person's Home Assistant link changes: fetch the picture again and redraw their circles. */
export function refreshPicture(person) {
  document.querySelectorAll(`.disc[data-person="${CSS.escape(person.id)}"]`).forEach((el) => {
    el.classList.remove("pic");
    el.replaceChildren(person.name.slice(0, 1).toUpperCase());
    showPicture(el, person);
  });
}

export function chip(text, kind = "", extra = null) {
  return h("span", { class: `chip ${kind}` }, extra, text);
}

export function toggle(on, label, onChange) {
  const b = h("button", { class: "sw", role: "switch", "aria-checked": String(!!on), "aria-label": label }, h("span"));
  b.addEventListener("click", () => {
    const next = b.getAttribute("aria-checked") !== "true";
    b.setAttribute("aria-checked", String(next));
    onChange(next);
  });
  return b;
}

/** A section heading over a card: the title, an optional action on the right and a line under it. */
export function section(title, sub = null, action = null) {
  return h("div", { class: "sec" },
    h("div", { class: "sechead" }, h("h2", { class: "st" }, title), action),
    sub ? h("p", { class: "ss" }, sub) : null);
}

/** The page title, with an optional action on the right. */
export function pageHead(title, action = null) {
  // The top bar already shows which page this is: the title is for screen readers only.
  const heading = h("h1", { class: "sr" }, title);
  return action ? h("div", { class: "phead" }, heading, action) : heading;
}

/**
 * An in-page dialog. Resolves to the field's value (with ``field``), true
 * (confirmed) or null (cancelled).
 */
export function ask({ title, text = null, field = null, ok = "OK", danger = false }) {
  return new Promise((resolve) => {
    const input = field ? h("input", { class: "fld", id: "dlg-field", value: field.value || "",
      placeholder: field.placeholder || "", autocomplete: "off" }) : null;
    const dlg = h("dialog", { class: "dlg", "aria-labelledby": "dlg-title" });
    const done = (value) => { dlg.close(); dlg.remove(); resolve(value); };
    const submit = () => {
      if (!input) return done(true);
      const v = input.value.trim();
      if (!v) { input.focus(); return; }
      done(v);
    };
    append(dlg, [
      h("h2", { class: "dlgt", id: "dlg-title" }, title),
      text ? h("p", { class: "dlgx" }, text) : null,
      input ? h("label", { class: "dlgl", for: "dlg-field" }, field.label) : null,
      input,
      h("div", { class: "dlgb" },
        h("button", { class: "btn txt", onclick: () => done(null) }, "Cancel"),
        h("button", { class: `btn ${danger ? "fill danger-fill" : "fill"}`, onclick: submit }, ok))]);
    input?.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); submit(); } });
    dlg.addEventListener("cancel", (e) => { e.preventDefault(); done(null); });
    dlg.addEventListener("click", (e) => { if (e.target === dlg) done(null); });
    document.body.append(dlg);
    dlg.showModal();
    (input || dlg.querySelector(".btn.fill")).focus();
    if (input) input.select();
  });
}

/** A dialog that only tells: ``content`` and a Close button. */
export function inform({ title, content }) {
  const dlg = h("dialog", { class: "dlg", "aria-labelledby": "dlg-title" });
  const done = () => { dlg.close(); dlg.remove(); };
  append(dlg, [
    h("h2", { class: "dlgt", id: "dlg-title" }, title),
    content,
    h("div", { class: "dlgb" }, h("button", { class: "btn fill", onclick: done }, "Close"))]);
  dlg.addEventListener("cancel", (e) => { e.preventDefault(); done(); });
  dlg.addEventListener("click", (e) => { if (e.target === dlg) done(); });
  document.body.append(dlg);
  dlg.showModal();
  dlg.querySelector(".btn.fill").focus();
}

let toastTimer = null;
export function toast(text) {
  document.querySelector(".toast")?.remove();
  const t = h("div", { class: "toast", role: "status" }, text);
  document.body.append(t);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.remove(), 3500);
}

export function ago(seconds) {
  const d = Date.now() / 1000 - seconds;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.round(d / 60)} min ago`;
  if (d < 86400) { const n = Math.round(d / 3600); return `${n} hour${n === 1 ? "" : "s"} ago`; }
  const n = Math.round(d / 86400);
  return `${n} day${n === 1 ? "" : "s"} ago`;
}

export function clock(seconds) {
  const d = new Date(seconds * 1000);
  const today = new Date();
  const time = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  if (d.toDateString() === today.toDateString()) return time;
  return `${d.toLocaleDateString([], { month: "short", day: "numeric" })}, ${time}`;
}

export function secs(s) {
  return s === null || s === undefined ? "" : `${Number(s).toFixed(1)} s`;
}

/** What ended a session, in a few words. */
/** live, gated or follow, as the pages name it. */
export function modeLabel(mode) {
  return mode === "gated" ? "Gated" : mode === "follow" ? "Follow" : "Live";
}

/** Who a session's kiosk answers, from its policy and chosen people. */
export function answersLabel(record, people = []) {
  if (record?.policy === "enrolled") return "Enrolled voices only";
  if (record?.policy === "people") {
    const byId = Object.fromEntries(people.map((p) => [p.id, p.name]));
    const names = (record.people || []).map((id) => byId[id]).filter(Boolean);
    const missing = missingPeople(record, people);
    const gone = missing ? `${missing === 1 ? "1 person" : `${missing} people`} not on this Ground Control` : "";
    if (names.length) return `Only ${listOf(names)}${gone ? `, and ${gone}` : ""}`;
    return gone || "Chosen people only";
  }
  return "Anyone";
}

/** How many of the people a kiosk answers aren't here, as after moving to another Ground Control. */
export function missingPeople(record, people = []) {
  if (record?.policy !== "people") return 0;
  const ids = new Set(people.map((p) => p.id));
  return (record.people || []).filter((id) => !ids.has(id)).length;
}

/** A kiosk's state: waiting for approval and declined come before being connected. */
export function kioskChip(k) {
  if (k.access === "pending") return chip("Waiting for approval", "c-warn");
  if (k.access === "declined") return chip("Declined", "c-err");
  if (k.in_session) return chip("In a session", "c-info");
  if (k.connected) return chip("Connected", "c-ok");
  return chip("Offline", "c-off");
}

/** Who a kiosk answers: what it says it has on, or else its last session. */
export function kioskAnswers(k, lastSession, people = []) {
  const f = k.features || {};
  if (f.policy) return answersLabel({ policy: f.policy, people: f.people }, people);
  return lastSession?.policy ? answersLabel(lastSession, people) : null;
}

/** "Ana", "Ana and Leo", "Ana, Leo and Mia". */
export function listOf(names) {
  return names.length < 2 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

export function outcome(stats) {
  const name = stats?.speaker?.name;
  if (stats?.mode === "follow" && stats?.end === "other_voice") {
    return `${stats.dropped_s ?? 0} s of other voices left out`;
  }
  switch (stats?.end) {
    case "silence": return name ? `Ended when ${name} stopped talking` : "Ended when the speaker stopped talking";
    case "other_voice": return "Ended when another voice started";
    case "rejected": return "Rejected because this kiosk doesn't answer this voice";
    case "false_positive": return "False positive: the wake word wasn't said";
    case "max_length": return "Stopped at the 30 s limit";
    default: return "Ended by the kiosk";
  }
}

export function outcomeChip(stats) {
  if (stats?.mode === "follow" && stats?.end === "other_voice") return chip("Voices left out", "c-info");
  switch (stats?.end) {
    case "other_voice": return chip("Other voice cut", "c-info");
    case "rejected": return chip("Rejected", "c-err");
    case "false_positive": return chip("False positive", "c-warn");
    case "silence": return stats?.speaker?.name || stats?.speaker?.by === "answer" ? chip("Speaker finished", "c-ok") : chip("Unknown voice", "c-warn");
    default: return chip("Ended", "c-off");
  }
}
