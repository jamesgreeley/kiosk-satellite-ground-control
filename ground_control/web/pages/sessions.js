import { api } from "../api.js";
import { shell } from "../app.js";
import { chip, clock, disc, h, icon, listOf, modeLabel, outcomeChip, pageHead, secs } from "../ui.js";

const PAGE = 25;

/** A vision session's outcome, as one chip: what the kiosk did, else what was seen. */
export function visionSummary(s) {
  const words = s.summary || [];
  if (!words.length) return chip("Nothing kept", "c-off");
  // A gesture's action says the most, then the screensaver, then what was seen.
  const top = words.find((w) => w.includes("→")) || words[0];
  const did = /dismissed|postponed|→/.test(top);
  const c = chip(top, `trunc ${did ? "c-ok" : "c-info"}`);
  c.title = words.join(", ");
  if (words.length < 2) return c;
  return h("span", { class: "cell", title: words.join(", ") }, c, h("span", { class: "sub" }, `+${words.length - 1}`));
}

export async function render() {
  const [{ people }, settings] = await Promise.all([api("api/people"), api("api/settings")]);
  const byId = Object.fromEntries(people.map((p) => [p.id, p]));
  const state = { kind: "", show: "all", kiosk: "", person: "", q: "", page: 0 };

  const body = h("tbody");
  const count = h("span");
  const prev = h("button", { class: "ib", "aria-label": "Previous page" }, icon("prev"));
  const next = h("button", { class: "ib", "aria-label": "Next page" }, icon("next"));
  const kioskSel = h("select", { class: "fld", "aria-label": "Kiosk" }, h("option", { value: "" }, "All kiosks"));
  let asked = 0;

  function voiceRow(s) {
    const p = s.speaker?.id ? byId[s.speaker.id] || s.speaker : null;
    return [
      h("td", {}, h("span", { class: "cell" }, disc(p, "sm"),
        h("span", {}, p?.name || "Unknown", " ", h("span", { class: "sub" }, s.speaker?.score ?? "")))),
      h("td", {}, h("span", { class: "cell" }, icon("mic", 18), modeLabel(s.mode))),
      h("td", {}, outcomeChip(s)),
      h("td", {}, secs(s.audio_s)),
    ];
  }

  function visionRow(s) {
    return [
      h("td", {}, h("span", { class: "sub" }, `${s.looks ?? 0} look${s.looks === 1 ? "" : "s"}`)),
      h("td", {}, h("span", { class: "cell" }, icon("eye", 18), "Vision")),
      h("td", {}, visionSummary(s)),
      h("td", {}, secs(s.duration_s)),
    ];
  }

  async function draw() {
    const mine = ++asked;
    const q = new URLSearchParams({ limit: PAGE, offset: state.page * PAGE });
    if (state.kind) q.set("kind", state.kind);
    if (state.kiosk) q.set("device", state.kiosk);
    if (state.kind !== "vision") {
      if (state.show !== "all") q.set("show", state.show);
      if (state.person) q.set("person", state.person);
    }
    if (state.q) q.set("q", state.q);
    const page = await api(`api/sessions?${q}`);
    if (mine !== asked) return; // a newer filter is on its way
    const pages = Math.max(1, Math.ceil(page.total / PAGE));
    if (state.page > pages - 1) { state.page = pages - 1; draw(); return; }
    const chosen = kioskSel.value;
    kioskSel.replaceChildren(h("option", { value: "" }, "All kiosks"), ...page.kiosks.map((k) => h("option", { value: k }, k)));
    kioskSel.value = chosen;
    body.replaceChildren(...(page.sessions.length ? page.sessions.map((s) => {
      const tr = h("tr", { tabindex: "0" },
        h("td", {}, clock(s.started)),
        h("td", {}, s.device_name || s.device_id || "Unknown kiosk"),
        ...(s.kind === "vision" ? visionRow(s) : voiceRow(s)));
      const open = () => { location.hash = `#/sessions/${s.id}`; };
      tr.addEventListener("click", open);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
      return tr;
    }) : [h("tr", {}, h("td", { colspan: "6", class: "empty" }, "No sessions match."))]));
    const from = page.total ? state.page * PAGE + 1 : 0;
    count.textContent = `${from} to ${Math.min(page.total, (state.page + 1) * PAGE)} of ${page.total}`;
    prev.disabled = state.page === 0;
    next.disabled = state.page >= pages - 1;
  }

  const redraw = () => { state.page = 0; draw(); };
  prev.addEventListener("click", () => { state.page -= 1; draw(); });
  next.addEventListener("click", () => { state.page += 1; draw(); });

  const typeSel = h("select", { class: "fld", "aria-label": "Type" }, h("option", { value: "" }, "All types"),
    h("option", { value: "voice" }, "Voice Match"), h("option", { value: "vision" }, "Vision Match"));
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Show" },
    [["all", "All"], ["cut", "Cut short"], ["rej", "Rejected"]].map(([k, label]) => h("button", {
      class: k === state.show ? "on" : "",
      onclick: (e) => {
        state.show = k;
        seg.querySelectorAll("button").forEach((b) => b.classList.remove("on"));
        e.currentTarget.classList.add("on");
        redraw();
      },
    }, label)));
  const search = h("input", { type: "search", placeholder: "Search", "aria-label": "Search kiosks and people" });
  let typing = null;
  search.addEventListener("input", () => {
    clearTimeout(typing);
    typing = setTimeout(() => { state.q = search.value.trim(); redraw(); }, 250);
  });
  kioskSel.addEventListener("change", () => { state.kiosk = kioskSel.value; redraw(); });
  const personSel = h("select", { class: "fld", "aria-label": "Person" }, h("option", { value: "" }, "Everyone"),
    people.map((p) => h("option", { value: p.id }, p.name)), h("option", { value: "unknown" }, "Unknown voices"));
  personSel.addEventListener("change", () => { state.person = personSel.value; redraw(); });
  typeSel.addEventListener("change", () => {
    state.kind = typeSel.value;
    // Cut short, rejected and the speaker are about voice sessions.
    const voiceOnly = state.kind === "vision";
    seg.hidden = voiceOnly;
    personSel.hidden = voiceOnly;
    redraw();
  });
  await draw();

  const kept = listOf(["Sessions", settings.keep_audio ? "their audio" : null, settings.keep_frames ? "their frames" : null].filter(Boolean));
  return {
    nodes: [
      shell("sessions"),
      h("main", { class: "page" },
        pageHead("Sessions"),
        h("div", { class: "filters" }, typeSel, seg, h("label", { class: "search" }, icon("search", 20), search), kioskSel, personSel),
        h("div", { class: "card" },
          h("div", { class: "scroll" },
            h("table", {},
              h("thead", {}, h("tr", {}, ["Time", "Kiosk", "Who", "Type", "Outcome", "Length"].map((t) => h("th", {}, t)))),
              body)),
          h("div", { class: "tfoot" }, count, h("span", { style: { display: "flex", gap: "2px" } }, prev, next))),
        h("p", { class: "note" }, `${kept} are kept for ${settings.keep_days} days. Change it in Settings.`)),
    ],
  };
}
