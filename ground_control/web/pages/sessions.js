import { api } from "../api.js";
import { shell } from "../app.js";
import { clock, disc, h, icon, modeLabel, outcomeChip, pageHead, secs } from "../ui.js";

const PAGE = 25;

export async function render() {
  const [{ sessions }, { people }, settings] = await Promise.all([
    api("api/sessions?limit=500"), api("api/people"), api("api/settings"),
  ]);
  const byId = Object.fromEntries(people.map((p) => [p.id, p]));
  const kiosks = [...new Set(sessions.map((s) => s.device_name || s.device_id).filter(Boolean))].sort();
  const state = { show: "all", kiosk: "", person: "", q: "", page: 0 };

  const body = h("tbody");
  const count = h("span");
  const prev = h("button", { class: "ib", "aria-label": "Previous page" }, icon("prev"));
  const next = h("button", { class: "ib", "aria-label": "Next page" }, icon("next"));

  function rows() {
    return sessions.filter((s) => {
      if (state.show === "cut" && s.end !== "other_voice") return false;
      if (state.show === "rej" && s.end !== "rejected") return false;
      const where = s.device_name || s.device_id || "";
      if (state.kiosk && where !== state.kiosk) return false;
      if (state.person === "unknown" && s.speaker?.id) return false;
      if (state.person && state.person !== "unknown" && s.speaker?.id !== state.person) return false;
      if (state.q) {
        const hay = `${where} ${s.speaker?.name || "unknown"}`.toLowerCase();
        if (!hay.includes(state.q.toLowerCase())) return false;
      }
      return true;
    });
  }

  function draw() {
    const all = rows();
    const pages = Math.max(1, Math.ceil(all.length / PAGE));
    state.page = Math.min(state.page, pages - 1);
    const shown = all.slice(state.page * PAGE, (state.page + 1) * PAGE);
    body.replaceChildren(...(shown.length ? shown.map((s) => {
      const p = s.speaker?.id ? byId[s.speaker.id] || s.speaker : null;
      const tr = h("tr", { tabindex: "0" },
        h("td", {}, clock(s.started)),
        h("td", {}, s.device_name || s.device_id || "Unknown kiosk"),
        h("td", {}, h("span", { class: "cell" }, disc(p, "sm"),
          h("span", {}, p?.name || "Unknown", " ", h("span", { class: "sub" }, s.speaker?.score ?? "")))),
        h("td", {}, modeLabel(s.mode)),
        h("td", {}, outcomeChip(s)),
        h("td", {}, secs(s.audio_s)));
      const open = () => { location.hash = `#/sessions/${s.id}`; };
      tr.addEventListener("click", open);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
      return tr;
    }) : [h("tr", {}, h("td", { colspan: "6", class: "empty" }, "No sessions match."))]));
    const from = all.length ? state.page * PAGE + 1 : 0;
    count.textContent = `${from} to ${Math.min(all.length, (state.page + 1) * PAGE)} of ${all.length}`;
    prev.disabled = state.page === 0;
    next.disabled = state.page >= pages - 1;
  }

  prev.addEventListener("click", () => { state.page -= 1; draw(); });
  next.addEventListener("click", () => { state.page += 1; draw(); });
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Show" },
    [["all", "All"], ["cut", "Cut short"], ["rej", "Rejected"]].map(([k, label]) => h("button", {
      class: k === state.show ? "on" : "",
      onclick: (e) => {
        state.show = k; state.page = 0;
        seg.querySelectorAll("button").forEach((b) => b.classList.remove("on"));
        e.currentTarget.classList.add("on");
        draw();
      },
    }, label)));
  const search = h("input", { type: "search", placeholder: "Search kiosks and people", "aria-label": "Search kiosks and people" });
  search.addEventListener("input", () => { state.q = search.value; state.page = 0; draw(); });
  const kioskSel = h("select", { class: "fld", "aria-label": "Kiosk" }, h("option", { value: "" }, "All kiosks"),
    kiosks.map((k) => h("option", { value: k }, k)));
  kioskSel.addEventListener("change", () => { state.kiosk = kioskSel.value; state.page = 0; draw(); });
  const personSel = h("select", { class: "fld", "aria-label": "Person" }, h("option", { value: "" }, "Everyone"),
    people.map((p) => h("option", { value: p.id }, p.name)), h("option", { value: "unknown" }, "Unknown voices"));
  personSel.addEventListener("change", () => { state.person = personSel.value; state.page = 0; draw(); });
  draw();

  return {
    nodes: [
      shell("sessions"),
      h("main", { class: "page" },
        pageHead("Sessions"),
        h("div", { class: "filters" }, seg, h("label", { class: "search" }, icon("search", 20), search), kioskSel, personSel),
        h("div", { class: "card" },
          h("div", { class: "scroll" },
            h("table", {},
              h("thead", {}, h("tr", {}, ["Time", "Kiosk", "Speaker", "Mode", "Outcome", "Length"].map((t) => h("th", {}, t)))),
              body)),
          h("div", { class: "tfoot" }, count, h("span", { style: { display: "flex", gap: "2px" } }, prev, next))),
        h("p", { class: "note" }, `Sessions${settings.keep_audio ? " and their audio" : ""} are kept for ${settings.keep_days} days. Change it in Settings.`)),
    ],
  };
}
