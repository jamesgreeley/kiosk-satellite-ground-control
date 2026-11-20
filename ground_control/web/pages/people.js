import { api } from "../api.js";
import { shell } from "../app.js";
import { ago, ask, chip, disc, h, icon, pageHead, refreshPicture, section, toast } from "../ui.js";

const COLORS = [["#56814F", "Green"], ["#7A5BA6", "Violet"], ["#8A6516", "Ochre"], ["#3D64A8", "Blue"],
  ["#3F7F86", "Teal"], ["#9A4A86", "Magenta"]];
const THINGS = [
  ["todo_list", "To-do list", 'Add to it and read it with "my list"', "todo.", "list"],
  ["calendar", "Calendar", '"What\'s on my calendar"', "calendar.", "calendar"],
  ["playlist", "Playlist", '"Play my playlist" in Music Assistant', "", "music"],
  ["person", "Person", "This person in Home Assistant", "person.", "person"],
];
const TARGET_S = 45;

export async function render({ id }) {
  const [{ people }, { kiosks }, suggestions] = await Promise.all([api("api/people"), api("api/kiosks"),
    api("api/suggestions").catch(() => ({}))]);
  const person = people.find((p) => p.id === id) || people[0];
  const kinds = [...new Set(kiosks.map((k) => k.model).filter(Boolean))];
  const coverage = (p) => {
    if (!p.has_voice) return "No voice enrolled yet";
    if (!kinds.length) return `${p.speech_s} s of speech`;
    const n = kinds.filter((m) => (p.models || []).includes(m)).length;
    return n === kinds.length ? `Enrolled on every kind of device` : `Enrolled on ${n} of ${kinds.length} kinds of device`;
  };

  const list = people.length
    ? people.map((p) => h("a", { class: `row ${p === person ? "sel" : ""}`, href: `#/people/${p.id}` },
        disc(p),
        h("div", { class: "grow" },
          h("div", { class: "rt" }, p.name),
          h("div", { class: "rd" }, coverage(p)))))
    : [h("div", { class: "empty" }, "Nobody here yet. Add a person, then enroll their voice on a kiosk.")];

  const add = async () => {
    const name = await ask({ title: "Add a person", field: { label: "Name", placeholder: "Name" }, ok: "Add" });
    if (!name) return;
    try {
      const p = await api("api/people", { method: "POST", json: { name } });
      location.hash = `#/people/${p.id}`;
    } catch (err) { toast(err.message); }
  };

  return {
    nodes: [
      shell("people"),
      h("main", { class: "page" },
        pageHead("People"),
        h("div", { class: "split" },
          h("section", { class: "left", "aria-label": "Everyone" },
            h("div", { class: "card" }, list,
              h("button", { class: "row nowrap add", onclick: add },
                h("span", { class: "ico" }, icon("plus", 24)),
                h("div", { class: "grow" }, h("div", { class: "rt" }, "Add a person"))),
              people.length ? h("div", { class: "hint", style: { paddingTop: "14px" } }, icon("info", 18),
                "Enroll each person once on every kind of device they talk to.") : null)),
          person ? detail(person, kiosks, suggestions) : null)),
    ],
  };
}

/** The kinds of device in the house, and whether this person is enrolled on each. */
function devices(p, kiosks) {
  const byModel = new Map();
  for (const k of kiosks) {
    const model = k.model || k.name;
    if (!byModel.has(model)) byModel.set(model, []);
    byModel.get(model).push(k);
  }
  if (!byModel.size) return [h("div", { class: "empty" }, "No kiosks yet.")];
  const covered = new Set(p.models || []);
  return [...byModel.entries()].sort().map(([model, ks]) => {
    const enrolled = covered.has(model);
    const seconds = (p.enrollments || []).filter((e) => e.model === model).reduce((a, e) => a + e.seconds, 0);
    const target = ks.find((k) => k.connected && k.enroll);
    const off = !target && ks.some((k) => k.connected);
    return h("div", { class: "row" },
      h("span", { class: "ico" }, icon("screen", 24)),
      h("div", { class: "grow" },
        h("div", { class: "rt" }, model),
        h("div", { class: "rd" }, `${ks.map((k) => k.name).join(", ")}${enrolled ? `. ${Math.round(seconds)} s enrolled.` : ""}`)),
      enrolled ? chip("Enrolled", "c-ok")
        : target ? h("a", { class: "btn out sm", href: `#/enroll/${p.id}/${encodeURIComponent(target.device_id)}`,
          "aria-label": `Enroll ${p.name} on the ${target.name}` }, "Enroll")
          : chip(off ? "Not enrolled, enrollment off" : "Not enrolled, offline", "c-off"));
  });
}

function enrollmentRows(p) {
  if (!p.enrollments?.length) return [h("div", { class: "empty" }, "No enrollments yet.")];
  return p.enrollments.map((e) => {
    const where = e.device_name ? `${e.device_name}${e.model ? ` (${e.model})` : ""}` : "Unknown device";
    const remove = async () => {
      if (!await ask({ title: "Delete this enrollment?", text: `${where}, ${e.seconds} s. ${p.name}'s voice is rebuilt without it.`,
        ok: "Delete", danger: true })) return;
      try {
        await api(`api/people/${p.id}/enrollments/${e.id}`, { method: "DELETE" });
        location.reload();
      } catch (err) { toast(err.message); }
    };
    return h("div", { class: "row" },
      h("span", { class: "ico" }, icon("mic", 24)),
      h("div", { class: "grow" }, h("div", { class: "rt" }, where), h("div", { class: "rd" }, `${e.seconds} s of speech. ${ago(e.at)}.`)),
      h("button", { class: "ib", "aria-label": `Delete the ${where} enrollment`, onclick: remove }, icon("trash")));
  });
}

function detail(p, kiosks, suggestions = {}) {
  const save = async (body, done = "Saved") => {
    try { await api(`api/people/${p.id}`, { method: "PATCH", json: body }); toast(done); }
    catch (err) { toast(err.message); }
  };
  const rename = async () => {
    const name = await ask({ title: "Rename", field: { label: "Name", value: p.name }, ok: "Save" });
    if (name && name !== p.name) {
      await save({ name }, "Renamed");
      location.reload();
    }
  };
  const forget = async () => {
    if (!await ask({ title: `Forget ${p.name}?`, text: "Their voice and all their enrollment audio are deleted.",
      ok: "Forget", danger: true })) return;
    await api(`api/people/${p.id}`, { method: "DELETE" });
    location.hash = "#/people";
  };
  const swatches = h("div", { class: "swatches", role: "radiogroup", "aria-label": "Color on the kiosk" },
    COLORS.map(([c, label]) => h("button", {
      class: "swatch", style: { background: c }, role: "radio", "aria-checked": String(c === p.color),
      "aria-label": label,
      onclick: async (e) => {
        swatches.querySelectorAll(".swatch").forEach((s) => s.setAttribute("aria-checked", "false"));
        e.currentTarget.setAttribute("aria-checked", "true");
        await save({ color: c }, "Color saved");
      },
    })));
  const things = THINGS.map(([key, label, desc, prefix, ic]) => {
    // Home Assistant's matching entities (Music Assistant's playlists), offered as it is typed in.
    const options = suggestions?.[key] || [];
    const list = options.length ? h("datalist", { id: `s-${key}` },
      options.map((o) => h("option", { value: o.value, label: o.label || null }))) : null;
    const input = h("input", { class: "fld wide", id: `t-${key}`, value: p.things?.[key] || "", autocomplete: "off",
      list: list ? `s-${key}` : null,
      placeholder: prefix ? `${prefix}${p.name.toLowerCase().replace(/[^a-z0-9]+/g, "_")}` : "Playlist name" });
    input.addEventListener("change", async () => {
      const value = input.value.trim() || null;
      // A to-do list, calendar or person is a Home Assistant entity: half-typed text isn't saved.
      if (value && prefix && !new RegExp(`^${prefix.replace(".", "\\.")}[a-z0-9_]+$`).test(value)) {
        input.setAttribute("aria-invalid", "true");
        toast(`Not saved. Pick one from the list, or enter an entity id like ${prefix}${p.name.toLowerCase().replace(/[^a-z0-9]+/g, "_")}.`);
        return;
      }
      input.removeAttribute("aria-invalid");
      await save({ things: { [key]: value } });
      p.things = { ...p.things, [key]: value };
      if (key === "person") refreshPicture({ ...p, picture: !!value });
    });
    return h("div", { class: "row" },
      h("span", { class: "ico" }, icon(ic, 24)),
      h("div", { class: "grow" }, h("div", { class: "rt" }, h("label", { for: `t-${key}` }, label)), h("div", { class: "rd" }, desc)),
      input, list);
  });
  const pct = Math.min(100, Math.round((p.speech_s / TARGET_S) * 100));
  return h("section", { class: "right", "aria-label": p.name },
    h("div", { class: "card" },
      h("div", { class: "head" },
        disc(p, "big"),
        h("div", { class: "grow" }, h("div", { class: "hname" }, p.name),
          h("div", { class: "rd" }, p.has_voice ? `${p.clips} recordings` : "No voice enrolled yet")),
        h("button", { class: "ib", "aria-label": `Rename ${p.name}`, onclick: rename }, icon("edit"))),
      swatches,
      h("div", { class: "hint" }, icon("info", 18), `The Echo Dot ring and the Echo Show screen take this color while ${p.name} is talking.`)),
    section("Devices", `${p.name}'s voice is recognized best on the kinds of device it was enrolled on. Enroll once on each kind.`),
    h("div", { class: "card" }, devices(p, kiosks)),
    section("Voice Match"),
    h("div", { class: "card" },
      h("div", { class: "row" },
        h("div", { class: "grow" }, h("div", { class: "rt" }, "Speech enrolled"), h("div", { class: "rd" }, "45 s on each kind of device. Confident commands add more over time.")),
        h("div", { class: "meter", role: "img", "aria-label": `${p.speech_s} s of speech` }, h("span", { style: { width: `${pct}%` } })),
        h("span", { class: "val" }, `${p.speech_s} s`)),
      enrollmentRows(p)),
    section("Vision Match"),
    h("div", { class: "card" },
      h("div", { class: "row" },
        h("span", { class: "ico" }, icon("eye", 24)),
        h("div", { class: "grow" }, h("div", { class: "rt" }, "Coming soon")))),
    section(`What's ${p.name}'s`, `What "my" means when ${p.name} talks. "Add milk to my list" goes on this list on every kiosk.`),
    h("div", { class: "card" }, things,
      h("div", { class: "hint" }, icon("info", 18), 'Kiosk Satellite uses these when this person says "my", with its personal commands and to-do blueprints. Without a list of their own, "my list" is the shared list.')),
    h("div", { class: "card", style: { marginTop: "24px" } },
      h("div", { class: "actions", style: { paddingTop: "20px" } },
        h("a", { class: "btn out", href: `#/enroll/${p.id}` }, icon("plus", 20), "Enroll on a kiosk"),
        h("a", { class: "btn out", href: "#/sessions" }, "Add from a session"),
        h("button", { class: "btn danger", onclick: forget }, `Forget ${p.name}`))));
}
