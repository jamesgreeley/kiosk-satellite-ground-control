import { api, audioUrl, socket } from "../api.js";
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

// Face Recognition's guided capture, as Ground Control reports it.
const FACE_STEPS = { look: "Look at the screen", side_a: "Turn your head a little to one side", side_b: "Now a little to the other side",
  far: "Take a step back" };
const FACE_HINTS = { ok: "", no_face: "No face in view.", several_faces: "Only one person in view, please.",
  closer: "Come a little closer.", farther: "Step back a little more.", look_at_screen: "Look at the screen.",
  turn_more: "Turn a little more.", turn_less: "Turn back a little.", other_side: "Now to the other side.",
  too_dark: "Too dark to see the face.", other_face: "That's a different face." };

/** The device models in the house: kiosks by model, a kiosk without one on its own. */
function deviceKinds(kiosks) {
  const byModel = new Map();
  for (const k of kiosks) {
    const model = k.model || k.name;
    if (!byModel.has(model)) byModel.set(model, []);
    byModel.get(model).push(k);
  }
  return [...byModel.entries()].sort(([a], [b]) => a.localeCompare(b));
}

/** How far a person's voice reaches: device models enrolled out of those in the house. */
function voiceCoverage(p, kinds) {
  // A kiosk that doesn't say its model can't be matched to a recording, so it isn't counted.
  const known = kinds.filter(([, ks]) => ks.some((k) => k.model));
  if (!p.has_voice) return { text: "No voice", n: 0, of: known.length, ok: false };
  const covered = new Set(p.models || []);
  const n = known.filter(([model]) => covered.has(model)).length;
  return { text: known.length ? `${n} of ${known.length}` : "Voice", n, of: known.length, ok: true };
}

/** A card's head: the feature's icon, its name, what it is for and where this person stands. */
function cardHead(id, ic, title, about, status) {
  return h("div", { class: "fhead" },
    h("span", { class: "ftile" }, icon(ic, 22)),
    h("div", { class: "grow" }, h("h2", { class: "ftitle", id }, title), h("p", { class: "fabout" }, about)),
    status);
}

/** A list that opens and closes under a card, with its count. */
function disclose(title, count, aside, rows) {
  return h("details", { class: "fmore" },
    h("summary", {},
      h("span", { class: "grow" }, title, " ", h("span", { class: "fcount" }, String(count))),
      aside ? h("span", { class: "faside" }, aside) : null,
      icon("down", 20)),
    h("div", { class: "fmorebody" }, rows));
}

function deleteButton(label, onclick) {
  return h("button", { class: "ib", "aria-label": label, onclick }, icon("trash", 20));
}

// ── Voice Match ─────────────────────────────────────────────────────────

function voiceCard(p, kiosks, kinds) {
  const cover = voiceCoverage(p, kinds);
  const status = !p.has_voice ? chip("Not enrolled", "c-off big")
    : !cover.of || cover.n === cover.of ? chip("Enrolled", "c-ok big", icon("check", 14))
      : chip(`${cover.n} of ${cover.of} models`, "c-warn big");
  const recordings = p.enrollments || [];
  const speech = p.has_voice ? ` ${Math.round(p.speech_s)} s of speech in ${recordings.length} ${recordings.length === 1 ? "recording" : "recordings"}.` : "";

  // Enroll: a connected kiosk that allows it records the voice through its own microphone.
  const ready = kiosks.filter((k) => k.connected && k.enroll);
  const covered = new Set(p.models || []);
  const sel = h("select", { class: "fld", id: "voice-kiosk" },
    [...ready].sort((a, b) => Number(covered.has(a.model)) - Number(covered.has(b.model)))
      .map((k) => h("option", { value: k.device_id }, k.model && k.model !== k.name ? `${k.name} (${k.model})` : k.name)));
  const enroll = h("button", { class: "btn fill sm", disabled: !ready.length,
    onclick: () => { location.hash = `#/enroll/${p.id}/${encodeURIComponent(sel.value)}`; } }, "Enroll voice");
  const bar = h("div", { class: "fbar" },
    ready.length ? [h("label", { for: "voice-kiosk" }, "Enroll on"), sel] : null,
    enroll,
    h("span", { class: "fnote" }, ready.length
      ? `${p.name} reads a few sentences aloud on the kiosk, about 45 seconds of speech.`
      : "Needs a connected kiosk with Allow enrollment here on."));

  const tiles = kinds.map(([model, ks]) => {
    const enrolled = covered.has(model);
    const seconds = recordings.filter((e) => e.model === model).reduce((a, e) => a + e.seconds, 0);
    const target = ks.find((k) => k.connected && k.enroll);
    const on = ks.some((k) => k.connected);
    return h("div", { class: `kind ${enrolled ? "on" : ""}` },
      icon("screen", 22),
      h("div", { class: "grow" }, h("div", { class: "kt" }, model), h("div", { class: "kd" }, ks.map((k) => k.name).join(", "))),
      enrolled ? chip(`${Math.round(seconds)} s`, "c-ok", icon("check", 14))
        : target ? h("a", { class: "btn out sm", href: `#/enroll/${p.id}/${encodeURIComponent(target.device_id)}`,
          "aria-label": `Enroll ${p.name} on the ${target.name}` }, "Enroll")
          : chip(on ? "Enrollment off" : "Offline", "c-off"));
  });

  const rows = recordings.map((e) => {
    const where = e.device_name ? `${e.device_name}${e.model ? `, ${e.model}` : ""}` : "Uploaded file";
    const remove = async () => {
      if (!await ask({ title: "Delete this recording?", text: `${where}, ${e.seconds} s. ${p.name}'s voice is rebuilt without it.`,
        ok: "Delete", danger: true })) return;
      try {
        await api(`api/people/${p.id}/enrollments/${e.id}`, { method: "DELETE" });
        location.reload();
      } catch (err) { toast(err.message); }
    };
    return h("div", { class: "frow" },
      h("div", { class: "grow" }, h("div", { class: "ft" }, where), h("div", { class: "rd" }, `${e.seconds} s of speech. ${ago(e.at)}.`)),
      deleteButton(`Delete the ${where} recording`, remove));
  });

  return h("section", { class: "card fcard", "aria-labelledby": "voice-title" },
    cardHead("voice-title", "mic", "Voice Match", `Who is talking, for the Speaker sensor and personal commands.${speech}`, status),
    bar,
    kinds.length ? h("div", { class: "fpart" },
      h("h3", {}, "Device models"),
      h("p", {}, "A voice is recognized best on the device models it was enrolled on. Enroll once on each model."),
      h("div", { class: "kinds" }, tiles)) : null,
    recordings.length ? disclose("Recordings", recordings.length, "Confident commands add more over time", rows) : null);
}

// ── Vision Match ────────────────────────────────────────────────────────

/** The enrolled face pictures, small, from the newest capture. */
function faceThumbs(p) {
  const e = (p.face_enrollments || []).at(-1);
  if (!e) return null;
  return h("div", { class: "facethumbs", role: "list", "aria-label": "Face pictures from the latest capture" },
    (e.crop_names || []).map((name) => {
      const img = h("img", { alt: "", width: 56, height: 56, role: "listitem" });
      audioUrl(`api/people/${p.id}/faces/${e.id}/${name}`).then((url) => { img.src = url; }).catch(() => img.remove());
      return img;
    }));
}

/** The capture's live progress: the kiosk, the step, how far and what to fix. */
function captureView() {
  const where = h("div", { class: "cwhere" });
  const step = h("div", { class: "cstep" });
  const segs = h("div", { class: "csegs", "aria-hidden": "true" });
  const have = h("span", {});
  const hint = h("span", { class: "chip c-warn", hidden: true });
  const node = h("div", { class: "capture", role: "status", "aria-live": "polite", hidden: true },
    where, step, segs, h("div", { class: "cline" }, have, hint),
    h("div", { class: "fnote" }, "To stop it early, use Stop on the kiosk. The pictures from a finished first step are kept."));
  const show = (e, kiosk) => {
    node.hidden = false;
    const steps = e.steps || 4;
    const i = Math.min(e.step_index ?? 0, steps - 1);
    where.textContent = `On ${kiosk}. Step ${i + 1} of ${steps}.`;
    step.textContent = FACE_STEPS[e.step] || "Starting";
    const part = e.need ? Math.min(1, (e.have || 0) / e.need) : 0;
    segs.replaceChildren(...Array.from({ length: steps }, (_, n) => {
      const fill = n < i ? 100 : n === i ? Math.round(part * 100) : 0;
      return h("span", {}, h("i", { style: { width: `${fill}%` } }));
    }));
    have.textContent = e.need ? `${e.have || 0} of ${e.need} pictures` : "";
    const text = FACE_HINTS[e.hint] || "";
    hint.hidden = !text;
    hint.textContent = text;
  };
  const starting = (kiosk) => {
    node.hidden = false;
    where.textContent = `On ${kiosk}.`;
    step.textContent = "Waiting for the kiosk to start";
    segs.replaceChildren();
    have.textContent = "";
    hint.hidden = true;
  };
  return { node, show, starting, hide: () => { node.hidden = true; } };
}

function visionCard(p, kiosks, events) {
  const captures = p.face_enrollments || [];
  const pictures = captures.reduce((n, e) => n + e.crops, 0);
  const restStatus = () => (p.has_face ? chip("Enrolled", "c-ok big", icon("check", 14)) : chip("Not enrolled", "c-off big"));
  const statusSlot = h("span", { class: "fstatus" }, restStatus());
  const about = `Who is looking at the kiosk, for the Face sensor on kiosks with Face Recognition on.${p.has_face
    ? ` ${pictures} face pictures from ${captures.length} ${captures.length === 1 ? "capture" : "captures"}.` : ""}`;

  const ready = kiosks.filter((k) => k.connected && k.access === "approved" && k.features?.vision_match
    && k.features?.face_recognition);
  const name = (id) => kiosks.find((k) => k.device_id === id)?.name || "a kiosk";
  const sel = h("select", { class: "fld", id: "face-kiosk" }, ready.map((k) => h("option", { value: k.device_id }, k.name)));
  const go = h("button", { class: "btn fill sm", disabled: !ready.length }, "Enroll face");
  const bar = h("div", { class: "fbar" },
    ready.length ? [h("label", { for: "face-kiosk" }, "Enroll on"), sel] : null,
    go,
    h("span", { class: "fnote" }, ready.length
      ? `${p.name} stands in front of the kiosk and follows four steps on its screen. It takes about half a minute.${p.has_face ? " Another capture in different light helps." : ""}`
      : "Needs a connected kiosk with Face Recognition on."));
  const live = captureView();

  const running = (on) => {
    bar.hidden = on;
    statusSlot.replaceChildren(on ? chip("Capturing", "c-info big") : restStatus());
    if (!on) live.hide();
  };
  go.addEventListener("click", async () => {
    try {
      await api(`api/kiosks/${encodeURIComponent(sel.value)}/enroll`, { method: "POST", json: { person_id: p.id, kind: "face" } });
    } catch (err) { toast(err.message); return; }
    running(true);
    live.starting(name(sel.value));
  });
  // A capture for this person, started here, on the kiosk or from another browser.
  events.addEventListener("message", (m) => {
    let e;
    try { e = JSON.parse(m.data); } catch { return; }
    if (e.person_id !== p.id || !e.type?.startsWith("face_enroll_")) return;
    if (e.type === "face_enroll_start") {
      running(true);
      live.show({ step: "look", step_index: 0, steps: e.steps?.length || 4, have: 0, need: e.steps?.[0]?.need || 0, hint: "ok" },
        name(e.device_id));
    } else if (e.type === "face_enroll_progress") {
      running(true);
      live.show(e, name(e.device_id));
    } else if (e.type === "face_enroll_saved") {
      toast(`Saved ${e.crops} face pictures.`);
      setTimeout(() => location.reload(), 800);
    } else if (e.type === "face_enroll_error") {
      running(false);
      toast(e.message || "Nothing was saved.");
    }
  });

  const rows = captures.map((e) => {
    const where = e.source?.device_name || "Unknown kiosk";
    const remove = async () => {
      if (!await ask({ title: "Delete this capture?", text: `${where}, ${e.crops} pictures. ${p.name}'s face is rebuilt without them.`,
        ok: "Delete", danger: true })) return;
      try {
        await api(`api/people/${p.id}/faces/${e.id}`, { method: "DELETE" });
        location.reload();
      } catch (err) { toast(err.message); }
    };
    return h("div", { class: "frow" },
      h("div", { class: "grow" }, h("div", { class: "ft" }, where), h("div", { class: "rd" }, `${e.crops} face pictures. ${ago(e.at)}.`)),
      deleteButton(`Delete the ${where} capture`, remove));
  });

  return h("section", { class: "card fcard", "aria-labelledby": "face-title" },
    cardHead("face-title", "face", "Vision Match", about, statusSlot),
    bar,
    live.node,
    p.has_face ? h("div", { class: "fpart" }, h("h3", {}, "Face pictures"), faceThumbs(p)) : null,
    captures.length ? disclose("Captures", captures.length, null, rows) : null);
}

// ── The page ────────────────────────────────────────────────────────────

export async function render({ id }) {
  const [{ people }, { kiosks }, suggestions] = await Promise.all([api("api/people"), api("api/kiosks"),
    api("api/suggestions").catch(() => ({}))]);
  const person = people.find((p) => p.id === id) || people[0];
  const kinds = deviceKinds(kiosks);

  const list = people.length
    ? people.map((p) => {
      const voice = voiceCoverage(p, kinds);
      return h("a", { class: `row nowrap ${p === person ? "sel" : ""}`, href: `#/people/${p.id}` },
        disc(p),
        h("div", { class: "grow" },
          h("div", { class: "rt" }, p.name),
          h("div", { class: "pstat" },
            h("span", { class: voice.ok ? "on" : "", title: voice.ok ? "Voice enrolled" : "No voice enrolled" },
              icon("mic", 14), h("span", { class: "sr" }, "Voice: "), voice.text),
            h("span", { class: p.has_face ? "on" : "", title: p.has_face ? "Face enrolled" : "No face enrolled" },
              icon("face", 14), p.has_face ? "Face" : "No face"))));
    })
    : [h("div", { class: "empty" }, "Nobody here yet. Add a person, then enroll their voice or face on a kiosk.")];

  const add = async () => {
    const name = await ask({ title: "Add a person", field: { label: "Name", placeholder: "Name" }, ok: "Add" });
    if (!name) return;
    try {
      const p = await api("api/people", { method: "POST", json: { name } });
      location.hash = `#/people/${p.id}`;
    } catch (err) { toast(err.message); }
  };

  const events = person ? socket("api/events", { type: "listen" }) : null;
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
                h("div", { class: "grow" }, h("div", { class: "rt" }, "Add a person"))))),
          person ? detail(person, kiosks, kinds, suggestions, events) : null)),
    ],
    cleanup: () => events?.close(),
  };
}

function detail(p, kiosks, kinds, suggestions, events) {
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
    if (!await ask({ title: `Forget ${p.name}?`, text: "Their voice, their face and all their enrollment audio and face pictures are deleted.",
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
  const voice = voiceCoverage(p, kinds);
  const summary = h("div", { class: "chips" },
    p.has_voice
      ? chip(voice.of ? `Voice on ${voice.n} of ${voice.of} device models` : "Voice enrolled", voice.n === voice.of ? "c-ok big" : "c-warn big", icon("mic", 15))
      : chip("No voice yet", "c-off big", icon("mic", 15)),
    p.has_face ? chip("Face enrolled", "c-ok big", icon("face", 15)) : chip("No face yet", "c-off big", icon("face", 15)));
  return h("section", { class: "right", "aria-label": p.name },
    h("div", { class: "card" },
      h("div", { class: "head" },
        disc(p, "big"),
        h("div", { class: "grow" }, h("div", { class: "hname" }, p.name), summary),
        h("button", { class: "ib", "aria-label": `Rename ${p.name}`, onclick: rename }, icon("edit"))),
      h("div", { class: "colorrow" },
        h("span", { class: "clabel" }, "Color on the kiosk"),
        swatches,
        h("span", { class: "rd" }, `The Echo Dot ring and the Echo Show screen take this color while ${p.name} is talking.`))),
    voiceCard(p, kiosks, kinds),
    visionCard(p, kiosks, events),
    section(`What's ${p.name}'s`, `What "my" means when ${p.name} talks. "Add milk to my list" goes on this list on every kiosk.`),
    h("div", { class: "card" }, things,
      h("div", { class: "hint" }, icon("info", 18), 'Kiosk Satellite uses these when this person says "my", with its personal commands and to-do blueprints. Without a list of their own, "my list" is the shared list.')),
    h("div", { class: "card forget" },
      h("div", { class: "grow" }, h("div", { class: "rt" }, `Forget ${p.name}`),
        h("div", { class: "rd" }, `Deletes ${p.name}'s voice and face with every recording and face picture.`)),
      h("button", { class: "btn out danger sm", onclick: forget }, `Forget ${p.name}`)));
}
