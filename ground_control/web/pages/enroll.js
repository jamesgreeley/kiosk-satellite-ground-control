// Enrollment always records through a kiosk's own microphone: a voice is
// recognized best on the device models it was enrolled on. This page shows
// the text to read, the same the kiosk shows (an Echo Dot has no screen),
// with the sentence to read filling as it is spoken, and follows the progress.
import { api, socket } from "../api.js";
import { shell } from "../app.js";
import { h, icon, section, toast } from "../ui.js";

const TARGET_S = 45;
// Why a kiosk couldn't start, by the code it sends back.
const KIOSK_ERRORS = {
  busy: "The kiosk is busy with a voice command or an alarm. Try again in a moment.",
  off: "Allow enrollment here is off on that kiosk.",
  no_mic: "The kiosk's microphone isn't available. Enrollment needs Voice Satellite's native runtime with wake word detection on.",
  muted: "The kiosk's microphone is muted.",
  lost: "The kiosk lost its connection to Ground Control.",
  stopped: "Stopped on the kiosk.",
};

function stepper(at) {
  return h("ol", { class: "steps" }, ["Name and kiosk", "Read aloud", "Check"].map((label, i) =>
    h("li", { class: `step ${i < at ? "done" : i === at ? "on" : ""}`, "aria-current": i === at ? "step" : null },
      h("span", { class: "n" }, i + 1), label)));
}

function kioskLabel(k) {
  return k.model ? `${k.name} (${k.model})` : k.name;
}

export async function render({ id, device }) {
  const [{ people }, { kiosks }] = await Promise.all([api("api/people"), api("api/kiosks")]);
  const existing = people.find((p) => p.id === id);
  const connected = kiosks.filter((k) => k.connected);
  const live = connected.filter((k) => k.enroll);
  const main = h("main", { class: "page narrow" });
  const title = h("span");
  let stop = () => {};

  // ── step 1 ──────────────────────────────────────────────────────────────
  function stepOne() {
    title.textContent = existing ? `Enroll ${existing.name}` : "Enroll a person";
    if (!live.length) {
      main.replaceChildren(stepper(0), h("div", { class: "card" },
        h("div", { class: "empty" }, connected.length
          ? "No connected kiosk allows enrollment. Turn on Allow enrollment here on a kiosk's Ground Control Voice Match page."
          : "No kiosk is connected. Enrollment records through a kiosk's own microphone, so it needs one running Kiosk Satellite connected to Ground Control.")));
      return;
    }
    const covered = new Set(existing?.models || []);
    const pick = live.find((k) => k.device_id === device)
      || live.find((k) => k.model && !covered.has(k.model)) || live[0];
    const name = h("input", { class: "fld wide", id: "name", value: existing?.name || "", disabled: !!existing, placeholder: "Name" });
    const where = h("select", { class: "fld", id: "where" }, live.map((k) => h("option", {
      value: k.device_id, selected: k === pick,
    }, `${kioskLabel(k)}${k.model && covered.has(k.model) ? ", enrolled" : ""}`)));
    const start = h("button", { class: "btn fill" }, "Start");
    start.addEventListener("click", async () => {
      let person = existing;
      if (!person) {
        if (!name.value.trim()) { name.focus(); return; }
        try { person = await api("api/people", { method: "POST", json: { name: name.value.trim() } }); }
        catch (err) { toast(err.message); return; }
      }
      stepTwo(person, live.find((k) => k.device_id === where.value));
    });
    main.replaceChildren(stepper(0),
      h("div", { class: "card" },
        h("div", { class: "row" }, h("div", { class: "grow" }, h("div", { class: "rt" }, h("label", { for: "name" }, "Name")),
          h("div", { class: "rd" }, "How the kiosks and the assistant call this person.")), name),
        h("div", { class: "row" }, h("div", { class: "grow" }, h("div", { class: "rt" }, h("label", { for: "where" }, "Record on")),
          h("div", { class: "rd" }, "The kiosk's own microphone records. Enroll on each device model this person talks to.")), where),
        h("div", { class: "hint" }, icon("info", 18), "A voice is recognized best on the device models it was enrolled on. An Echo Show and an Echo Dot hear the same voice differently.")),
      h("div", { class: "foot" }, h("a", { class: "btn txt", href: existing ? `#/people/${existing.id}` : "#/people" }, "Cancel"), start));
  }

  // ── step 2 ──────────────────────────────────────────────────────────────
  function stepTwo(person, kiosk) {
    title.textContent = `Enroll ${person.name}`;
    const secs = h("span", { class: "secs" }, "0 s");
    const ring = h("div", { class: "ring", role: "img", "aria-label": `0 of ${TARGET_S} seconds of speech` },
      h("div", { class: "ringin" }, secs, h("span", { class: "of" }, `of ${TARGET_S} s`)));
    const status = h("div", { class: "rt" }, `Starting on the ${kiosk.name}`);
    const levels = Array(40).fill(4);
    const bars = h("div", { class: "level", "aria-hidden": "true" });
    const warn = h("div", { class: "hint warn", style: { display: "none" } });
    // Ground Control paces the passage: which sentence to read and how much
    // of it was spoken, the same on the kiosk.
    const passage = h("div", { class: "passage" }, "Waiting for the kiosk");
    let sentences = [];
    let at = -1;
    const now = h("div", { class: "now" });
    const fillColor = `color-mix(in srgb, ${person.color || "var(--primary)"} 32%, var(--surface))`;
    const useText = (list) => {
      if (!Array.isArray(list) || !list.length) return;
      sentences = list;
      at = -1;
      drawSentence(0, 0);
    };
    // The sentences read so far, the one to read on a line of its own, filling
    // as it is spoken, then the rest.
    const drawSentence = (index, fill) => {
      if (!sentences.length) return;
      if (index !== at) {
        at = index;
        now.textContent = sentences[at];
        passage.replaceChildren(
          at ? h("span", { class: "said" }, sentences.slice(0, at).join(" ")) : "",
          now,
          h("span", {}, sentences.slice(at + 1).join(" ")));
        now.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
      const pct = Math.round(Math.min(1, fill) * 100);
      now.style.background = `linear-gradient(90deg, ${fillColor} ${pct}%, var(--s2) ${pct}%)`;
    };
    const progress = (e) => {
      secs.textContent = `${Math.floor(e.speech_s)} s`;
      const pct = Math.min(100, (e.speech_s / TARGET_S) * 100);
      ring.style.background = `conic-gradient(var(--fill) 0 ${pct}%, var(--s2) ${pct}% 100%)`;
      ring.setAttribute("aria-label", `${Math.floor(e.speech_s)} of ${TARGET_S} seconds of speech`);
      levels.push(Math.max(4, Math.min(40, (e.level + 60) * 0.8)));
      levels.shift();
      bars.replaceChildren(...levels.map((v) => h("span", { style: { height: `${v}px` } })));
      if (typeof e.sentence === "number") drawSentence(e.sentence, e.sentence_fill || 0);
      status.textContent = "Keep reading";
    };
    const otherVoice = () => {
      warn.style.display = "";
      warn.replaceChildren(icon("warn", 18), `Another voice was heard at ${new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" })}. That part was left out.`);
    };
    const finished = (e) => { stop(); stepThree(person, kiosk, e); };
    const failed = (e) => { stop(); toast(KIOSK_ERRORS[e.code] || e.message || "Nothing was saved"); stepOne(); };

    const events = socket("api/events", { type: "listen" });
    events.addEventListener("message", (m) => {
      const e = JSON.parse(m.data);
      if (e.person_id !== person.id) return;
      if (e.type === "enroll_start") useText(e.passage);
      if (e.type === "enroll_progress") progress(e);
      if (e.type === "enroll_other_voice") otherVoice();
      if (e.type === "enroll_saved") finished(e);
      if (e.type === "enroll_error") failed(e);
    });
    events.addEventListener("open", () => api(`api/kiosks/${encodeURIComponent(kiosk.device_id)}/enroll`, {
      method: "POST", json: { person_id: person.id, target_s: TARGET_S },
    }).then(() => { status.textContent = `Read aloud near the ${kiosk.name}`; }).catch((err) => failed(err)));
    stop = () => events.close();

    main.replaceChildren(stepper(1),
      h("div", { class: "card" },
        h("div", { class: "live" }, ring, h("div", { class: "grow" }, status,
          h("div", { class: "rd" }, `Only ${person.name}'s speech counts. The ${kiosk.name} stops on its own at ${TARGET_S} s.`), bars)),
        warn),
      section("Read this aloud"),
      h("div", { class: "card" }, passage),
      section("While reading"),
      h("div", { class: "card tips" },
        h("div", { class: "tip" }, icon("walk", 24), h("div", {}, h("div", { class: "rt" }, "Walk slowly"), h("div", { class: "rd" }, "Close to the kiosk, then across the room."))),
        h("div", { class: "tip" }, icon("turn", 24), h("div", {}, h("div", { class: "rt" }, "Turn around"), h("div", { class: "rd" }, "Face away from the kiosk now and then."))),
        h("div", { class: "tip" }, icon("mic", 24), h("div", {}, h("div", { class: "rt" }, "Talk normally"), h("div", { class: "rd" }, "The voice you use for commands, not a reading voice.")))),
      h("div", { class: "foot" },
        h("button", { class: "btn txt", onclick: () => { stop(); location.hash = `#/people/${person.id}`; } }, "Cancel")));
  }

  // ── step 3 ──────────────────────────────────────────────────────────────
  async function stepThree(person, kiosk, e) {
    const { kiosks: all } = await api("api/kiosks");
    const fresh = (await api("api/people")).people.find((p) => p.id === person.id) || person;
    const covered = new Set(fresh.models || []);
    const next = all.find((k) => k.connected && k.model && !covered.has(k.model));
    main.replaceChildren(stepper(2),
      h("div", { class: "card" },
        h("div", { class: "row" }, h("div", { class: "grow" }, h("div", { class: "rt" }, `${person.name} is enrolled on the ${kiosk.name}`),
          h("div", { class: "rd" }, `${e.added_s} s of speech added, ${e.speech_s} s in all.`))),
        e.dropped_s ? h("div", { class: "hint" }, icon("info", 18), `${e.dropped_s} s that sounded like someone else was left out.`) : null,
        next ? h("div", { class: "hint" }, icon("info", 18), `Next, enroll on the ${kioskLabel(next)}: no ${next.model} has ${person.name}'s voice yet.`) : null),
      h("div", { class: "foot" },
        next ? h("a", { class: "btn out", href: `#/enroll/${person.id}/${encodeURIComponent(next.device_id)}` }, "Enroll next") : null,
        h("a", { class: "btn fill", href: `#/people/${person.id}` }, "Done")));
  }

  stepOne();
  return { nodes: [shell("people", { back: "#/people", title: [title] }), main], cleanup: () => stop() };
}
