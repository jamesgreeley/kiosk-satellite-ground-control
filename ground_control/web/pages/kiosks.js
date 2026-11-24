// Kiosks: every kiosk that has connected, and the one picked with its
// details, its Voice Match features, what it hears and sees right now (the
// live view) and how much it looks for Vision Match.
import { api, socket } from "../api.js";
import { shell } from "../app.js";
import { ago, ask, chip, disc, h, icon, kioskAnswers, kioskChip, missingPeople, modeLabel, pageHead, section, toast } from "../ui.js";

const WINDOW_S = 10;
const fingers = (n) => `${n} finger${n === 1 ? "" : "s"}`;

/** The state in the list: a dot, so the kiosk's name keeps the room. */
function stateDot(k) {
  const [label, color] = k.access === "pending" ? ["Waiting for approval", "var(--warn)"]
    : k.access === "declined" ? ["Declined", "var(--error)"]
    : k.in_session ? ["In a session", "var(--primary)"] : k.connected ? ["Connected", "#4f9a55"] : ["Offline", "var(--outline)"];
  return h("span", { class: "dot", role: "img", "aria-label": label, title: label, style: { background: color, width: "10px", height: "10px" } });
}

function row(title, desc, value) {
  return h("div", { class: "row" },
    h("div", { class: "grow" }, h("div", { class: "rt" }, title), desc ? h("div", { class: "rd" }, desc) : null),
    value);
}

const onOff = (on, yes = "On") => (on ? chip(yes, "c-ok") : chip("Off", "c-off"));

/** Re-render the page, after the kiosk's access changed. */
const again = () => window.dispatchEvent(new HashChangeEvent("hashchange"));

async function setAccess(kiosk, access) {
  try {
    await api(`api/kiosks/${encodeURIComponent(kiosk.device_id)}/access`, { method: "POST", json: { access } });
    toast(access === "approved" ? `${kiosk.name} approved` : `${kiosk.name} declined`);
    again();
  } catch (err) {
    toast(`Couldn't change ${kiosk.name}'s access: ${err.message}`);
  }
}

/** A kiosk that isn't approved: what that means and the buttons to decide. */
function accessCard(kiosk) {
  const pending = kiosk.access === "pending";
  return h("div", { class: "card", style: { marginBottom: "16px" } },
    h("div", { class: "row" },
      h("span", { class: "ico" }, icon("info", 24)),
      h("div", { class: "grow" },
        h("div", { class: "rt" }, pending ? "Waiting for approval" : "Declined"),
        h("div", { class: "rd" }, pending
          ? "This kiosk connected with Kiosk Satellite. Approve it to let it use Voice Match. It gets no names or voices until then."
          : "This kiosk can't use Ground Control until you approve it."))),
    h("div", { class: "row", style: { justifyContent: "flex-end", gap: "8px" } },
      pending ? h("button", { class: "btn out sm", onclick: () => setAccess(kiosk, "declined") }, "Decline") : null,
      h("button", { class: "btn fill sm", onclick: () => setAccess(kiosk, "approved") }, "Approve")));
}

/** The Voice Match settings the kiosk has on, as it last said. */
function featureRows(kiosk, lastSession, people, status) {
  const f = kiosk.features || {};
  if (!Object.keys(f).length) {
    return [h("div", { class: "empty" }, "Not known yet. The kiosk says which features it has on when it connects.")];
  }
  if (f.voice_match === false) return [row("Voice Match", "Off on the kiosk, under Enable Voice Match.", onOff(false))];
  const isolation = !f.trimming ? chip("Needs Trimming", "c-off")
    : f.isolation === false ? onOff(false)
    : status.isolation ? chip("On", "c-ok") : chip("Off in Ground Control", "c-off");
  return [
    missingPeople(f, people)
      ? row("Recognition", `Answers: ${kioskAnswers(kiosk, lastSession, people)}. Choose them again in Kiosk Satellite, under Ground Control, Voice Match, People.`,
            chip("Choose again", "c-warn"))
      : row("Recognition", `Answers: ${kioskAnswers(kiosk, lastSession, people) || "Anyone"}.`, onOff(true)),
    row("Trimming", "Other voices cut out between the user's words.", onOff(f.trimming)),
    row("Isolation", "Other voices removed under the user's words.", isolation),
    row("Speaker sensor", "Who is talking, in Home Assistant.", onOff(f.speaker_sensor)),
    row("Show who is talking", "The speaker's name on the voice overlay.", onOff(f.show_speaker)),
    row("Enrollment", "Allow enrollment here.", onOff(kiosk.enroll, "Allowed")),
  ];
}

export async function render({ id }) {
  const [{ kiosks }, { people }, status] = await Promise.all([api("api/kiosks"), api("api/people"), api("api/status")]);
  const kiosk = kiosks.find((k) => k.device_id === id) || kiosks[0];
  const listChips = new Map();

  const list = kiosks.length
    ? kiosks.map((k) => {
        const c = h("span", { class: "ico" }, stateDot(k));
        listChips.set(k.device_id, c);
        return h("a", { class: `row nowrap ${k === kiosk ? "sel" : ""}`, href: `#/kiosks/${encodeURIComponent(k.device_id)}` },
          h("span", { class: "ico" }, icon("screen", 24)),
          h("div", { class: "grow" }, h("div", { class: "rt" }, k.name), h("div", { class: "rd" }, k.model || "Unknown model")),
          c);
      })
    : [h("div", { class: "empty" }, "No kiosks yet. A kiosk shows up here once Kiosk Satellite connects to Ground Control.")];

  const out = kiosk ? await detail(kiosk, people, status, (k) => listChips.get(k.device_id)?.replaceChildren(stateDot(k))) : null;
  return {
    nodes: [
      shell("kiosks"),
      h("main", { class: "page" },
        pageHead("Kiosks"),
        h("div", { class: "split" },
          h("section", { class: "left", "aria-label": "Every kiosk" },
            h("div", { class: "card" }, list,
              kiosks.length ? h("div", { class: "hint", style: { paddingTop: "14px" } }, icon("info", 18),
                "Each kiosk's settings are on the kiosk, in Kiosk Satellite under Ground Control.") : null)),
          out?.node)),
    ],
    cleanup: () => out?.cleanup(),
  };
}

async function detail(kiosk, people, status, onKiosk) {
  const device = kiosk.device_id;
  const q = `device_id=${encodeURIComponent(device)}`;
  const [{ sessions }, voice, looks] = await Promise.all([
    api(`api/sessions?limit=5&${q}`), api(`api/sessions?limit=1&kind=voice&${q}`), api(`api/vision/looks?days=7&${q}`),
  ]);
  const byId = Object.fromEntries(people.map((p) => [p.id, p]));
  const last = voice.sessions[0];
  const days = looks[device] || [];

  const state = h("span");
  const connection = h("span", { class: "val" });
  const drawState = () => {
    state.replaceChildren(kioskChip(kiosk));
    connection.textContent = kiosk.connected
      ? `Connected ${ago(kiosk.connected_at)}.`
      : `Last seen ${ago(kiosk.last_seen)}.`;
    onKiosk(kiosk);
  };
  drawState();

  // Who is talking, and the scores of the current session's last 10 seconds.
  const nowBox = h("div", { class: "now" });
  const drawNow = (speaker, level) => nowBox.replaceChildren(...[
    disc(speaker), h("span", { class: "big" }, speaker?.name || "Nobody yet"),
    level ? h("span", { class: "val" }, level) : null].filter(Boolean));
  drawNow(null);
  const points = [];
  let current = null;
  const chartBox = h("div", { class: "chart", role: "img", "aria-label": "Voice scores, last 10 seconds" });
  const drawChart = () => {
    const end = points.length ? points[points.length - 1].t : 0;
    chartBox.replaceChildren(
      h("span", { class: "yl", style: { bottom: "100%" } }, "1.0"),
      h("span", { class: "yl", style: { bottom: "0%" } }, "0"),
      h("div", { class: "thr", style: { bottom: `${status.threshold * 100}%` } }, h("span", { class: "thrlbl" }, `Recognized, ${status.threshold}`)),
      h("div", { class: "thr", style: { bottom: `${status.other_threshold * 100}%` } }, h("span", { class: "thrlbl" }, `Other voice, ${status.other_threshold}`)),
      ...points.filter((p) => p.t > end - WINDOW_S).map((p) => h("span", {
        class: "pt", style: {
          left: `${100 - ((end - p.t) / WINDOW_S) * 100}%`, bottom: `${Math.max(0, Math.min(1, p.score)) * 100}%`,
          background: p.low ? "var(--error)" : current?.color || "var(--primary)",
        },
      })));
  };
  drawChart();

  const log = h("div", { class: "card" }, h("div", { class: "empty" }, "Waiting for the next command or vision session on this kiosk."));
  const entries = [];
  const addLog = (title, desc) => {
    entries.unshift([new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" }), title, desc]);
    entries.length = Math.min(entries.length, 30);
    log.replaceChildren(...entries.map(([t, a, b]) => h("div", { class: "row" }, h("span", { class: "t" }, t),
      h("div", { class: "grow" }, h("div", { class: "rt" }, a), h("div", { class: "rd" }, b)))));
  };

  // What the camera sees, while a vision session goes on: the last 10 seconds of looks.
  const seen = [];
  const seeBox = h("div", { class: "now" });
  const strip = h("div", { class: "vlanes", style: { marginTop: "4px" } });
  const drawSeen = () => {
    const lk = seen[seen.length - 1];
    const face = lk?.face ? (lk.face.frontal ? "Looking at the kiosk" : "Face turned away") : "No face";
    const hand = lk?.hand ? (lk.hand.fingers != null ? fingers(lk.hand.fingers) : "Hand") : null;
    const ms = lk ? Object.values(lk.ms || {}).reduce((a, b) => a + b, 0) : null;
    seeBox.replaceChildren(...[h("span", { class: "disc vis" }, icon("eye", 20)),
      h("span", { class: "big" }, lk ? [face, hand].filter(Boolean).join(", ") : "Nobody in front of it"),
      ms != null ? h("span", { class: "val" }, `${ms.toFixed(0)} ms`) : null].filter(Boolean));
    const end = Date.now() / 1000;
    const mark = (lk2, m) => h("span", { class: `vm ${m.cls}`, style: { left: `${100 - ((end - lk2.at) / WINDOW_S) * 100}%` } }, m.text ?? null);
    const recentLooks = seen.filter((x) => x.at > end - WINDOW_S);
    strip.replaceChildren(
      h("div", { class: "vlane" }, h("span", { class: "vll" }, "Face"), h("div", { class: "vlt" },
        recentLooks.map((x) => mark(x, { cls: x.face ? (x.face.frontal ? "on" : "away") : "none" })))),
      h("div", { class: "vlane" }, h("span", { class: "vll" }, "Hand"), h("div", { class: "vlt" },
        recentLooks.filter((x) => x.hand !== undefined).map((x) => mark(x, x.hand ? { cls: "hand", text: x.hand.fingers ?? "" } : { cls: "none" })))));
  };
  drawSeen();
  const seeTimer = setInterval(() => { if (seen.length) drawSeen(); }, 1000);

  const events = socket("api/events", { type: "listen" });
  events.addEventListener("message", (m) => {
    const e = JSON.parse(m.data);
    if (e.type === "kiosk" && e.device_id === device) {
      const was = [kiosk.access, JSON.stringify(kiosk.features), kiosk.enroll].join();
      Object.assign(kiosk, e);
      drawState();
      if ([kiosk.access, JSON.stringify(kiosk.features), kiosk.enroll].join() !== was) drawFeatures();
      return;
    }
    if (e.device_id !== device) return;
    switch (e.type) {
      case "session_start":
        kiosk.in_session = true; drawState(); points.length = 0; current = null; drawChart(); drawNow(null);
        addLog("Session started", `${modeLabel(e.mode)} mode.`); break;
      case "speech_start": addLog("Speech", "Someone started talking."); break;
      case "speaker":
        current = e.id ? byId[e.id] || e : null;
        drawNow(current, e.score ? `score ${e.score}` : null);
        drawChart();
        addLog(e.name ? `${e.name} recognized` : e.by === "answer" ? "Short answer taken" : "Unknown voice", e.score ? `Score ${e.score}.` : "The only person enrolled.");
        break;
      case "score": points.push(e); drawChart(); break;
      case "other_voice": addLog("Someone else", `Two checks under ${status.other_threshold}.`); break;
      case "rejected": addLog("Rejected", "Not a voice this kiosk answers."); break;
      case "end_of_speech": addLog("Speech ended", e.reason === "silence" ? "The speaker stopped talking." : e.reason === "other_voice" ? "Another voice started." : "The session limit."); break;
      case "session_end": kiosk.in_session = false; drawState(); break;
      case "enroll_start": addLog(`Enrolling ${e.name}`, "Started from the app."); break;
      case "enroll_saved": addLog("Enrollment saved", `${e.added_s} s of speech added.`); break;
      case "vision_start": seen.length = 0; addLog("Vision session started", "A face or a hand in front of the kiosk."); break;
      case "vision_look":
        seen.push({ ...e, at: Date.now() / 1000 });
        if (seen.length > 200) seen.splice(0, seen.length - 200);
        drawSeen();
        break;
      case "vision_outcome":
        if (e.what === "gesture") addLog(`${fingers(e.fingers ?? 0)} gesture`, e.action ? `${e.action}.` : "");
        else addLog(e.what === "screensaver_dismissed" ? "Screensaver dismissed" : "Screensaver postponed", "");
        break;
      case "vision_end":
        addLog("Vision session ended", e.reason === "idle" ? "Nothing seen for 10 s." : e.reason === "long" ? "It goes on in a new session." : "The camera stopped.");
        seen.length = 0; drawSeen();
        break;
      default: break;
    }
  });

  const recent = sessions.length
    ? sessions.map((s) => {
        if (s.kind === "vision") {
          return h("a", { class: "row", href: `#/sessions/${s.id}` },
            h("span", { class: "disc vis" }, icon("eye", 20)),
            h("div", { class: "grow" }, h("div", { class: "rt" }, "Vision Match"),
              h("div", { class: "rd" }, `${[...(s.summary || []), `${s.duration_s ?? 0} s`].join(". ")}.`)),
            h("span", { class: "time" }, ago(s.started)));
        }
        const person = s.speaker?.id ? byId[s.speaker.id] || s.speaker : null;
        return h("a", { class: "row", href: `#/sessions/${s.id}` },
          disc(person),
          h("div", { class: "grow" }, h("div", { class: "rt" }, person?.name || "Unknown voice"),
            h("div", { class: "rd" }, `${modeLabel(s.mode)} mode. ${s.audio_s ?? 0} s.`)),
          h("span", { class: "time" }, ago(s.started)));
      })
    : [h("div", { class: "empty" }, "No sessions from this kiosk yet.")];

  const features = h("div", { class: "card" });
  const drawFeatures = () => {
    features.replaceChildren(...featureRows(kiosk, last, people, status),
      h("div", { class: "hint", style: { paddingTop: "14px" } }, icon("info", 18),
        "Set on the kiosk, in Kiosk Satellite under Ground Control, Voice Match."));
  };
  drawFeatures();

  const removeAccess = async () => {
    const yes = await ask({ title: `Remove ${kiosk.name}'s access?`, ok: "Remove access", danger: true,
      text: "It stops using Voice Match until you approve it again." });
    if (yes !== null) setAccess(kiosk, "declined");
  };

  const node = h("section", { class: "right", "aria-label": kiosk.name },
    kiosk.access !== "approved" ? accessCard(kiosk) : null,
    h("div", { class: "card" },
      h("div", { class: "head" },
        h("span", { class: "kioskmark" }, icon("screen", 32)),
        h("div", { class: "grow" }, h("div", { class: "hname" }, kiosk.name), h("div", { class: "rd" }, kiosk.model || "Unknown model")),
        state),
      row("Connection", null, connection),
      row("Kiosk Satellite", null, h("span", { class: "val" }, kiosk.version || "Unknown"))),
    section("Voice Match"),
    features,
    section("Vision Match", "Looks a day. An empty look found no face or hand and opened no session."),
    h("div", { class: "card" }, days.length
      ? days.map((d) => h("div", { class: "row" },
        h("div", { class: "grow" }, h("div", { class: "rt" },
          new Date(`${d.day}T12:00:00`).toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" }))),
        h("span", { class: "pvs" },
          ...[[d.looks, "looks"], [d.empty, "empty"], [d.sessions, "sessions"]].map(([v, label]) =>
            h("span", { class: "pv" }, h("b", {}, v.toLocaleString()), h("small", {}, label))))))
      : h("div", { class: "empty" }, "No looks from this kiosk in the last 7 days.")),
    section("Who is talking"),
    h("div", { class: "card" }, nowBox, h("div", { class: "tl", style: { paddingTop: "4px" } }, chartBox),
      h("p", { class: "hint", style: { margin: 0 } }, "Each check of the current command. Colored passed, red means someone else.")),
    section("What it sees"),
    h("div", { class: "card" }, seeBox, h("div", { class: "tl", style: { paddingTop: "4px" } }, strip),
      h("p", { class: "hint", style: { margin: 0 } }, "Each look of the current vision session, for kiosks with Vision Match on.")),
    section("Events"),
    log,
    section("Recent sessions", null, h("a", { class: "txtlink", href: "#/sessions" }, "All sessions")),
    h("div", { class: "card" }, recent),
    kiosk.access === "approved" ? h("div", { style: { display: "flex", justifyContent: "flex-end", marginTop: "16px" } },
      h("button", { class: "btn danger sm", onclick: removeAccess }, "Remove access")) : null);
  return { node, cleanup: () => { events.close(); clearInterval(seeTimer); } };
}
