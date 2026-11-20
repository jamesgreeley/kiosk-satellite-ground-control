import { api, audioBytes, audioUrl } from "../api.js";
import { shell } from "../app.js";
import { answersLabel, ask, chip, clock, h, icon, modeLabel, outcome, outcomeChip, secs, section, toast } from "../ui.js";

const BARS = 104;

function pcm(buffer) {
  // Our WAVs are 16 kHz mono PCM16 with a 44-byte header.
  return new Int16Array(buffer.slice(44));
}

/** Follow mode: the stretches that went to the assistant in the speaker's
 * color, everything left out in the other voice's. */
function followWave(samples, total, released, color) {
  const per = Math.max(1, Math.floor(samples.length / BARS));
  const bars = [];
  for (let i = 0; i < BARS; i++) {
    let peak = 0;
    for (let j = i * per; j < Math.min(samples.length, (i + 1) * per); j++) peak = Math.max(peak, Math.abs(samples[j]));
    const t = ((i + 0.5) / BARS) * total;
    const sent = released.some(([a, b]) => t >= a && t < b);
    bars.push(h("span", { style: { height: `${Math.max(3, Math.round((peak / 32768) * 72))}px`,
      background: sent ? color : "var(--other)", opacity: sent ? 1 : 0.45 } }));
  }
  return bars;
}

function waveform(samples, total, events, color) {
  const start = events.find((e) => e.type === "speech_start")?.t ?? 0;
  const cut = events.find((e) => e.type === "other_voice")?.cut;
  const end = events.find((e) => e.type === "end_of_speech")?.t;
  const per = Math.max(1, Math.floor(samples.length / BARS));
  const bars = [];
  for (let i = 0; i < BARS; i++) {
    let peak = 0;
    for (let j = i * per; j < Math.min(samples.length, (i + 1) * per); j++) peak = Math.max(peak, Math.abs(samples[j]));
    const t = ((i + 0.5) / BARS) * total;
    let c = color, o = 1;
    if (t < start) o = 0.55;
    if (cut !== undefined && t >= cut) c = "var(--other)";
    if (end !== undefined && t > end) o = 0.3;
    bars.push(h("span", { style: { height: `${Math.max(3, Math.round((peak / 32768) * 72))}px`, background: c, opacity: o } }));
  }
  return bars;
}

// What each point on the chart is. Who is speaking is judged against the
// Recognized line, whether it is still the user's voice against Other voice.
const POINT = {
  anchor: "The wake word: who said it",
  identify: "Who is speaking",
  check: "Still the user's voice",
  word: "Still the user's voice",
};

function chart(record, total, color) {
  const pts = (record.timeline || []).filter((p) => p.t !== undefined && p.score !== undefined
    && (record.mode === "follow" || p.turn === 0 || p.turn === undefined)).map((p) => {
    const who = p.kind === "anchor" || p.kind === "identify";
    const label = p.kind === "identify" && p.wake_word ? "Who is speaking, from the wake word and the command"
      : p.kind === "check" && p.low ? "Not the user's voice" : POINT[p.kind] || "Check";
    return h("span", {
      class: `pt ${who ? "who" : ""}`,
      title: `${label} at ${p.t} s: ${p.score}`,
      style: {
        left: `${(p.t / total) * 100}%`, bottom: `${Math.max(0, Math.min(1, p.score)) * 100}%`,
        background: p.kind === "check" && p.low ? "var(--error)" : who ? "var(--primary)" : color,
      },
    });
  });
  return h("div", { class: "chart", role: "img", "aria-label": "Voice score over time" },
    h("span", { class: "yl", style: { bottom: "100%" } }, "1.0"),
    h("span", { class: "yl", style: { bottom: "0%" } }, "0"),
    h("div", { class: "thr", style: { bottom: `${record.threshold * 100}%` } }, h("span", { class: "thrlbl" }, `Recognized, ${record.threshold}`)),
    h("div", { class: "thr", style: { bottom: `${record.other_threshold * 100}%` } }, h("span", { class: "thrlbl" }, `Other voice, ${record.other_threshold}`)),
    pts);
}

function chartKey(record, color) {
  const low = (record.timeline || []).some((p) => p.kind === "check" && p.low);
  return h("div", { class: "chartkey" },
    h("span", {}, h("span", { class: "pt who key", style: { background: "var(--primary)" } }), "Who is speaking, against Recognized"),
    h("span", {}, h("span", { class: "pt key", style: { background: color } }), "Still the user's voice, against Other voice"),
    low ? h("span", {}, h("span", { class: "pt key", style: { background: "var(--error)" } }), "Under Other voice: not the user, left out") : null);
}

function steps(record) {
  const out = (record.followups || []).map((t) => ["Follow-up", "The assistant asked something and the user answered. This part is the answer.", t]);
  for (const e of record.events || []) {
    if (e.type === "speech_start") out.push(["Speech started", "The voice check starts here.", e.t]);
    if (e.type === "speaker" && e.name) {
      const both = (record.timeline || []).some((p) => p.kind === "identify" && p.wake_word && p.score === e.score);
      out.push([`${e.name} recognized`, e.by === "only_person" ? "The only person enrolled, with no voice to compare."
        : e.by === "conversation" ? "Recognized earlier in this conversation. Only their voice goes on."
        : both ? `Score ${e.score}, from the wake word and the command together. The command alone was too short.` : `Score ${e.score}.`, e.t]);
    }
    if (e.type === "speaker" && !e.name) out.push(["Unknown voice", `Best score ${e.score}, under the threshold.`, e.t]);
    if (e.type === "other_voice" && record.mode === "follow") {
      out.push(["Another voice left out", "It never reached the assistant.", e.t]);
    } else if (e.type === "other_voice") {
      out.push(["Speaker's last word", "Everything after it was left out.", e.cut]);
      out.push(["Another voice", `Two checks in a row under ${record.other_threshold}.`, e.t]);
    }
    if (e.type === "turn_start") out.push([e.turn > 1 ? `Turn ${e.turn}` : "The user started talking", "What they say goes to the assistant.", e.t]);
    if (e.type === "turn_end") out.push(["The user stopped", `${e.speech_s ?? 0} s of speech. The command ended here.`, e.t]);
    if (e.type === "rejected") out.push(["Rejected", "This kiosk answers enrolled voices only.", e.t]);
    if (e.type === "end_of_speech" && e.reason === "silence") out.push(["Speaker stopped", "Silence after the speech.", e.t]);
  }
  for (const p of record.timeline || []) {
    if (p.kind === "isolated") out.push(["Isolation", `Other voices under the user's words removed, in ${p.ms} ms. The command went to the assistant after it.`, p.t]);
  }
  return out.sort((a, b) => a[2] - b[2]).map(([title, desc, t]) => h("div", { class: "row" },
    h("div", { class: "grow" }, h("div", { class: "rt" }, title), h("div", { class: "rd" }, desc)),
    h("span", { class: "t" }, `${Number(t).toFixed(2)} s`)));
}

function player(path, title, desc, length) {
  const audio = new Audio();
  const btn = h("button", { class: "play", "aria-label": `Play ${title.toLowerCase()}` }, icon("play", 20));
  btn.addEventListener("click", async () => {
    if (!audio.src) {
      try { audio.src = await audioUrl(path); } catch { toast("That audio wasn't kept."); return; }
    }
    if (audio.paused) { audio.play(); btn.replaceChildren(icon("pause", 20)); }
    else { audio.pause(); btn.replaceChildren(icon("play", 20)); }
  });
  audio.addEventListener("ended", () => btn.replaceChildren(icon("play", 20)));
  return [h("div", { class: "row" }, btn,
    h("div", { class: "grow" }, h("div", { class: "rt" }, title), h("div", { class: "rd" }, desc)),
    h("span", { class: "t" }, length)), audio];
}

export async function render({ id }) {
  const [record, { people }] = await Promise.all([api(`api/sessions/${id}`), api("api/people")]);
  const stats = record.stats || {};
  const speaker = stats.speaker || {};
  const person = people.find((p) => p.id === speaker.id);
  const color = person?.color || "var(--primary)";
  const total = stats.audio_s || 1;
  const where = record.device_name || record.device_id || "Unknown kiosk";
  const end = (record.events || []).find((e) => e.type === "end_of_speech");
  const sent = end && end.reason !== "silence" ? end.t : total;

  const wave = h("div", { class: "wave", role: "img", "aria-label": "Waveform of the session" });
  let audios = [];
  const follow = record.mode === "follow";
  if (record.has_audio) {
    audioBytes(`api/sessions/${id}/audio/raw`).then((buf) => wave.replaceChildren(...(follow
      ? followWave(pcm(buf), total, record.released || [], color)
      : waveform(pcm(buf), total, record.events || [], color)))).catch(() => {});
  }
  const marker = end && !follow ? h("div", { class: "mk", style: { left: `${(sent / total) * 100}%` } }, h("span", { class: "mklbl" }, `Stopped, ${secs(sent)}`)) : null;

  const who = h("select", { class: "fld", "aria-label": "Person" }, people.map((p) => h("option", { value: p.id, selected: p.id === speaker.id }, p.name)));
  const add = async () => {
    try {
      const r = await api(`api/sessions/${id}/enroll`, { method: "POST", json: { person_id: who.value } });
      toast(`Added ${r.added_s} s of speech to ${r.name}`);
    } catch (err) { toast(err.message); }
  };
  const remove = async () => {
    if (!await ask({ title: "Delete this session?", text: "Its audio is deleted too.", ok: "Delete", danger: true })) return;
    await api(`api/sessions/${id}`, { method: "DELETE" });
    location.hash = "#/sessions";
  };
  // Everything about this session in one zip, for an issue report.
  const download = async () => {
    try {
      const a = h("a", { href: await audioUrl(`api/sessions/${id}/download`), download: `ground-control-session-${id}.zip` });
      a.click();
    } catch (err) { toast(`Couldn't download the session: ${err.message}`); }
  };

  let audioSection = [h("div", { class: "empty" }, "Audio keeping was off for this session.")];
  if (record.has_audio) {
    const [rawRow, rawAudio] = player(`api/sessions/${id}/audio/raw`, "As heard", "As heard by the kiosk's microphone.", secs(total));
    const [cleanRow, cleanAudio] = follow
      ? player(`api/sessions/${id}/audio/clean`, "Sent to the assistant",
        stats.isolated_ms !== undefined ? "Only the user's speech: other voices left out between their words and removed under them."
          : "Only the user's speech, other voices left out.",
        secs(stats.released_s ?? 0))
      : player(`api/sessions/${id}/audio/clean`, record.mode === "gated" ? "Sent to the assistant" : "Cleaned",
        "Other voices replaced with silence.", secs(sent));
    audioSection = [rawRow, cleanRow];
    audios = [rawAudio, cleanAudio];
  }
  if (record.wake_s) {
    // Kept apart from the rest: the kiosk heard it before the session began.
    const [wakeRow, wakeAudio] = player(`api/sessions/${id}/audio/wake`, "Wake word",
      "The wake word as the kiosk heard it. Whoever says it is the user for this command, and it's the first sample of their voice.", secs(record.wake_s));
    audioSection = [wakeRow, ...(record.has_audio ? audioSection : [])];
    audios = [wakeAudio, ...audios];
  }

  return {
    nodes: [
      shell("sessions", {
        back: "#/sessions", title: `${where}, ${clock(record.started)}`,
        actions: [
          h("button", { class: "ib", "aria-label": "Download the session", title: "Download the session: its audio, details and log", onclick: download }, icon("download")),
          h("button", { class: "ib", "aria-label": "Delete session", onclick: remove }, icon("trash")),
        ],
      }),
      h("main", { class: "page" },
        h("div", { class: "chips" },
          speaker.name ? chip(speaker.score != null ? `${speaker.name}, ${speaker.score}` : speaker.name, "big", h("span", { class: "dot", style: { background: color, width: "10px", height: "10px" } }))
            : chip(`Unknown voice${speaker.score ? `, ${speaker.score}` : ""}`, "big"),
          outcomeChip(stats),
          chip(`${modeLabel(record.mode)} mode`, "big"),
          chip(`Answers: ${answersLabel(record, people)}`, "big"),
          chip(`${record.model === "light" ? "Light" : "Accurate"} model`, "big")),
        section("Timeline"),
        h("div", { class: "card tl" },
          follow
            ? h("div", { class: "legend" },
              h("span", {}, h("i", { class: "swb", style: { background: color } }), `${speaker.name || "The user"}, sent to the assistant`),
              h("span", {}, h("i", { class: "swb", style: { background: "var(--other)", opacity: 0.45 } }), "Left out"))
            : h("div", { class: "legend" },
              h("span", {}, h("i", { class: "swb", style: { background: color, opacity: 0.55 } }), "Before speech"),
              h("span", {}, h("i", { class: "swb", style: { background: color } }), speaker.name || "Speaker"),
              h("span", {}, h("i", { class: "swb", style: { background: "var(--other)" } }), "Other voice, left out"),
              h("span", {}, h("i", { class: "swb", style: { background: "var(--other)", opacity: 0.3 } }), "After the end")),
          h("div", { style: { position: "relative", marginLeft: "30px" } }, wave, marker,
            (record.followups || []).map((t) => h("div", { class: "mk", style: { left: `${(t / total) * 100}%` } },
              h("span", { class: "mklbl" }, "Follow-up")))),
          chart(record, total, color),
          h("div", { class: "axis" }, h("span", {}, "0 s"), h("span", {}, secs(total / 2)), h("span", {}, secs(total))),
          chartKey(record, color)),
        h("div", { class: "cols" },
          h("section", {}, section("Steps"), h("div", { class: "card" }, steps(record),
            h("div", { class: "row" }, h("div", { class: "grow" }, h("div", { class: "rt" }, "Result"), h("div", { class: "rd" }, `${outcome(stats)}. ${stats.checks ?? 0} checks, ${stats.cpu_ms ?? 0} ms of CPU.${stats.isolated_ms !== undefined ? ` Isolation: ${stats.isolated_ms} ms.` : ""}`))))),
          h("section", {},
            section("Audio"), h("div", { class: "card" }, audioSection),
            section("Use this voice"),
            h("div", { class: "card" },
              h("div", { class: "row" },
                h("div", { class: "grow" }, h("div", { class: "rt" }, "Add to a person"), h("div", { class: "rd" }, follow ? "The user's speech is added, without the voices left out." : "Only the speech before another voice is added.")),
                who, h("button", { class: "btn out", onclick: add, disabled: !record.has_audio || !people.length }, "Add")))))),
    ],
    cleanup: () => audios.forEach((a) => { a.pause(); if (a.src) URL.revokeObjectURL(a.src); }),
  };
}
