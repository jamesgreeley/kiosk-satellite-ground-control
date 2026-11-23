// Settings: one page per area, picked from a list on the left like People and
// Kiosks. Inside a page every group is the same: a section title with a line
// under it, then a card of rows.
import { api, loggedIn, logOut } from "../api.js";
import { shell } from "../app.js";
import { chip, h, icon, pageHead, section, toast, toggle } from "../ui.js";

const TEMPLATE = [
  "{%- if llm_context.device_id %}",
  "{%- set speaker_entity = device_entities(llm_context.device_id) | select('match', 'sensor\\..*_speaker$') | first | default('') %}",
  "{%- if speaker_entity %}",
  "The person speaking is in the \"{{ state_attr(speaker_entity, 'friendly_name') }}\" sensor. It changes from one request to the next, so whenever the answer depends on who is speaking (their name, \"my\", \"me\", \"mine\"), call GetLiveContext first and read that sensor. Never reuse a name from earlier in the conversation. If the sensor is empty, unknown or unavailable, nobody was identified.",
  "{%- endif %}",
  "{%- endif %}",
].join("\n");

const AREAS = [
  ["voice-match", "Voice Match", "mark", "Recognition, Trimming, Wake Word Aligner and Isolation"],
  ["vision-match", "Vision Match", "eye", "Coming soon"],
  ["general", "General", "tune", "Engine, sessions, agents and kiosks"],
];

async function copy(text) {
  try { await navigator.clipboard.writeText(text); }
  catch {
    // Inside the Home Assistant panel's frame the clipboard API can be off.
    const area = h("textarea", { style: { position: "fixed", opacity: "0" } });
    area.value = text;
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  toast("Copied");
}

function row(title, desc, control, id) {
  return h("div", { class: "row" },
    h("div", { class: "grow" }, h("div", { class: "rt" }, id ? h("label", { for: id }, title) : title), h("div", { class: "rd" }, desc)),
    control);
}

const hint = (text) => h("div", { class: "hint" }, icon("info", 18), text);

function areas(current) {
  return h("div", { class: "card" },
    AREAS.map(([key, label, ico, desc]) => h("a", {
      class: `row nowrap ${key === current ? "sel" : ""}`, href: `#/settings/${key}`, "aria-current": key === current ? "page" : null,
    }, h("span", { class: "ico" }, icon(ico, 24)),
    h("div", { class: "grow" }, h("div", { class: "rt" }, label), h("div", { class: "rd" }, desc)))));
}

export async function render({ area } = {}) {
  const current = AREAS.some(([key]) => key === area) ? area : "voice-match";
  const s = await api("api/settings");
  let poll = null;

  const save = async (body) => {
    try { await api("api/settings", { method: "PATCH", json: body }); toast("Saved"); }
    catch (err) { toast(err.message); }
  };
  const number = (id, value, key) => {
    const input = h("input", { class: "fld num", id, value: String(value), inputmode: "decimal" });
    input.addEventListener("change", () => save({ [key]: Number(input.value) }));
    return input;
  };

  // ── Voice Match ──────────────────────────────────────────────────────────

  const voiceMatch = () => {
    const progress = h("div", { class: "row", role: "status", style: { display: s.rebuilding ? "" : "none" } },
      h("div", { class: "grow" }, h("div", { class: "rt" }, "Updating voiceprints"),
        h("div", { class: "rd" }, "Kiosks keep using the current model until this finishes.")));
    const option = (key, label, desc) => {
      const b = h("button", { class: "opt", role: "radio", "aria-checked": String(s.model === key) },
        h("span", { class: "radio" }),
        h("span", { class: "grow" },
          h("span", { class: "rt", style: { display: "block" } }, label, key === "accurate" ? h("span", { class: "chip c-info", style: { marginLeft: "8px", height: "22px" } }, "Recommended") : null),
          h("span", { class: "rd", style: { display: "block" } }, desc)));
      b.addEventListener("click", async () => {
        if (s.model === key) return;
        const r = await api("api/settings", { method: "PATCH", json: { model: key } });
        progress.style.display = r.rebuilding ? "" : "none";
        document.querySelectorAll(".opt").forEach((o) => o.setAttribute("aria-checked", "false"));
        b.setAttribute("aria-checked", "true");
        clearInterval(poll);
        poll = setInterval(async () => {
          const now = await api("api/settings");
          if (!now.rebuilding) {
            clearInterval(poll);
            s.model = now.model;
            progress.style.display = "none";
            toast(`Now using ${now.models[now.model]}`);
            location.reload();
          }
        }, 1000);
      });
      return b;
    };

    const usable = s.isolator_available && s.model === s.isolator_model;
    const where = h("span", {});
    const showWhere = (onGpu) => {
      where.replaceChildren(onGpu === null || onGpu === undefined ? h("span", { class: "val" }, "Off")
        : chip(onGpu ? "NVIDIA GPU (CUDA)" : "CPU", onGpu ? "c-ok" : ""));
    };
    showWhere(s.isolator ? s.isolator_on_gpu : null);
    const isolation = toggle(s.isolator && usable, "Use Isolation", async (v) => {
      await save({ isolator: v });
      const now = await api("api/settings");
      showWhere(now.isolator ? now.isolator_on_gpu : null);
    });
    if (!usable) isolation.disabled = true;
    const isolationNote = !s.isolator_available ? "The isolation model is not in this image."
      : s.model !== s.isolator_model ? "Isolation needs the Accurate speaker model."
      : "Each command waits for Isolation once the user stops talking: about a tenth of a second on an NVIDIA GPU and about a second on a fast x86 CPU. It is too slow on a Raspberry Pi.";

    const aligner = toggle(s.aligner && !s.aligner_missing, "Use Wake Word Aligner", (v) => save({ aligner: v }));
    if (s.aligner_missing) aligner.disabled = true;
    const alignerNote = s.aligner_missing === "model" ? "The Wake Word Aligner model is not in this install."
      : s.aligner_missing ? "The Wake Word Aligner needs espeak-ng, which is not installed. The Docker images and the Home Assistant app have it."
      : "It works with any wake word and any wake word engine. It doesn't delay the command.";

    return [
      section("Recognition", "Who is speaking, from the voiceprints of the people enrolled."),
      h("div", { class: "card" },
        h("div", { role: "radiogroup", "aria-label": "Speaker model" },
          row("Speaker model", "Voiceprints are rebuilt from the enrollment audio when this changes. Nobody has to record again.", null),
          option("accurate", "Accurate", "Fewest mix-ups between people and with other voices. Best on an x86 CPU or a GPU."),
          option("light", "Light", "About a third of the CPU, for a Raspberry Pi or Home Assistant Green. Another voice cutting in ends the turn about a second later."),
          progress),
        row("Threshold", `How close a voice has to be to count as someone enrolled. Higher means fewer mix-ups and more unknown voices. Each model has its own, ${s.default_threshold} by default.`,
          number("thr", s.threshold, "threshold"), "thr"),
        row("Learn from commands", "Confident commands fold into the speaker's voice, so it follows colds and room changes.",
          toggle(s.learn, "Learn from commands", (v) => save({ learn: v })))),

      section("Trimming", "Other voices are cut out between the user's words, and long pauses are shortened."),
      h("div", { class: "card" },
        row("Other voice threshold", `Two checks in a row under this score mean someone else is talking. ${s.default_other_threshold} by default.`,
          number("oth", s.other_threshold, "other_threshold"), "oth"),
        hint("Each kiosk turns Trimming on or off in Kiosk Satellite, under Voice Satellite, Ground Control Voice Match.")),

      section("Wake Word Aligner", "The end of the wake word is left out of the command, like the \"xa\" of \"Alexa\"."),
      h("div", { class: "card" },
        row("Use Wake Word Aligner", "For kiosks with Trimming and Talk right after the wake word on in Kiosk Satellite.", aligner),
        hint(alignerNote)),

      section("Isolation", "Other voices talking over the user's words are removed too, using the user's voiceprint."),
      h("div", { class: "card" },
        row("Use Isolation", "For kiosks with Trimming and Isolation on in Kiosk Satellite.", isolation),
        row("Runs on", "Set by the CUDA option, like the speaker model.", where),
        hint(isolationNote)),
    ];
  };

  // ── Vision Match ─────────────────────────────────────────────────────────

  const visionMatch = () => [
    h("div", { class: "card" },
      h("div", { class: "soon" },
        h("span", { class: "soonicon" }, icon("eye", 32)),
        h("div", { class: "rt" }, "Coming soon"))),
  ];

  // ── General ──────────────────────────────────────────────────────────────

  const host = `${location.hostname}:${s.port}`;
  const general = () => [
    section("Engine", "Where voices are checked."),
    h("div", { class: "card" },
      row("Runs on", s.cuda && !s.on_gpu ? "CUDA is on, but no GPU is usable, so the CPU is used." : "Set by the CUDA option.",
        chip(s.on_gpu ? "NVIDIA GPU (CUDA)" : "CPU", s.on_gpu ? "c-ok" : "")),
      row("Threads", "Threads per check. Set in the app's options.", h("span", { class: "val" }, String(s.threads))),
      hint("CUDA is available in the Docker image on a machine with an NVIDIA GPU.")),

    section("Sessions", "Saved sessions, for the Sessions page."),
    h("div", { class: "card" },
      row("Keep sessions for", "Older sessions and their audio are deleted.",
        (() => {
          const sel = h("select", { class: "fld", id: "keep" },
            [1, 7, 30, 90].map((d) => h("option", { value: d, selected: d === s.keep_days }, `${d} day${d === 1 ? "" : "s"}`)));
          sel.addEventListener("change", () => save({ keep_days: Number(sel.value) }));
          return sel;
        })(), "keep"),
      row("Keep audio", "Needed to replay a session or add it to a person.",
        toggle(s.keep_audio, "Keep audio", (v) => save({ keep_audio: v })))),

    section("Conversation agents", "Tells an LLM agent in Home Assistant who is speaking. One copy works for every kiosk."),
    h("div", { class: "card" },
      row("Speaker in the agent's instructions", "Paste at the end of the agent's instructions in Home Assistant.",
        h("button", { class: "ib", "aria-label": "Copy the template", onclick: () => copy(TEMPLATE) }, icon("copy"))),
      h("pre", { class: "code" }, TEMPLATE),
      hint("The agent needs the Assist API, and each kiosk's Speaker sensor exposed to Assist.")),

    section("Kiosks", "Kiosks on the same network find the service on their own. Each signs in with its Home Assistant token, so there is nothing else to enter."),
    h("div", { class: "card" },
      row("Address", "For a kiosk on another network, enter this in Kiosk Satellite.",
        h("span", { class: "cell" }, h("span", { class: "val" }, host),
          h("button", { class: "ib", "aria-label": "Copy the address", onclick: () => copy(host) }, icon("copy", 20))))),

    // Standalone only: the browser logged in with Home Assistant.
    loggedIn() ? section("This browser") : null,
    loggedIn() ? h("div", { class: "card" },
      row("Signed in with Home Assistant", "Signing out asks Home Assistant to end this sign-in.",
        h("button", { class: "btn out sm", onclick: logOut }, "Sign out"))) : null,
  ];

  const content = { "voice-match": voiceMatch, "vision-match": visionMatch, general }[current]();
  return {
    nodes: [
      shell("settings"),
      h("main", { class: "page" },
        pageHead("Settings"),
        h("div", { class: "split" },
          h("section", { class: "left", "aria-label": "Settings" }, areas(current)),
          h("section", { class: "right", "aria-label": AREAS.find(([key]) => key === current)[1] }, content))),
    ],
    cleanup: () => clearInterval(poll),
  };
}
