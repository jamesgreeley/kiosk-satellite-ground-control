import { api } from "../api.js";
import { shell } from "../app.js";
import { ago, chip, disc, h, icon, kioskAnswers, kioskChip, outcome, pageHead, section } from "../ui.js";

const modelName = (key) => (key === "accurate" ? "Accurate" : "Light");

/** The Face Detection and Hand Gestures models, as one name when they match. */
const visionModels = (settings) => {
  const m = settings.vision_models || {};
  return m.face === m.hands ? `${modelName(m.face)} models.`
    : `${modelName(m.face)} Face Detection and ${modelName(m.hands)} Hand Gestures.`;
};

export async function render() {
  const [status, { sessions }, latest, counts, { kiosks }, { people }, settings] = await Promise.all([
    api("api/status"), api("api/sessions?kind=voice&limit=500"), api("api/sessions?limit=12"),
    api("api/sessions/counts"), api("api/kiosks"), api("api/people"), api("api/settings"),
  ]);
  const byId = Object.fromEntries(people.map((p) => [p.id, p]));
  const voice = counts.voice || {};
  const vision = counts.vision || {};
  const last = (id) => sessions.find((s) => s.device_id === id);
  const answers = (k) => {
    const label = kioskAnswers(k, last(k.device_id), people);
    return label ? `Answers: ${label}.` : "";
  };
  const waiting = kiosks.filter((k) => k.access === "pending");

  const kioskRows = kiosks.length
    ? kiosks.map((k) => h("a", { class: "row", href: `#/kiosks/${encodeURIComponent(k.device_id)}` },
        h("span", { class: "ico" }, icon("screen", 24)),
        h("div", { class: "grow" },
          h("div", { class: "rt" }, k.name),
          h("div", { class: "rd" }, [k.model ? `${k.model}.` : "", answers(k),
            k.connected ? (last(k.device_id) ? "" : "No sessions yet.") : `Last seen ${ago(k.last_seen)}.`].filter(Boolean).join(" "))),
        kioskChip(k)))
    : [h("div", { class: "empty" }, "No kiosks yet. Kiosks show up here once Kiosk Satellite connects.")];

  // Trimming is turned on per kiosk, in Kiosk Satellite. A kiosk says so, or its sessions run in follow mode.
  const approved = kiosks.filter((k) => k.access === "approved");
  const trims = (k) => (k.features?.trimming ?? null) !== null ? k.features.trimming : last(k.device_id)?.mode === "follow";
  const trimming = approved.filter(trims).length;
  const trimChip = !approved.length ? null
    : trimming === approved.length ? chip("On", "c-ok")
    : trimming ? chip(`${trimming} of ${approved.length} kiosks`, "c-info") : chip("Off");
  const feature = (ico, title, desc, status, area = "voice-match") => h("a", { class: "row nowrap", href: `#/settings/${area}` },
    h("span", { class: "ico" }, icon(ico, 24)),
    h("div", { class: "grow" }, h("div", { class: "rt" }, title), h("div", { class: "rd" }, desc)),
    status);

  const sessionRows = latest.sessions.length
    ? latest.sessions.map((s) => {
        const where = s.device_name ? ` in the ${s.device_name}` : "";
        if (s.kind === "vision") {
          return h("a", { class: "row", href: `#/sessions/${s.id}` },
            h("span", { class: "disc vis" }, icon("eye", 20)),
            h("div", { class: "grow" },
              h("div", { class: "rt" }, `Vision Match${where}`),
              h("div", { class: "rd" }, `${[...(s.summary || []), `${s.duration_s ?? 0} s`].join(". ")}.`)),
            h("span", { class: "time" }, ago(s.started)));
        }
        const person = s.speaker?.id ? byId[s.speaker.id] || s.speaker : null;
        const who = person?.name || "Unknown voice";
        return h("a", { class: "row", href: `#/sessions/${s.id}` },
          disc(person),
          h("div", { class: "grow" },
            h("div", { class: "rt" }, `${who}${where}`),
            h("div", { class: "rd" }, `${outcome(s)}. ${s.audio_s ?? 0} s.`)),
          h("span", { class: "time" }, ago(s.started)));
      })
    : [h("div", { class: "empty" }, "No sessions yet.")];

  const where = status.on_gpu ? "Running on the GPU" : "Running on the CPU";
  return {
    nodes: [
      shell("overview"),
      h("main", { class: "page" },
        pageHead("Overview"),
        status.update ? h("div", { class: "card", style: { marginBottom: "16px" } },
          h("div", { class: "row" },
            h("span", { class: "ico" }, icon("download", 24)),
            h("div", { class: "grow" },
              h("div", { class: "rt" }, `Ground Control ${status.update.version} is available`),
              h("div", { class: "rd" }, "To update, pull the new image and recreate the container.")),
            h("a", { class: "btn out sm", href: status.update.changelog_url, target: "_blank", rel: "noopener" }, "What's new"),
            h("a", { class: "btn out sm", href: status.update.repo_url, target: "_blank", rel: "noopener" }, icon("github", 18), "GitHub"))) : null,
        waiting.length ? h("div", { class: "card", style: { marginBottom: "16px" } },
          ...waiting.map((k) => h("a", { class: "row nowrap", href: `#/kiosks/${encodeURIComponent(k.device_id)}` },
            h("span", { class: "ico" }, icon("screen", 24)),
            h("div", { class: "grow" }, h("div", { class: "rt" }, `${k.name} is waiting for approval`),
              h("div", { class: "rd" }, "Approve it on its kiosk page to let it use Ground Control.")),
            chip("Review", "c-warn")))) : null,
        h("div", { class: "card stats" },
          h("div", { class: "stat" }, h("div", { class: "statnum" }, voice.total ?? 0), h("div", { class: "lbl" }, "Commands in the last 24 hours")),
          h("div", { class: "stat" }, h("div", { class: "statnum" }, voice.other_voice ?? 0), h("div", { class: "lbl" }, "Ended early when another voice started")),
          h("div", { class: "stat" }, h("div", { class: "statnum" }, voice.rejected ?? 0), h("div", { class: "lbl" }, "Rejected because the kiosk doesn't answer this voice")),
          h("div", { class: "stat" }, h("div", { class: "statnum" }, voice.false_positive ?? 0), h("div", { class: "lbl" }, "False positives stopped by the Wake Word Verifier")),
          h("div", { class: "stat" }, h("div", { class: "statnum" }, vision.total ?? 0), h("div", { class: "lbl" }, "Vision sessions in the last 24 hours"))),
        h("div", { class: "cols" },
          h("section", {},
            section("Kiosks"),
            h("div", { class: "card" }, kioskRows,
              kiosks.length ? h("div", { class: "hint", style: { paddingTop: "14px" } }, icon("info", 18),
                "Who a kiosk answers is set in Kiosk Satellite on each kiosk under Ground Control > Voice Match > Answer.") : null),
            section("Service"),
            h("div", { class: "card" },
              h("div", { class: "row" },
                h("span", { class: "ico" }, icon("chip", 24)),
                h("div", { class: "grow" },
                  h("div", { class: "rt" }, where),
                  h("div", { class: "rd" }, `${status.threads} threads. ${status.sessions} sessions right now. Version ${status.version}.`)),
                chip("Ready", "c-ok")),
            ),
            section("Voice Match"),
            h("div", { class: "card" },
              feature("person", "Recognition", `Who is speaking. The ${status.model_label} model with a threshold of ${status.threshold}.`,
                status.rebuilding ? chip("Updating voiceprints", "c-warn") : chip("On", "c-ok")),
              feature("cut", "Trimming", "Other voices cut out between the user's words. Each kiosk turns it on in Kiosk Satellite.",
                trimChip),
              feature("mark", "Isolation", "Other voices removed under the user's words.",
                status.isolation ? chip(status.isolation_on_gpu ? "On, GPU" : "On, CPU", "c-ok") : chip("Off"))),
            section("Vision Match"),
            h("div", { class: "card" },
              feature("eye", "Face Detection and Hand Gestures",
                `Faces looking at the kiosk and raised fingers. ${visionModels(settings)}`,
                settings.vision_error ? chip("Can't run", "c-err")
                  : settings.vision_loaded ? chip(settings.vision_on_gpu ? "On, GPU" : "On, CPU", "c-ok") : chip("Not loaded yet"),
                "vision-match"),
              feature("person", "Face Recognition", `Who is looking at each kiosk. It shows in each kiosk's Face sensor. ${modelName(settings.vision_models?.recognition)} model.`,
                settings.faces_enrolled ? chip(`${settings.faces_enrolled} ${settings.faces_enrolled === 1 ? "face" : "faces"}`, "c-ok") : chip("No faces yet"),
                "vision-match"))),
          h("section", {},
            section("Recent sessions", null, h("a", { class: "txtlink", href: "#/sessions" }, "All sessions")),
            h("div", { class: "card" }, sessionRows)))),
    ],
  };
}
