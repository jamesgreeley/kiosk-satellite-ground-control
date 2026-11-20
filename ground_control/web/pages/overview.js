import { api } from "../api.js";
import { shell } from "../app.js";
import { ago, chip, disc, h, icon, kioskAnswers, kioskChip, outcome, pageHead, section } from "../ui.js";

export async function render() {
  const [status, { sessions }, { kiosks }, { people }] = await Promise.all([
    api("api/status"), api("api/sessions?limit=500"), api("api/kiosks"), api("api/people"),
  ]);
  const byId = Object.fromEntries(people.map((p) => [p.id, p]));
  const day = Date.now() / 1000 - 86400;
  const recent = sessions.filter((s) => s.started >= day);
  const count = (end) => recent.filter((s) => s.end === end).length;
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
  const feature = (ico, title, desc, status) => h("a", { class: "row nowrap", href: "#/settings/voice-match" },
    h("span", { class: "ico" }, icon(ico, 24)),
    h("div", { class: "grow" }, h("div", { class: "rt" }, title), h("div", { class: "rd" }, desc)),
    status);

  const sessionRows = sessions.length
    ? sessions.slice(0, 12).map((s) => {
        const person = s.speaker?.id ? byId[s.speaker.id] || s.speaker : null;
        const who = person?.name || "Unknown voice";
        const where = s.device_name ? ` in the ${s.device_name}` : "";
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
        waiting.length ? h("div", { class: "card", style: { marginBottom: "16px" } },
          ...waiting.map((k) => h("a", { class: "row nowrap", href: `#/kiosks/${encodeURIComponent(k.device_id)}` },
            h("span", { class: "ico" }, icon("screen", 24)),
            h("div", { class: "grow" }, h("div", { class: "rt" }, `${k.name} is waiting for approval`),
              h("div", { class: "rd" }, "Approve it on its kiosk page to let it use Ground Control.")),
            chip("Review", "c-warn")))) : null,
        h("div", { class: "card stats" },
          h("div", { class: "stat" }, h("div", { class: "statnum" }, recent.length), h("div", { class: "lbl" }, "Commands in the last 24 hours")),
          h("div", { class: "stat" }, h("div", { class: "statnum" }, count("other_voice")), h("div", { class: "lbl" }, "Ended early when another voice started")),
          h("div", { class: "stat" }, h("div", { class: "statnum" }, count("rejected")), h("div", { class: "lbl" }, "Rejected, not a voice the kiosk answers"))),
        h("div", { class: "cols" },
          h("section", {},
            section("Kiosks"),
            h("div", { class: "card" }, kioskRows,
              kiosks.length ? h("div", { class: "hint", style: { paddingTop: "14px" } }, icon("info", 18),
                "Who a kiosk answers is set on the kiosk, in Kiosk Satellite under Voice Satellite, Ground Control Voice Match, Answer.") : null),
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
              feature("person", "Recognition", `Who is speaking. ${status.model_label} model, threshold ${status.threshold}.`,
                status.rebuilding ? chip("Updating voiceprints", "c-warn") : chip("On", "c-ok")),
              feature("cut", "Trimming", "Other voices cut out between the user's words. Each kiosk turns it on in Kiosk Satellite.",
                trimChip),
              feature("mark", "Isolation", "Other voices removed under the user's words.",
                status.isolation ? chip(status.isolation_on_gpu ? "On, GPU" : "On, CPU", "c-ok") : chip("Off"))),
            section("Vision Match"),
            h("div", { class: "card" },
              h("div", { class: "row nowrap" },
                h("span", { class: "ico" }, icon("eye", 24)),
                h("div", { class: "grow" }, h("div", { class: "rt" }, "Coming soon"))))),
          h("section", {},
            section("Recent sessions", null, h("a", { class: "txtlink", href: "#/sessions" }, "All sessions")),
            h("div", { class: "card" }, sessionRows)))),
    ],
  };
}
