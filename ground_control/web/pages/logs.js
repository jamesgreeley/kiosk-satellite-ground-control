// Logs: what the service has been doing since it started, newest first,
// with the report to attach to an issue.
import { api } from "../api.js";
import { shell } from "../app.js";
import { h, icon, pageHead, toast, toggle } from "../ui.js";

const POLL_MS = 2000;
const SHOWN = 1000;
const LEVELS = [["DEBUG", "All"], ["INFO", "Info"], ["WARNING", "Warnings"], ["ERROR", "Errors"]];
const RANK = { DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50 };
const LABEL = { DEBUG: "Debug", INFO: "Info", WARNING: "Warning", ERROR: "Error", CRITICAL: "Critical" };

function time(t) {
  const d = new Date(t * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

function day(t) {
  return new Date(t * 1000).toLocaleDateString([], { month: "short", day: "numeric" });
}

/** Where a line came from, without the package name every one of them starts with. */
function where(logger) {
  return logger === "ground_control" ? "main" : logger.replace(/^ground_control\./, "");
}

function asText(e) {
  const t = new Date(e.t * 1000);
  const stamp = `${t.toLocaleDateString("en-CA")} ${time(e.t)}`;
  return `${stamp} ${e.level.padEnd(7)} ${e.logger}: ${e.message}${e.exc ? `\n${e.exc}` : ""}`;
}

export async function render() {
  const first = await api("api/logs");
  const entries = first.entries;
  let last = first.last;
  const state = { level: "INFO", q: "" };

  const list = h("div", { class: "logs", role: "log", "aria-live": "off" });
  const count = h("span");

  function shown() {
    const q = state.q.toLowerCase();
    const floor = RANK[state.level];
    const out = [];
    for (let i = entries.length - 1; i >= 0 && out.length < SHOWN; i -= 1) {
      const e = entries[i];
      if ((RANK[e.level] ?? 20) < floor) continue;
      if (q && !`${e.logger} ${e.message} ${e.exc || ""}`.toLowerCase().includes(q)) continue;
      out.push(e);
    }
    return out;
  }

  function row(e) {
    const lv = e.level.toLowerCase();
    return h("div", { class: `logrow lv-${lv}` },
      h("span", { class: "lt", title: day(e.t) }, time(e.t)),
      h("span", { class: `lvl lv-${lv}` }, LABEL[e.level] || e.level),
      h("span", { class: "lw" }, where(e.logger)),
      h("span", { class: "lm" }, e.message,
        e.exc ? h("details", {}, h("summary", {}, "Details"), h("pre", {}, e.exc)) : null));
  }

  function draw() {
    const rows = shown();
    list.replaceChildren(...(rows.length ? rows.map(row)
      : [h("div", { class: "empty" }, state.q ? "No lines match." : "Nothing at this level yet.")]));
    count.textContent = rows.length === SHOWN ? `Newest ${SHOWN} lines` : `${rows.length} line${rows.length === 1 ? "" : "s"}`;
  }

  // New lines every few seconds, while the page is open and visible.
  let timer = null;
  let gone = false;
  async function poll() {
    if (gone) return;
    if (!document.hidden) {
      try {
        const got = await api(`api/logs?after=${last}`);
        if (got.entries.length) {
          entries.push(...got.entries);
          if (entries.length > 5000) entries.splice(0, entries.length - 5000);
          draw();
        }
        last = got.last;
      } catch {
        // The service restarting: try again on the next round.
      }
    }
    if (!gone) timer = setTimeout(poll, POLL_MS);
  }

  const seg = h("div", { class: "seg", role: "group", "aria-label": "Show" },
    LEVELS.map(([k, label]) => h("button", {
      class: k === state.level ? "on" : "",
      onclick: (e) => {
        state.level = k;
        seg.querySelectorAll("button").forEach((b) => b.classList.remove("on"));
        e.currentTarget.classList.add("on");
        draw();
      },
    }, label)));
  const search = h("input", { type: "search", placeholder: "Search the log", "aria-label": "Search the log" });
  search.addEventListener("input", () => { state.q = search.value; draw(); });

  const detailed = toggle(first.detailed, "Detailed logs", async (on) => {
    try {
      await api("api/logs", { method: "PATCH", json: { detailed: on } });
      toast(on ? "Detailed logs on until Ground Control restarts" : "Detailed logs off");
      if (on && RANK[state.level] > RANK.DEBUG) seg.querySelector("button").click();
    } catch (err) {
      toast(`Not changed: ${err.message}`);
    }
  });

  async function download() {
    try {
      const r = await api("api/logs/report");
      const name = /filename="([^"]+)"/.exec(r.headers.get("Content-Disposition") || "")?.[1] || "ground-control-logs.txt";
      const url = URL.createObjectURL(await r.blob());
      const a = h("a", { href: url, download: name });
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (err) {
      toast(`Couldn't download the logs: ${err.message}`);
    }
  }

  async function copy() {
    const text = shown().slice().reverse().map(asText).join("\n");
    try {
      await navigator.clipboard.writeText(text);
      toast("Copied the lines shown");
    } catch {
      toast("Couldn't copy here. Use Download instead.");
    }
  }

  draw();
  timer = setTimeout(poll, POLL_MS);

  return {
    nodes: [
      shell("logs"),
      h("main", { class: "page" },
        pageHead("Logs", h("div", { class: "pactions" },
          h("button", { class: "ib", "aria-label": "Copy the lines shown", title: "Copy the lines shown", onclick: copy }, icon("copy")),
          h("button", { class: "btn fill sm", onclick: download }, icon("download", 18), "Download"))),
        h("p", { class: "note lognote" },
          "When you report an issue, download the logs and attach them. They include the version, the speaker model and every kiosk but never a token. For a problem you can repeat, turn on Detailed logs first and then repeat it."),
        h("div", { class: "filters" }, seg, h("label", { class: "search" }, icon("search", 20), search),
          h("label", { class: "swlabel" }, detailed, "Detailed logs")),
        h("div", { class: "card logcard" }, list,
          h("div", { class: "tfoot" }, count,
            h("span", { class: "sub" }, `Kept since Ground Control started on ${day(first.started)} at ${time(first.started)}`)))),
    ],
    cleanup: () => { gone = true; clearTimeout(timer); },
  };
}
