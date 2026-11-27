// The shell: top bar, tabs, routing and sign-in.
import { api, HomeAssistantDown, logIn, SignInNeeded } from "./api.js";
import { h, icon } from "./ui.js";

const LINKS = [
  ["https://github.com/jxlarrea/kiosk-satellite-ground-control", "GitHub", "github"],
  ["https://buymeacoffee.com/jxlarrea", "Buy me a coffee", "coffee"],
];

const TABS = [
  ["#/", "Overview", "overview"],
  ["#/people", "People", "people"],
  ["#/kiosks", "Kiosks", "kiosks"],
  ["#/sessions", "Sessions", "sessions"],
  ["#/settings", "Settings", "settings"],
  ["#/logs", "Logs", "logs"],
];

const PAGES = {
  overview: () => import("./pages/overview.js"),
  people: () => import("./pages/people.js"),
  enroll: () => import("./pages/enroll.js"),
  sessions: () => import("./pages/sessions.js"),
  session: () => import("./pages/session.js"),
  settings: () => import("./pages/settings.js"),
  kiosks: () => import("./pages/kiosks.js"),
  logs: () => import("./pages/logs.js"),
};

let cleanup = null;

function route() {
  const parts = (location.hash || "#/").slice(2).split("/").filter(Boolean).map(decodeURIComponent);
  const [first, second] = parts;
  if (!first) return ["overview", {}];
  if (first === "people") return ["people", { id: second }];
  if (first === "enroll") return ["enroll", { id: second, device: parts[2] }];
  if (first === "sessions") return second ? ["session", { id: second }] : ["sessions", {}];
  if (first === "settings") return ["settings", { area: second }];
  if (first === "logs") return ["logs", {}];
  // The kiosk pages used to be at #/live/<device>.
  if (first === "kiosks" || first === "live") return ["kiosks", { id: second }];
  return ["overview", {}];
}

/** The top bar: tabs on the main pages, a back arrow and a title on detail pages. */
export function shell(tab, { title, back, actions = [] } = {}) {
  const bar = back
    ? h("header", { class: "bar detail" },
        h("a", { class: "ib", href: back, "aria-label": "Back" }, icon("back")),
        h("span", { class: "apptitle" }, title),
        actions)
    : h("header", { class: "bar" },
        h("a", { class: "brand", href: "#/", "aria-label": "Kiosk Satellite Ground Control, Overview" },
          h("img", { class: "brandmark", src: "static/mark.svg", alt: "" }),
          h("span", { class: "brandtext" },
            h("span", { class: "bt1" }, "Kiosk Satellite"),
            h("span", { class: "bt2" }, "Ground Control"))),
        h("nav", { class: "tabs", "aria-label": "Sections" },
          TABS.map(([href, label, key]) => h("a", {
            class: `tab ${key === tab ? "on" : ""}`, href, "aria-current": key === tab ? "page" : null,
          }, label))),
        // On narrower screens only the icons show, and the label is the link's name.
        h("div", { class: "barlinks" },
          LINKS.map(([href, label, ico]) => h("a", {
            class: "barlink", href, target: "_blank", rel: "noreferrer", "aria-label": label, title: label,
          }, icon(ico, 20), h("span", { class: "bl" }, label)))),
        actions);
  return bar;
}

/** The credit under every page, as in Kiosk Satellite's remote admin. */
function footer() {
  const link = (href, text) => h("a", { href, target: "_blank", rel: "noreferrer" }, text);
  return h("footer", { class: "made-by" },
    h("span", {}, "Made with ", h("span", { class: "heart" }, "\u2665"), " by ",
      link("https://github.com/jxlarrea", "Xavier Larrea")),
    " \u00b7 ",
    h("span", {}, "\u2615 ", link("https://buymeacoffee.com/jxlarrea", "Buy me a coffee")));
}

async function signInPage() {
  let info = { login: false };
  try { info = await (await fetch("auth/info")).json(); } catch { /* service restarting */ }
  // HA_TOKEN opens the web UI: say why it didn't, then offer the login instead.
  const token = {
    refused: "Home Assistant refused the token in HA_TOKEN. Set a valid long-lived access token and restart Ground Control, or log in below.",
    unreachable: `Ground Control couldn't reach Home Assistant at ${info.ha_url} to check the token in HA_TOKEN. Reload this page once Home Assistant is up, or log in below.`,
  }[info.token];
  const body = info.login
    ? [token ? h("div", { class: "row" }, h("div", { class: "grow" },
        h("div", { class: "rt" }, info.token === "refused" ? "HA_TOKEN was refused" : "HA_TOKEN not checked"),
        h("div", { class: "rd" }, token))) : null,
      h("div", { class: "row" },
        h("div", { class: "grow" },
          h("div", { class: "rt" }, "Log in with your Home Assistant account"),
          h("div", { class: "rd" }, `Sign in with your Home Assistant user name and password at ${info.ha_url}. You come back here afterwards.`)),
        h("button", { class: "btn fill", onclick: logIn }, "Log in with Home Assistant"))]
    : [h("div", { class: "empty" }, "Open Ground Control from the Home Assistant sidebar.")];
  return [
    shell(null),
    h("main", { class: "page narrow" },
      h("h1", { class: "ptitle" }, "Sign in"),
      h("div", { class: "card" }, ...body)),
  ];
}

let retry = null;

async function render() {
  clearTimeout(retry);
  if (cleanup) { try { cleanup(); } catch { /* page already gone */ } cleanup = null; }
  const app = document.getElementById("app");
  const [name, params] = route();
  try {
    const page = await PAGES[name]();
    const out = await page.render(params);
    app.replaceChildren(...[out.nodes].flat(), footer());
    cleanup = out.cleanup || null;
    window.scrollTo(0, 0);
  } catch (err) {
    if (err instanceof SignInNeeded) {
      app.replaceChildren(...(await signInPage()), footer());
      return;
    }
    if (err instanceof HomeAssistantDown) {
      // Home Assistant restarting: keep the sign-in and come back on our own once it answers.
      app.replaceChildren(shell(null), h("main", { class: "page narrow" },
        h("div", { class: "card" }, h("div", { class: "empty" },
          `${err.message}. Trying again in a few seconds. There is no need to log in again.`))), footer());
      retry = setTimeout(render, 5000);
      return;
    }
    app.replaceChildren(shell(null), h("main", { class: "page" },
      h("div", { class: "card" }, h("div", { class: "empty" }, `Something went wrong: ${err.message}`))), footer());
  }
}

window.addEventListener("hashchange", render);
render();
