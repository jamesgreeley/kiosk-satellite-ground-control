// The service's API. Every URL is relative, so the UI works behind Home
// Assistant ingress (its own path prefix) and standalone alike.

// Standalone, the web UI logs in with Home Assistant. The login is a refresh
// token in a cookie only Ground Control reads, so Safari's 7-day wipe of what
// pages store leaves it alone. The access token lives in this page and comes
// from /auth/refresh. As a Home Assistant app none are needed, since ingress
// has signed the user in.
const OLD_AUTH_KEY = "ground-control-auth";
try { localStorage.removeItem("ground-control-token"); } catch { /* storage blocked */ }

export class SignInNeeded extends Error {}

/** Ground Control couldn't ask Home Assistant about the sign-in: wait for it, the login is kept. */
export class HomeAssistantDown extends Error {}

let auth = null; // { access_token, expires }
let noLogin = false; // /auth/refresh found no login: don't ask again until one
let haDown = false; // the last refresh couldn't reach Home Assistant

/** A login kept in this browser before the cookie: handed over once. */
function oldLogin() {
  try { return JSON.parse(localStorage.getItem(OLD_AUTH_KEY)); } catch { return null; }
}

function dropOldLogin() {
  try { localStorage.removeItem(OLD_AUTH_KEY); } catch { /* storage blocked */ }
}

/** Whether this browser logged in with Home Assistant, so it can sign out. */
export function loggedIn() {
  return !!auth;
}

let refreshing = null;

/** The access token, renewed a minute before Home Assistant lets it expire. */
async function accessToken(renew = false) {
  if (!renew && auth && auth.expires - Date.now() > 60000) return auth.access_token;
  if (noLogin) return null;
  refreshing ??= (async () => {
    try {
      const old = oldLogin();
      const r = await fetch("auth/refresh", {
        method: "POST", credentials: "same-origin",
        ...(old ? {
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ refresh_token: old.refresh_token, client_id: old.client_id }),
        } : {}),
      });
      if (old && (r.ok || r.status === 401)) dropOldLogin();
      haDown = !r.ok && r.status !== 401;
      if (r.status === 401) { auth = null; noLogin = true; return null; }
      if (!r.ok) return auth?.access_token ?? null;
      const got = await r.json();
      auth = { access_token: got.access_token, expires: Date.now() + got.expires_in * 1000 };
      return auth.access_token;
    } catch {
      // Ground Control restarting: keep what there is and try again later.
      haDown = true;
      return auth?.access_token ?? null;
    } finally {
      refreshing = null;
    }
  })();
  return refreshing;
}

/** fetch, signed in, once more with a renewed token if the first was refused. */
async function signedFetch(path, init = {}) {
  const t = await accessToken();
  init.headers = { ...(init.headers || {}), ...(t ? { Authorization: `Bearer ${t}` } : {}) };
  let r = await fetch(path, init);
  if (r.status === 401 && auth) {
    const again = await accessToken(true);
    if (again) {
      init.headers.Authorization = `Bearer ${again}`;
      r = await fetch(path, init);
    }
  }
  return r;
}

/** Off to Home Assistant's login page, back to this page after. */
export function logIn() {
  location.href = `auth/login?next=${encodeURIComponent(location.hash || "#/")}`;
}

export async function logOut() {
  const was = auth;
  auth = null;
  noLogin = true;
  const old = oldLogin();
  dropOldLogin();
  await fetch("auth/logout", {
    method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ access_token: was?.access_token, refresh_token: old?.refresh_token }),
  }).catch(() => {});
  location.reload();
}

export async function api(path, { method = "GET", json, body, type } = {}) {
  const init = { method, headers: {} };
  if (json !== undefined) {
    init.body = JSON.stringify(json);
    init.headers["Content-Type"] = "application/json";
  } else if (body !== undefined) {
    init.body = body;
    if (type) init.headers["Content-Type"] = type;
  }
  const r = await signedFetch(path, init);
  if (r.status === 401 && haDown) throw new HomeAssistantDown("Home Assistant can't be reached");
  if (r.status === 401) throw new SignInNeeded();
  if (r.status === 503) throw new HomeAssistantDown(await r.text());
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  if (r.status === 204) return null;
  const ct = r.headers.get("Content-Type") || "";
  return ct.includes("json") ? r.json() : r;
}

/** A blob URL for audio, since an <audio> tag can't send the sign-in header. */
export async function audioUrl(path) {
  const r = await signedFetch(path);
  if (!r.ok) throw new Error("no audio");
  return URL.createObjectURL(await r.blob());
}

export async function audioBytes(path) {
  const r = await signedFetch(path);
  if (!r.ok) throw new Error("no audio");
  return r.arrayBuffer();
}

/** A socket to the service, signed in with its first message. */
export function socket(path, first) {
  const url = new URL(path, location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  ws.addEventListener("open", async () => {
    const t = await accessToken();
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ ...first, token: t }));
  });
  return ws;
}
