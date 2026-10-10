// 카드뉴스 편집기 서비스 워커(실습용). 배포할 때마다 version을 바꿔요.
/* config:start */
const CONFIG = {
  "version": "card-editor-v1",
  "precache": [
    "/", "/style.css", "/app.min.js", "/fonts/card-sans.woff2",
    "/images/logo.svg", "/images/bg-dots.svg",
    "/templates/index.json", "/templates/notice.json", "/templates/quote.json",
    "/manifest.json", "/icons/icon-192.png", "/icons/icon-512.png"
  ],
  "routes": [
    {"match": "navigate", "strategy": "network-first"},
    {"match": "/templates/", "strategy": "stale-while-revalidate"},
    {"match": "/api/", "strategy": "network-only"},
    {"match": "*", "strategy": "cache-first"}
  ]
};
/* config:end */

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CONFIG.version).then((cache) =>
      cache.addAll(CONFIG.precache.map((u) => new Request(u, { cache: "reload" })))
    )
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n !== CONFIG.version).map((n) => caches.delete(n)))
    )
  );
});

// HTTP 캐시(하루 max-age)를 건너뛰고 서버에서 바로 받기
function fromNetwork(req) {
  return fetch(req, { cache: "reload" });
}

function pickStrategy(request) {
  const url = new URL(request.url);
  for (const r of CONFIG.routes) {
    if (r.match === "navigate") {
      if (request.mode === "navigate") return r.strategy;
    } else if (r.match === "*" || url.pathname.startsWith(r.match)) {
      return r.strategy;
    }
  }
  return "network-only";
}

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  if (new URL(req.url).origin !== self.location.origin) return;
  const strategy = pickStrategy(req);
  if (strategy === "network-only") return;
  if (strategy === "cache-first") event.respondWith(cacheFirst(req));
  else if (strategy === "network-first") event.respondWith(networkFirst(req));
  else event.respondWith(staleWhileRevalidate(event, req));
});

async function cacheFirst(req) {
  const hit = await caches.match(req);
  if (hit) return hit;
  const res = await fromNetwork(req);
  if (res.ok) {
    const cache = await caches.open(CONFIG.version);
    await cache.put(req, res.clone());
  }
  return res;
}

async function networkFirst(req) {
  try {
    const res = await fromNetwork(req);
    if (res.ok) {
      const cache = await caches.open(CONFIG.version);
      await cache.put(req, res.clone());
    }
    return res;
  } catch (err) {
    const hit = await caches.match(req);
    return hit || Response.error();
  }
}

async function staleWhileRevalidate(event, req) {
  const cache = await caches.open(CONFIG.version);
  const hit = await cache.match(req);
  const update = fromNetwork(req).then(async (res) => {
    if (res.ok) await cache.put(req, res.clone());
    return res;
  });
  if (hit) {
    event.waitUntil(update.catch(() => {}));
    return hit;
  }
  return update;
}
