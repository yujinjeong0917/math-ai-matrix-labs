// 고치기 전 번들(실습용). AI로 빠르게 만든 첫 버전을 흉내 냈어요. 모든 키와 주소는 가짜예요.
const DB_URL = "/rest/v1/cards";
const DB_PUBLISHABLE_KEY = "demo_publishable_key_ch03";
const AI_SECRET_KEY = "DEMO_SECRET_NOT_REAL_0003";
const AI_URL = "https://ai.example.invalid/v1/images";

async function loadCards(session) {
  const res = await fetch(DB_URL, { headers: { apikey: DB_PUBLISHABLE_KEY, Authorization: "Bearer " + session.token } });
  const rows = await res.json();
  // 화면에서만 내 카드로 거른다
  return rows.filter((c) => c.owner_id === session.userId);
}

function showAdminLink(session) {
  // 관리자에게만 링크를 보여 준다
  if (session.role === "admin") document.getElementById("admin-link").hidden = false;
}

async function makeBackground(prompt) {
  // 브라우저가 AI 회사에 바로 요청한다
  const res = await fetch(AI_URL, { method: "POST", headers: { Authorization: "Bearer " + AI_SECRET_KEY }, body: JSON.stringify({ prompt }) });
  return res.blob();
}
