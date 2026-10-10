// 고친 뒤 번들(실습용). 공개용 키만 있고, AI 호출은 우리 서버를 거쳐요. 모든 키와 주소는 가짜예요.
const DB_URL = "/rest/v1/cards";
const DB_PUBLISHABLE_KEY = "demo_publishable_key_ch03";

async function loadCards(session) {
  // 거르기는 DB의 행 수준 보안(RLS) 정책이 해요. 화면은 받은 것을 그대로 그려요
  const res = await fetch(DB_URL, { headers: { apikey: DB_PUBLISHABLE_KEY, Authorization: "Bearer " + session.token } });
  return res.json();
}

function showAdminLink(session) {
  // 링크를 숨기는 건 편의일 뿐이에요. 막는 일은 서버가 /admin 에서 해요
  if (session.role === "admin") document.getElementById("admin-link").hidden = false;
}

async function makeBackground(prompt, session) {
  const res = await fetch("/api/generate-image", { method: "POST", headers: { Authorization: "Bearer " + session.token }, body: JSON.stringify({ prompt }) });
  if (res.status === 429) throw new Error("잠시 뒤에 다시 시도해 주세요");
  return res.text();
}
