// 카드뉴스 편집기 2장: 잘못 고친 판 (실습용)
// 저장이 권한 오류로 막히자 비밀 키로 바꾸고, AI도 브라우저에서 바로 부르게 고쳤어요.
const SUPABASE_URL = "https://demo-project.invalid";
const SUPABASE_KEY = "DEMO_SUPABASE_SECRET_0002";
const AI_KEY = "DEMO_AI_SECRET_0003";

async function loadMyCards(userToken) {
  const response = await fetch(SUPABASE_URL + "/rest/v1/cards?select=id,title", {
    headers: { apikey: SUPABASE_KEY, Authorization: "Bearer " + userToken }
  });
  return response.json();
}

async function requestBackground(prompt) {
  const response = await fetch("https://ai-images.invalid/v1/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer " + AI_KEY },
    body: JSON.stringify({ prompt: prompt })
  });
  return response.text();
}
//# sourceMappingURL=app.js.map
