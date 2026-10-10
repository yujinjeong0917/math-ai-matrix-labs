// 카드뉴스 편집기 2장: 내 카드 저장하기 (실습용 원본 코드)
// 공개용 키만 브라우저에 둬요. 누가 어떤 행을 읽을지는 DB의 RLS 정책이 정해요.
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_KEY = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;

async function loadMyCards(userToken) {
  const response = await fetch(SUPABASE_URL + "/rest/v1/cards?select=id,title", {
    headers: { apikey: SUPABASE_KEY, Authorization: "Bearer " + userToken }
  });
  return response.json();
}

// AI 배경은 1장처럼 우리 서버에 부탁해요. AI 키는 서버에만 있어요.
async function requestBackground(prompt) {
  const response = await fetch("/api/generate-image", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: prompt })
  });
  return response.text();
}
