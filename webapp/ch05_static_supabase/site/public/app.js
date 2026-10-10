// 5장 카드뉴스 편집기(정적 호스팅에 올리는 브라우저 코드, 실습용 예시)
// 여기 들어가는 값은 둘뿐이에요: 프로젝트 주소와 공개용(publishable) 키. 둘 다 보여도 되는 값이고,
// 실제로 지키는 건 data/policies.sql의 RLS·스토리지 정책이에요. AI 키는 여기 없어요.
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = "https://demo-project.supabase.example";
const SUPABASE_PUBLISHABLE_KEY = "DEMO_SUPABASE_PUBLISHABLE_0005";
const supabase = createClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY);

// 카드 저장: 정책 "카드: 내 카드만 만든다"가 owner_id를 확인해요
async function saveCard(userId, title, layout) {
  return supabase.from("cards").insert({ owner_id: userId, title, layout, is_public: false });
}

// 그림 올리기: 경로 첫 폴더가 내 id여야 정책을 통과해요
async function uploadImage(userId, file) {
  const path = `${userId}/upload/${crypto.randomUUID()}.png`;
  return supabase.storage.from("card-images").upload(path, file, { cacheControl: "3600", upsert: false });
}

// AI 배경: 브라우저는 설명 글만 보내고, 키는 Edge Function이 서버 쪽 환경 변수에서 꺼내 써요
async function makeBackground(prompt) {
  return supabase.functions.invoke("generate-background", { body: { prompt } });
}

export { saveCard, uploadImage, makeBackground };
