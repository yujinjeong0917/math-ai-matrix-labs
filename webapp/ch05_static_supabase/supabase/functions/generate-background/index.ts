// 5장 Edge Function 예시: AI 배경 만들기. 실습은 이 파일을 실행하지 않고
// webapp_ch05_edge.py가 같은 순서를 파이썬으로 따라 해요.
// withSupabase / ctx 의 모양은 Supabase 문서 "Securing Edge Functions"의 예시를 그대로 따랐어요.
import { withSupabase } from 'npm:@supabase/server@1'

const DAILY_LIMIT = 5      // 사용자당 하루 횟수(가상)
const MAX_PROMPT = 200

export default {
  fetch: withSupabase({ auth: 'user' }, async (req, ctx) => {
    const uid = ctx.userClaims?.id                      // 로그인 토큰에서 꺼낸 내 id
    const { prompt } = await req.json()
    if (!prompt || prompt.length > MAX_PROMPT) {
      return Response.json({ error: 'prompt length' }, { status: 400 })
    }

    const day = new Date().toISOString().slice(0, 10)
    const { data: row } = await ctx.supabaseAdmin       // 비밀 키 권한(RLS 건너뜀)
      .from('ai_usage').select('count').eq('user_id', uid).eq('day', day).maybeSingle()
    const used = row?.count ?? 0
    if (used >= DAILY_LIMIT) {
      return Response.json({ error: 'daily limit', used, limit: DAILY_LIMIT }, { status: 429 })
    }

    const aiKey = Deno.env.get('AI_IMAGE_KEY')!          // supabase secrets set 으로 넣은 비밀
    const ai = await fetch('https://ai-image.example/v1/generate', {   // 가상의 AI 회사 주소
      method: 'POST',
      headers: { Authorization: `Bearer ${aiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    })
    const png = new Uint8Array(await ai.arrayBuffer())

    await ctx.supabaseAdmin.from('ai_usage')
      .upsert({ user_id: uid, day, count: used + 1 })
    const path = `${uid}/ai/${day}-${String(used + 1).padStart(3, '0')}.png`
    await ctx.supabaseAdmin.storage.from('card-images')
      .upload(path, png, { contentType: 'image/png' })
    const { data } = await ctx.supabaseAdmin.storage.from('card-images')
      .createSignedUrl(path, 600)                        // 10분 동안만 열리는 주소
    return Response.json({ path, signedUrl: data?.signedUrl })
  }),
}
