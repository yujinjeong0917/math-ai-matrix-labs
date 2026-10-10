-- 카드 테이블의 읽기 규칙 (실습용 예시, Postgres·Supabase 문법)
-- 1) RLS를 켜요. 켜고 정책이 하나도 없으면 공개용 키로는 아무 행도 읽히지 않아요.
alter table public.cards enable row level security;

-- 2) 공개로 표시한 카드는 로그인 안 한 사람(anon)도, 로그인한 사람(authenticated)도 읽어요.
create policy "Anyone can read public cards"
on public.cards for select
to anon, authenticated
using ( is_public );

-- 3) 로그인한 사람은 자기 카드를 읽어요.
create policy "Owners can read their own cards"
on public.cards for select
to authenticated
using ( (select auth.uid()) = owner_id );
