-- 1인 개발 앱 출시 실전 3장: 카드뉴스 편집기의 cards 표에 거는 행 수준 보안(RLS) 정책.
-- Supabase(Postgres)에서 쓰는 형태로 적었어요. 실습은 Postgres 없이 같은 규칙을
-- webapp_ch03_rls.py(파이썬)와 sqlite WHERE 절 두 가지로 다시 만들어 비교해요.
-- auth.uid()는 Supabase가 로그인 토큰에서 꺼내 주는 사용자 id예요. 로그인하지 않으면 null이에요.

alter table public.cards enable row level security;

-- 권한(grant)은 "이 역할이 이 명령을 쓸 수 있나", 정책은 "그 명령이 어느 줄에 닿나"를 정해요.
-- 정책을 더해도 이미 준 권한은 그대로라서 먼저 거두고 필요한 것만 돌려줘요.
revoke all on table public.cards from anon, authenticated;
grant select on table public.cards to anon;
grant select, insert, update, delete on table public.cards to authenticated;

create policy "누구나 공개 카드를 읽는다"
on public.cards for select
to anon, authenticated
using ( is_public = true );

create policy "자기 카드를 읽는다"
on public.cards for select
to authenticated
using ( (select auth.uid()) = owner_id );

create policy "자기 카드만 만든다"
on public.cards for insert
to authenticated
with check ( (select auth.uid()) = owner_id );

create policy "자기 카드만 고친다"
on public.cards for update
to authenticated
using ( (select auth.uid()) = owner_id )
with check ( (select auth.uid()) = owner_id );

create policy "자기 카드만 지운다"
on public.cards for delete
to authenticated
using ( (select auth.uid()) = owner_id );
