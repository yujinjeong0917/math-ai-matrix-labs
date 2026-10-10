-- 1인 개발 앱 출시 실전 5장: 카드뉴스 편집기를 정적 호스팅 + Supabase로 옮길 때 거는 규칙.
-- Supabase(Postgres)에서 쓰는 형태로 적었어요. 실습은 Postgres 없이 같은 규칙을
-- webapp_ch05_rules.py(파이썬)와 sqlite WHERE 절 두 가지로 다시 만들어 비교해요.
-- auth.uid(), auth.jwt()는 Supabase가 로그인 토큰에서 꺼내 주는 값이에요(로그인 안 하면 null).
-- storage.foldername(name)은 파일 경로의 폴더 이름 목록, storage.extension(name)은 확장자예요.

-- ① 카드 표: 3장의 다섯 정책 중 이 장에서 쓰는 두 가지(읽기, 만들기)만 다시 적어요.
alter table public.cards enable row level security;

create policy "카드: 공개 카드나 내 카드를 읽는다"
on public.cards for select
to anon, authenticated
using ( is_public = true or (select auth.uid()) = owner_id );

create policy "카드: 내 카드만 만든다"
on public.cards for insert
to authenticated
with check ( (select auth.uid()) = owner_id );

-- ② AI 사용량 표: 사용자는 자기 사용량을 읽기만 하고, 늘리는 건 Edge Function(비밀 키)만 해요.
--    insert/update 정책을 만들지 않았으니 공개용 키로는 고칠 수 없어요(default deny).
alter table public.ai_usage enable row level security;

create policy "사용량: 내 사용량만 읽는다"
on public.ai_usage for select
to authenticated
using ( (select auth.uid()) = user_id );

-- ③ 스토리지 버킷 두 개
--    templates   : 공개(public) 버킷. 주소만 알면 누구나 받고, 올리기 정책은 만들지 않아요.
--    card-images : 비공개 버킷. 경로 첫 폴더가 내 사용자 id인 파일만 다뤄요. 예: <내 id>/upload/a.png
insert into storage.buckets (id, name, public) values ('templates', 'templates', true);
insert into storage.buckets (id, name, public) values ('card-images', 'card-images', false);

create policy "그림: 내 폴더 파일을 받는다"
on storage.objects for select
to authenticated
using (
  bucket_id = 'card-images' and
  (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
);

create policy "그림: 내 폴더에 그림 파일만 올린다"
on storage.objects for insert
to authenticated
with check (
  bucket_id = 'card-images' and
  (storage.foldername(name))[1] = (select auth.jwt()->>'sub') and
  storage.extension(name) in ('png', 'jpg', 'jpeg', 'webp')
);

create policy "그림: 내 폴더 파일을 덮어쓴다"
on storage.objects for update
to authenticated
using (
  bucket_id = 'card-images' and
  (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
)
with check (
  bucket_id = 'card-images' and
  (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
);

create policy "그림: 내 폴더 파일을 지운다"
on storage.objects for delete
to authenticated
using (
  bucket_id = 'card-images' and
  (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
);
