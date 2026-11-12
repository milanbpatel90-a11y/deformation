-- Apply in Supabase SQL Editor before enabling the product dashboard.
-- The API uses each signed-in user's JWT; no service-role key is required.

create table if not exists public.eyewear_products (
 id uuid primary key,
 user_id uuid not null references auth.users(id) on delete cascade,
 owner_email text not null,
 item_name text not null check (char_length(item_name) between 1 and 160),
 price numeric(12,2) check (price is null or price >= 0),
 currency char(3) not null default 'INR',
 description text,
 weight_grams numeric(10,2) check (weight_grams is null or weight_grams >= 0),
 size text, model text, style text, colour text, logo_url text, brand text, frame_material text,
 status text not null default 'draft' check (status in ('draft','active','inactive')),
 glb_path text not null unique,
 created_at timestamptz not null default now(),
 updated_at timestamptz not null default now()
);
create index if not exists eyewear_products_user_created_idx on public.eyewear_products(user_id, created_at desc);
alter table public.eyewear_products enable row level security;

drop policy if exists "Users read own eyewear products" on public.eyewear_products;
create policy "Users read own eyewear products" on public.eyewear_products for select to authenticated
using (auth.uid() = user_id or lower(coalesce(auth.jwt()->>'email','')) in ('milanbpatel90@gmail.com','milan.patel90@yahoo.com'));

drop policy if exists "Users create own eyewear products" on public.eyewear_products;
create policy "Users create own eyewear products" on public.eyewear_products for insert to authenticated
with check (auth.uid() = user_id);

drop policy if exists "Users update own eyewear products" on public.eyewear_products;
create policy "Users update own eyewear products" on public.eyewear_products for update to authenticated
using (auth.uid() = user_id or lower(coalesce(auth.jwt()->>'email','')) in ('milanbpatel90@gmail.com','milan.patel90@yahoo.com'))
with check (auth.uid() = user_id or lower(coalesce(auth.jwt()->>'email','')) in ('milanbpatel90@gmail.com','milan.patel90@yahoo.com'));

drop policy if exists "Users delete own eyewear products" on public.eyewear_products;
create policy "Users delete own eyewear products" on public.eyewear_products for delete to authenticated
using (auth.uid() = user_id or lower(coalesce(auth.jwt()->>'email','')) in ('milanbpatel90@gmail.com','milan.patel90@yahoo.com'));

insert into storage.buckets (id,name,public,file_size_limit,allowed_mime_types)
values ('eyewear-products','eyewear-products',false,52428800,array['model/gltf-binary','application/octet-stream'])
on conflict (id) do update set public=false,file_size_limit=52428800;

drop policy if exists "Users read own eyewear GLBs" on storage.objects;
create policy "Users read own eyewear GLBs" on storage.objects for select to authenticated
using (bucket_id='eyewear-products' and ((storage.foldername(name))[1]=auth.uid()::text or lower(coalesce(auth.jwt()->>'email','')) in ('milanbpatel90@gmail.com','milan.patel90@yahoo.com')));

drop policy if exists "Users upload own eyewear GLBs" on storage.objects;
create policy "Users upload own eyewear GLBs" on storage.objects for insert to authenticated
with check (bucket_id='eyewear-products' and (storage.foldername(name))[1]=auth.uid()::text);

drop policy if exists "Users delete own eyewear GLBs" on storage.objects;
create policy "Users delete own eyewear GLBs" on storage.objects for delete to authenticated
using (bucket_id='eyewear-products' and ((storage.foldername(name))[1]=auth.uid()::text or lower(coalesce(auth.jwt()->>'email','')) in ('milanbpatel90@gmail.com','milan.patel90@yahoo.com')));

create or replace function public.set_eyewear_products_updated_at() returns trigger language plpgsql as $$
begin new.updated_at=now(); return new; end;
$$;
drop trigger if exists set_eyewear_products_updated_at on public.eyewear_products;
create trigger set_eyewear_products_updated_at before update on public.eyewear_products for each row execute function public.set_eyewear_products_updated_at();
grant select,insert,update,delete on public.eyewear_products to authenticated;
