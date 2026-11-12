# Supabase Google authentication and access policy

The application uses Supabase Auth with Google OAuth. FastAPI verifies each browser session against the Supabase Auth user endpoint and enforces access rules on the server.

## Access policy

| Capability | Owner (two configured admin emails) | Google user |
| --- | --- | --- |
| Sign in with Google | Yes | Yes |
| Upload eyewear photos | Yes | Yes |
| Generate 3D models | Yes | Yes |
| Use deformation pipeline | Yes | Yes |
| Rim detection and calibration | Yes | Yes |
| View and try on generated models | Yes | Yes |
| Load a model in the browser | Yes | Yes |
| Download GLB using the download endpoint | Yes | **No** |

Every authenticated non-owner is assigned the `editor` role, which permits the generation and rim-calibration workflows. Only the two exact emails listed in `SUPABASE_ADMIN_EMAILS` are assigned `owner` and allowed to call `GET /api/output/{filename}`. All other Google accounts receive the `editor` role and do not see the Download GLB button.

## 1. Configure Supabase

1. Create or select a project at https://supabase.com/dashboard.
2. Open **Authentication -> Sign In / Providers -> Google** and enable Google.
3. In Google Cloud Console, create a **Web application** OAuth client.
4. Add the app's origin (for example `https://YOUR-APP-DOMAIN`) to Google Authorized JavaScript origins.
5. Add the Supabase callback URL shown on the Google provider settings page (normally `https://YOUR_PROJECT_REF.supabase.co/auth/v1/callback`) to Google Authorized redirect URIs.
6. Paste the Google OAuth Client ID and Client Secret into Supabase's Google provider settings and save.
7. In **Authentication -> URL Configuration**, set the Site URL to the app's public origin and add these Redirect URLs:
   - `https://YOUR-APP-DOMAIN/login`
   - `http://localhost:8000/login` (local development only)

The application uses the PKCE OAuth flow and exchanges the callback code in the browser. The resulting access and refresh tokens are posted to FastAPI, validated against Supabase, and stored in HttpOnly cookies.

## 2. Configure backend environment

Set these variables in Railway (or your local `.env`):

| Variable | Value |
| --- | --- |
| `SUPABASE_URL` | Project URL from Supabase Project Settings -> API |
| `SUPABASE_PUBLISHABLE_KEY` | Supabase publishable key from Project Settings -> API |
| `SUPABASE_ADMIN_EMAILS` | Comma-separated allowlist: `milanbpatel90@gmail.com,milan.patel90@yahoo.com` |
| `AUTH_COOKIE_SECURE` | `true` in production over HTTPS; `false` only for local HTTP |

A legacy `SUPABASE_ANON_KEY` can be used instead of `SUPABASE_PUBLISHABLE_KEY`. Never place a Supabase secret/service-role key in frontend code or commit it to Git.

Copy `.env.example` to `.env` for local development. Do not commit `.env`.

## 3. Server-side routes and enforcement

- `GET /api/auth/config` returns only the public Supabase URL and publishable key needed by the Google login page.
- `POST /api/auth/session` verifies the Supabase access token before issuing HttpOnly application cookies.
- `GET /api/auth/me` validates the session and returns the user's role and permissions.
- `POST /api/auth/logout` clears application session cookies.
- `GET /api/model/{filename}` serves an authenticated GLB inline for Three.js so users can view and try it on.
- `GET /api/output/{filename}` is the explicit download endpoint and is Owner-only.

The viewer hides the Download GLB control for regular users, but the FastAPI endpoint also rejects non-owner downloads. UI hiding is not the security boundary.

**Model confidentiality limitation:** browser-based 3D viewing requires the browser to receive the GLB data. A user with browser developer tools may be able to capture the inline model response. This implementation blocks the explicit download endpoint and UI control; it is not DRM and cannot make a model impossible to extract after it has been rendered in the browser.

## 4. Deployment checklist

1. Set all four environment variables in Railway.
2. Ensure Railway serves the application over HTTPS and `AUTH_COOKIE_SECURE=true`.
3. Add the production `/login` URL to Supabase's redirect allowlist.
4. Confirm the two configured admin emails match the Google accounts that should have GLB download access.
5. Restart/redeploy the backend after changing environment variables.
6. Test with three accounts: both listed admin accounts should see Download GLB; another Google account should not see the button, should still generate and view models, and should receive HTTP 403 from `/api/output/{filename}`.

No database migration is required for this Supabase Auth integration; Supabase manages Google identities and sessions.


## 5. Private product dashboards and GLB storage

Apply `supabase/migrations/202611120001_private_eyewear_products.sql` in the Supabase SQL Editor. It creates:

- `public.eyewear_products` with product name, price/currency, description, weight, size, model/SKU, style, colour, brand, frame material, logo URL, status, owner and private GLB object path.
- A Supabase Postgres product table with Row Level Security (RLS): normal accounts can read/update/delete only their own product rows; the two configured admin emails can view and manage all product rows.
- GLB binaries are stored in a separate **private Wasabi bucket**, accessed only by the FastAPI backend.
- An index and updated-at trigger.

### User experience

1. Generate a GLB in the viewer.
2. Click **Add Product to Dashboard**.
3. Enter product details and save. The API verifies that the generated GLB belongs to the signed-in user, uploads it to the private Wasabi bucket, and saves the product row in Supabase.
4. Open `/dashboard` to manage saved products. Normal accounts see only their own products.
5. Admin accounts see all users' saved products, can edit/delete them, and have a server-authorized **Download GLB** action.

The viewer also has a **My Product Dashboard** link. Saved product model viewing is authenticated; saved GLB downloads are restricted by the backend to the two allowlisted admin emails.

### Security and deployment notes

- Keep the Wasabi bucket private. Wasabi credentials are server-side Railway variables and must never be exposed to the browser or committed to Git.
- Supabase access tokens continue to authorize product metadata through PostgREST and RLS. Wasabi object access is performed by FastAPI after checking the product owner/admin role.
- New generated GLBs receive a local ownership manifest. Product creation rejects a model if the manifest does not match the signed-in Supabase user.
- Generated files on Railway's local filesystem are temporary working artifacts. Saving a product copies the GLB into persistent Wasabi object storage; do not rely on the local output directory as permanent product storage.
- The migration currently contains the same two admin email addresses as `SUPABASE_ADMIN_EMAILS`. If that allowlist changes, update the SQL RLS policies to match and review the resulting access before deploying.
- The dashboard currently lists all products and their owner email to admins; it does not enumerate Supabase Auth accounts that have never created a product. A separate privileged user-directory feature would require a carefully scoped server-side admin credential and is intentionally not part of this change.

### Verification checklist

After applying the migration and setting the existing Supabase/Railway auth variables, test with:
- Admin account `milanbpatel90@gmail.com`: can see all product rows and download saved GLBs.
- Admin account `milan.patel90@yahoo.com`: same permissions.
- Normal account A: can create/read/update/delete its own product; cannot see account B's product or download any saved GLB.
- Normal account B: same private-only behavior; cannot see account A's product or download any saved GLB.
- Verify direct API calls, not just hidden buttons: normal accounts must receive HTTP 403 from the product download endpoint, and RLS must prevent cross-account reads/writes.

The repository unit tests cover filename validation and the owner-vs-user access helper. The four-account end-to-end checks require your live Supabase project, configured Google OAuth provider, and deployed backend; they cannot be honestly marked passed from repository-only testing.


## 6. Wasabi private object storage setup

The product database remains in Supabase Postgres. **All newly saved GLB files are stored in Wasabi**, not Supabase Storage.

1. In the Wasabi Console, create a bucket dedicated to the application (for example `defirmation-private-models`). Keep **Bucket Access: Private** and do not enable public access.
2. Create a Wasabi access key for the backend. Prefer a dedicated user/key with access restricted to this bucket and these object actions only: `s3:PutObject`, `s3:GetObject`, and `s3:DeleteObject` on `arn:aws:s3:::YOUR_BUCKET_NAME/*`. Do not use root account credentials.
3. In the bucket's region details, copy the exact S3 endpoint shown by Wasabi. Endpoint formats differ by region; do not guess the endpoint.
4. Add the following variables to the **Railway backend service**. Use the exact region and endpoint shown in your Wasabi console:

| Variable | Example / value |
| --- | --- |
| `WASABI_BUCKET` | `defirmation-private-models` |
| `WASABI_REGION` | Your bucket region, e.g. `ap-southeast-1` |
| `WASABI_ENDPOINT_URL` | Exact regional endpoint, e.g. `https://s3.ap-southeast-1.wasabisys.com` |
| `WASABI_ACCESS_KEY_ID` | Wasabi access key |
| `WASABI_SECRET_ACCESS_KEY` | Wasabi secret key |

5. Redeploy/restart the Railway backend so it installs `boto3` and reads the new variables.
6. Test create, view, admin download, and delete using real accounts.

### Wasabi security model

- Bucket is private; no browser-side Wasabi credentials and no public object URLs.
- Product object keys are namespaced as `{supabase_user_id}/{product_uuid}.glb`.
- FastAPI checks the authenticated user against the Supabase product row before reading, deleting, or downloading the object.
- Only the two admin accounts can use the product download endpoint. Normal users can view their own saved model through the authenticated viewer route.
- The application does not create presigned public links; model bytes are streamed through the authenticated FastAPI backend.

### Existing Supabase Storage objects

If the previous Supabase Storage migration was already applied, its old private bucket and policies may remain in that Supabase project, but this updated backend no longer writes new product GLBs there. Existing saved GLBs are not automatically copied to Wasabi; migrate them separately before removing any old bucket or data.
