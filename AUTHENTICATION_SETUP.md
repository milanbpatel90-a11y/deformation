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
- A **private** Storage bucket named `eyewear-products` with per-user folder policies.
- Row Level Security (RLS) policies: normal accounts can read/update/delete only their own products; the two configured admin emails can view and manage all product rows and read/delete all saved GLBs.
- An index and updated-at trigger.

### User experience

1. Generate a GLB in the viewer.
2. Click **Add Product to Dashboard**.
3. Enter product details and save. The API verifies that the generated GLB belongs to the signed-in user, copies it to private Supabase Storage, and saves the product row.
4. Open `/dashboard` to manage saved products. Normal accounts see only their own products.
5. Admin accounts see all users' saved products, can edit/delete them, and have a server-authorized **Download GLB** action.

The viewer also has a **My Product Dashboard** link. Saved product model viewing is authenticated; saved GLB downloads are restricted by the backend to the two allowlisted admin emails.

### Security and deployment notes

- Do not make the `eyewear-products` bucket public.
- The backend uses the user's Supabase access token for PostgREST and Storage. Do not add a service-role key to Vercel, the browser, or this repository.
- New generated GLBs receive a local ownership manifest. Product creation rejects a model if the manifest does not match the signed-in Supabase user.
- Generated files on Railway's local filesystem are temporary working artifacts. Saving a product copies the GLB into persistent Supabase Storage; do not rely on the local output directory as permanent product storage.
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
