# Top-50 Mobile App Connector Coverage

Status: exploration-only, not frozen paper result.
Date: 2026-07-10

This note maps a representative overseas Top-50 Android app list to public
connector catalogs from Composio and OpenConnector.  The purpose is to estimate
which common mobile GUI apps already expose API-like actions and which apps
still require GUI-level Function recovery.

## Data Sources

- App list: `datasets/top_android_apps.md` in this repository.  The list is a
  representative overseas/mobile GUI app list and currently has a placeholder
  source field; do not cite it as an authoritative ranking until the source is
  replaced.
- Composio catalog: `docs/public/data/toolkits-list.json` from
  `ComposioHQ/composio`, branch `next`, commit `30ba4d2`.
- OpenConnector catalog: generated from `oomol-lab/open-connector`, commit
  `f9f1647`, using `npm install`, which generated 1044 apps and 10090 actions.
  The hosted OpenConnector catalog endpoint reported 1051 providers and 10262
  actions at collection time.

## Coverage Labels

- `direct`: a named toolkit/provider exists and has non-empty actions that
  plausibly cover the app's public API workflows.
- `partial`: a toolkit/provider exists, but it mainly covers business, bot,
  creator, model, or adjacent APIs rather than the consumer mobile GUI app.
- `empty`: a toolkit exists but reports zero tools/actions.
- `none`: no direct toolkit/provider was found in the sampled catalog.

Tool counts measure connector surface size, not task coverage.  A large toolkit
can still miss consumer GUI flows such as feed browsing, local settings,
permission dialogs, mobile-only checkout flows, or OEM-specific screens.

## Summary

| Catalog | Direct coverage | Partial coverage | Empty toolkit | No direct coverage | Usable or partial |
|---|---:|---:|---:|---:|---:|
| Composio | 16/50 | 3/50 | 2/50 | 29/50 | 19/50 |
| OpenConnector | 9/50 | 5/50 | 0/50 | 36/50 | 14/50 |

## One-to-One Mapping

| # | App | Category | Composio toolkit / tools | OpenConnector provider / actions | Coverage note |
|---:|---|---|---:|---:|---|
| 1 | Google Chrome | Browser/search | - | - | Browser/local web UI; no direct app connector. |
| 2 | Google | Search/feed | - | - | Consumer search/feed app; no direct app connector. |
| 3 | Facebook | Social | `facebook` (43, direct) | - | Composio covers Facebook Graph/Page-style actions; OpenConnector has only Meta marketing-level tools, not counted as direct Facebook app coverage. |
| 4 | Google Messages | Messaging | - | - | SMS/RCS local app; no direct connector. |
| 5 | YouTube | Video | `youtube` (51, direct) | `youtube` (29, direct) | Creator/content API coverage; consumer watch UI remains GUI-heavy. |
| 6 | Gmail | Email | `gmail` (63, direct) | `gmail` (46, direct) | API-rich email workflow. |
| 7 | Messenger | Messaging/social | - | - | Consumer chat app; no direct connector. |
| 8 | Instagram | Social/media | `instagram` (36, direct) | - | Composio covers Instagram media/comment/business-style actions. |
| 9 | Google Maps | Maps/location | `google_maps` (23, direct) | `google_routes` (2, partial) | Composio has Google Maps toolkit; OpenConnector has Google Routes only. |
| 10 | Clock (Samsung) | Device utility | - | - | Device/OEM local app. |
| 11 | Amazon Shopping | Shopping | `amazon` (0, empty) | - | Composio toolkit has zero tools; shopping GUI remains uncovered. |
| 12 | Google Play services | System service | - | - | Background system service, not a user task connector. |
| 13 | Google Photos | Photos/storage | `googlephotos` (14, direct) | `googlephotos` (18, direct) | Photo/storage API coverage; editing and browsing can remain GUI-heavy. |
| 14 | TikTok | Short video/social | `tiktok` (10, direct) | `tiktok_business` (15, partial) | Composio has TikTok toolkit; OpenConnector is TikTok Business only. |
| 15 | Phone by Google | Dialer/phone | - | - | Local telephony app; validation/OpenPhone tools are not direct app coverage. |
| 16 | WhatsApp Messenger | Messaging/social | `whatsapp` (57, partial) | `whatsapp` (17, partial) | Mostly WhatsApp Business/API workflows, not full consumer chat UI. |
| 17 | Samsung Calendar | Calendar/OEM | - | - | OEM local calendar; Google Calendar connector does not cover Samsung app. |
| 18 | Clock (Google) | Device utility | - | - | Local device utility. |
| 19 | Cash App | Payments/finance | - | - | Consumer payments app; no direct connector. |
| 20 | Snapchat | Social/camera | `snapchat` (139, partial) | - | Composio tools are largely ads/business APIs; consumer camera/chat UI remains uncovered. |
| 21 | Discord | Community/chat | `discord` (28, direct) | `discord` (19, direct) | Community/chat API coverage. |
| 22 | Samsung Calculator | Device utility | - | - | Local device utility. |
| 23 | Yahoo Mail | Email | - | - | No direct toolkit in sampled catalogs. |
| 24 | Spotify | Music/audio | `spotify` (88, direct) | `spotify` (88, direct) | Music API coverage. |
| 25 | YouTube Music | Music/audio | - | - | No direct YouTube Music toolkit. |
| 26 | Telegram | Messaging/social | `telegram` (18, partial) | `telegram` (21, partial) | Telegram coverage is mostly bot/API workflow coverage. |
| 27 | Walmart | Shopping | - | - | Shopping GUI uncovered. |
| 28 | Samsung Browser | Browser | - | - | Local browser app uncovered. |
| 29 | Reddit | Community/social | `reddit` (23, direct) | - | Composio covers Reddit; OpenConnector has no direct Reddit provider. |
| 30 | NewsBreak | News | - | - | Content feed app uncovered. |
| 31 | X | Social/news | `twitter` (79, direct) | `twitter` (71, direct) | Twitter/X API coverage. |
| 32 | Google Calendar | Calendar/productivity | `googlecalendar` (49, direct) | `googlecalendar` (37, direct) | API-rich calendar workflow. |
| 33 | Pinterest | Lifestyle/social | `pinterest` (0, empty) | - | Composio toolkit has zero tools; no usable direct coverage. |
| 34 | Microsoft Outlook | Email/calendar | `outlook` (305, direct) | `outlook` (11, direct) | API-rich email/calendar workflow; OpenConnector has a smaller action set. |
| 35 | ChatGPT | AI/productivity | - | `openai` (15, partial) | OpenConnector OpenAI is model API coverage, not ChatGPT mobile app coverage. |
| 36 | Calculator (Google) | Device utility | - | - | Local device utility. |
| 37 | Temu | Shopping | - | - | Shopping GUI uncovered. |
| 38 | Moto Camera 3 | Camera/OEM | - | - | Local camera app. |
| 39 | Life360 | Family/location | - | - | No direct toolkit in sampled catalogs. |
| 40 | Samsung Notes | Notes/OEM | - | - | OEM notes app uncovered. |
| 41 | Google Meet | Video meeting | `googlemeet` (15, direct) | - | Composio covers Meet; OpenConnector has meeting-related tools but no Google Meet provider. |
| 42 | Pixel Camera | Camera/OEM | - | - | Local camera app. |
| 43 | Contacts | Contacts | `googlecontacts` (24, direct) | - | Composio covers Google Contacts; local/OEM contacts flows may still require GUI. |
| 44 | Google Drive | Cloud storage | `googledrive` (90, direct) | `googledrive` (43, direct) | API-rich storage workflow. |
| 45 | Samsung My Files | File manager/OEM | - | - | Local file manager. |
| 46 | Threads | Social | - | - | No direct toolkit in sampled catalogs. |
| 47 | Venmo | Payments/finance | - | - | Consumer payments app uncovered. |
| 48 | Pandora | Music/audio | - | - | No direct toolkit in sampled catalogs. |
| 49 | Samsung Wallet | Wallet/payment | - | - | Local wallet/payment app; no direct connector. |
| 50 | Netflix | Streaming/video | - | - | Consumer streaming app uncovered. |

## Paper-Facing Takeaway

Existing connector catalogs are useful for API-rich productivity and work apps,
but they do not cover many common mobile GUI apps.  Even when a connector
exists, its tools often target backend, business, creator, advertising, or bot
APIs rather than consumer mobile GUI flows.  This supports a cautious claim:
public APIs can provide one source of reusable Functions, while OmniFlow is
needed to recover GUI-level Functions for mobile-only, personalized,
undocumented, OEM-local, or visually mediated workflows.

