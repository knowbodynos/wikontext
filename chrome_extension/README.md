Wikontext Chrome extension — updated to Manifest V3

Quick summary
- Migrated manifest to Manifest V3.
- Removed jQuery dependency from the content script; `src/inject.js` is now vanilla JS.
- Added a minimal MV3 service worker at `src/background.js`.
- Host permissions are scoped to `*://*.wikipedia.org/*` in `manifest.json`.

Loading locally (Chrome / Chromium / Edge)
1. Open `chrome://extensions` (or `edge://extensions`).
2. Enable "Developer mode" (top-right).
3. Click "Load unpacked" and select this repository's `chrome_extension` folder.

Testing notes
- Open a Wikipedia page (e.g., https://en.wikipedia.org/wiki/Artificial_intelligence).
- Hover links in the article body to exercise the extension's content script. The script posts to `https://wikontext.us/api/ext` (same as before).

Linting, packaging, and cleanup suggestions
- Use an ES linter (ESLint) to enforce style rules; add a small config for browser/MV3 environment.
- If you plan to publish to the Chrome Web Store, bump version in `manifest.json` and provide a polished `action` icon.
- The old `js/jquery` directory has been removed from the content scripts; consider deleting it from the repo if unused.

If you want, I can add an ESLint config and a small npm-based dev setup for building and validating the extension.

NPM helper commands (from inside the `chrome_extension` folder)

- Install dev deps:

```bash
npm install
```

- Run ESLint on the content script(s):

```bash
npm run lint
```

- Create a zip package ready for upload/inspection (creates `../wikontext-chrome-extension.zip`):

```bash
npm run package
```

