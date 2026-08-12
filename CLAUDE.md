# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- `npm run build` — build the site into `dist/`
- `npm run serve` — build, then serve `dist/` at http://127.0.0.1:3000 with live reload (watches `content/` for `.md`/`.bib`/`.yml` changes; does NOT watch `public/`, `config.yml`, or `scripts/` — restart or rebuild manually for those)
- `npm run dev` — rebuild on change via `node --watch`, no server
- `npm run clean` — delete `dist/`

There are no tests and no linter. Verifying a change means running `npm run build` (it fails loudly on invalid chart JSON, missing bibliography files, unknown citation keys, or a missing `content/index.md`) and inspecting the output.

Deployment is automatic: `.github/workflows/deploy-pages.yml` runs `npm ci && npm run build` and publishes `dist/` to GitHub Pages on every push to `main`.

## Architecture

This is a custom static site generator, not a framework. The entire build pipeline is `scripts/build.mjs` (~600 lines, only deps: gray-matter, markdown-it, markdown-it-anchor, highlight.js). There are no template files — all HTML is assembled as strings in `build.mjs` (`shell()` for the page chrome, `articlePage()` for the article layout). Changing the page structure means editing those functions.

**Content model:** every page is a Markdown file in `content/` with YAML front matter (`title` required; `description`, `date`, `authors`, `affiliations`, `tags`, `bibliography`, back-matter fields optional). `content/foo.md` → `/foo/`, nested paths and `index.md`-as-directory-index supported. `content/index.md` is the homepage and the build errors without it. `public/` is copied verbatim into `dist/`, so assets are referenced with absolute paths like `/images/foo.png`. Site-wide settings (name, `base_url`, header logo, copyright) live in `config.yml`; `base_url` must be set if deploying as a project site rather than at the domain root.

**Custom pipeline features, all implemented inside `build.mjs`:**

- **Citations:** a hand-rolled BibTeX parser (`parseBibtex`) plus APA renderer. `[@key]` → parenthetical, `@key` → narrative; citing a key not in the page's declared `bibliography:` file throws at build time. A References section is generated automatically. Only APA is supported.
- **Math:** `$...$` / `$$...$$` segments are replaced with placeholder markers before markdown-it runs, then restored afterward (`renderMarkdown`), so MathJax (loaded from CDN at runtime) does the actual rendering client-side. Code fences are exempt from this protection.
- **Charts:** ` ```chart ` fences must contain JSON with `type` and `data`; the config is base64-encoded into a `data-chart-config` attribute and hydrated at runtime by `public/assets/charts.js` using Chart.js from CDN.
- **TOC:** the sticky sidebar nav is built from `<h2>`/`<h3>` ids in the rendered HTML (`outlineFromHtml`), with scroll tracking in `public/assets/toc.js`.
- **Superscripts in front matter:** author/affiliation strings support `$^1$` / `$^{...}$` notation, rendered by `configInlineMarkup` (not MathJax).

Client-side behavior lives in `public/assets/` (`toc.js`, `charts.js`, `copy-code.js`, `site.css`); MathJax and Chart.js are the only external runtime dependencies, both loaded from CDN.

`scripts/serve.mjs` is a dependency-free preview server (Node built-in `http`) that respawns `build.mjs` on content changes and pushes live reloads over server-sent events.
