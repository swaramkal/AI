# J.A.R.V.I.S. — Personal Offline Intelligence System

A static, browser-based personal knowledge assistant. No API key, backend, Firebase, or paid service is required.

## Features
- Import PDF, CSV, JSON, TXT, Markdown and HTML knowledge files.
- Search imported content with local keyword/similarity matching.
- Save custom question-answer memories.
- Suggest memories from explicit `Remember that ...` messages; approve or reject them.
- Basic built-in responses, date/time and arithmetic.
- English/Telugu-friendly text handling and browser speech input/output where supported.
- IndexedDB local storage, JSON backup and restore.
- Service worker for caching the app shell after the first visit.

## Publish with GitHub Pages
1. Create a new GitHub repository, for example `jarvis-offline`.
2. Upload **all files and folders** from this project to the repository root.
3. Open **Settings → Pages**.
4. Under Build and deployment, choose **Deploy from a branch**.
5. Select `main` and `/ (root)`, then Save.
6. Wait for GitHub Pages to publish and open the URL it provides.

## Important limitations
This is a local retrieval-based assistant, not a general-purpose generative language model. It cannot reliably answer arbitrary questions unless the answer is in its built-in rules, saved memories, or imported datasets. It does not silently learn facts from every conversation. It only proposes a memory when you explicitly write `Remember that ...`, and it saves that suggestion only after you approve it.

PDF support uses PDF.js from a CDN the first time it is needed, so PDF import requires an internet connection to load the reader. Other supported formats use browser code. The app shell is cached after first load; user data remains in that browser's IndexedDB.

Data is local to the browser/device. It is not synced to GitHub or other devices. Use Export full backup regularly. Do not put private memory exports or sensitive datasets in a public GitHub repository.

## Suggested tests
- Chat: `What can you do?`
- Chat: `I am back`
- Memory Engine: save Q `What is my preferred name?`, A `Swaramkal`; then ask the question in Chat.
- Chat: `Remember that my favorite editor is VS Code`; approve it in Learning approval; ask `What is my favorite editor?`
- Import a TXT file containing a fact, then ask a question about that fact.
