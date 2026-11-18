# J.A.R.V.I.S. — Agent-Trained Web HUD & Neural Knowledge Core

A GitHub Pages-ready J.A.R.V.I.S. web interface featuring **in-browser custom neural training**, pre-indexed **Agentic Training & Evaluation Datasets** (`ToolBench`, `BFCL`, `GAIA`, `SWE-bench`, `WebArena`, `ALFRED`, etc.), and live **BFCL-style tool routing** (Wikipedia, Weather, Dictionary, Math).

## Training J.A.R.V.I.S.
1. **Inline Chat Training**: Type `Train: Your Question => Your Desired Answer` directly in the terminal (e.g., `Train: Who is my creator? => You are my creator, Sir.`).
2. **🧠 TRAIN CORE Modal**: Click **🧠 TRAIN CORE** in the header to add custom Q&A pairs, inspect the 15 indexed agentic datasets, or export your trained weights as `jarvis-trained-weights.json`.
3. **Optional External LLM (`⚙ API KEY`)**: Plug in a Groq or OpenAI key so J.A.R.V.I.S. automatically injects your custom trained facts into full LLM conversations.

## Deploying to GitHub Pages
Upload `index.html`, `style.css`, `script.js`, and `README.md` to your GitHub repository root and enable **Settings > Pages > Deploy from a branch (`main`)**.
