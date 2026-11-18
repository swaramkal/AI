// J.A.R.V.I.S. Agent-Trained Core, Custom Memory & Live Tool Routing
const terminalLog = document.getElementById('terminal-log');
const cmdForm = document.getElementById('cmd-form');
const cmdInput = document.getElementById('cmd-input');
const micBtn = document.getElementById('mic-btn');
const micLabel = document.getElementById('mic-label');
const reactorCore = document.getElementById('reactor-core');
const coreState = document.getElementById('core-state');
const ttsToggle = document.getElementById('tts-toggle');
const aiModeBadge = document.getElementById('ai-mode-badge');

// 1. Pre-trained Agentic Training & Evaluation Knowledge Base (From User's Corpora)
const AGENTIC_DATASETS = [
    {
        name: "arcee-ai/agent-data",
        type: "Training Corpus",
        url: "https://huggingface.co/datasets/arcee-ai/agent-data",
        summary: "Function-calling and multi-turn agent conversations packaged as a supervised fine-tuning mixture for tool selection and dialogue planning."
    },
    {
        name: "ToolBench (ToolLLaMA)",
        type: "Training Corpus",
        url: "https://huggingface.co/datasets/tuandunghcmut/toolbench-v1",
        summary: "Large-scale supervised fine-tuning corpus for general real-world API and tool use, serving as the foundation for ToolLLM and ToolLLaMA."
    },
    {
        name: "ToolAlpaca",
        type: "Training Corpus",
        url: "https://github.com/tangqiaoyu/ToolAlpaca",
        summary: "Synthetic tool-use instances spanning 400+ tools with companion evaluation sets for generalized tool invocation."
    },
    {
        name: "API-Bank",
        type: "Training Corpus",
        url: "https://arxiv.org/abs/2304.08244",
        summary: "Tool-use dialogues paired with a runnable tool ecosystem designed for multi-step planning and verifiable API execution."
    },
    {
        name: "WebShop",
        type: "Training Environment",
        url: "https://github.com/princeton-nlp/WebShop",
        summary: "Simulated e-commerce environment grounded in real product data that tests multi-step search, attribute matching, and decision-making."
    },
    {
        name: "webshop_expert_trajectories",
        type: "Training Corpus",
        url: "https://huggingface.co/datasets/lclan/webshop_expert_trajectories",
        summary: "Expert agent rollouts containing both successes and failures for behavior cloning and error-recovery tuning."
    },
    {
        name: "ALFRED",
        type: "Embodied Corpus",
        url: "https://github.com/askforalfred/alfred",
        summary: "Embodied household robotics dataset mapping natural language instructions and egocentric vision to long-horizon action sequences."
    },
    {
        name: "WebArena",
        type: "Evaluation Benchmark",
        url: "https://webarena.dev/",
        summary: "Execution-validated web-task benchmark running on self-hosted realistic websites for reproducible agent regression testing."
    },
    {
        name: "WebChoreArena",
        type: "Evaluation Benchmark",
        url: "https://github.com/WebChoreArena/WebChoreArena",
        summary: "Benchmark of 532 tedious, multi-step web chore tasks designed to stress-test long-horizon workflow endurance."
    },
    {
        name: "MiniWoB++",
        type: "Evaluation Benchmark",
        url: "https://github.com/Farama-Foundation/miniwob-plusplus",
        summary: "Suite of 100+ synthetic web environments with a unified API for evaluating low-level UI and DOM action policies."
    },
    {
        name: "GAIA",
        type: "Evaluation Benchmark",
        url: "https://huggingface.co/datasets/gaia-benchmark/GAIA",
        summary: "General AI Assistants benchmark featuring tiered real-world questions requiring multi-step web browsing, reasoning, and tool synthesis."
    },
    {
        name: "AgentBench",
        type: "Evaluation Benchmark",
        url: "https://github.com/THUDM/AgentBench",
        summary: "Multi-dimensional benchmark across eight distinct environments—including OS and databases—scoring LLM-as-agent decision making."
    },
    {
        name: "SWE-bench",
        type: "Evaluation Benchmark",
        url: "https://github.com/SWE-bench/SWE-bench",
        summary: "Software engineering benchmark evaluating autonomous coding agents on real GitHub issues scored by Dockerized unit tests."
    },
    {
        name: "OmniACT",
        type: "Evaluation Benchmark",
        url: "https://huggingface.co/datasets/Writer/omniact",
        summary: "Multimodal desktop and web computer-use benchmark mapping screen states and instructions into executable GUI scripts."
    },
    {
        name: "BFCL (Berkeley Function Calling Leaderboard)",
        type: "Evaluation Benchmark",
        url: "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        summary: "Executable function-calling evaluation benchmark using prompts and tool schemas to measure strict tool-routing and parameter accuracy."
    }
];

// 2. Persistent Custom Trained Memory (LocalStorage)
let customMemory = JSON.parse(localStorage.getItem('jarvis_trained_memory') || '[]');

function saveMemory() {
    localStorage.setItem('jarvis_trained_memory', JSON.stringify(customMemory));
    renderTrainedMemory();
    updateWeightsIndicator();
}

function updateWeightsIndicator() {
    const total = AGENTIC_DATASETS.length + customMemory.length;
    document.getElementById('weights-val').textContent = `${total} NODES`;
    document.getElementById('mem-count').textContent = customMemory.length;
}
updateWeightsIndicator();

// Similarity Matching for Custom Trained Q&A
function findTrainedMatch(query) {
    const cleanQ = query.toLowerCase().replace(/[?.,!]/g, '').trim();
    if (!cleanQ) return null;

    // 1. Exact or substring match in custom user-trained memory
    for (const item of customMemory) {
        const trainedQ = item.q.toLowerCase().replace(/[?.,!]/g, '').trim();
        if (cleanQ === trainedQ || cleanQ.includes(trainedQ) || trainedQ.includes(cleanQ)) {
            return `${item.a} <span style="color:var(--amber);font-size:0.75rem;">[Trained Memory]</span>`;
        }
    }

    // 2. Token overlap match (>= 65% keyword overlap)
    const qTokens = cleanQ.split(/\s+/).filter(w => w.length > 2);
    if (qTokens.length > 0) {
        let bestScore = 0;
        let bestItem = null;
        for (const item of customMemory) {
            const tTokens = item.q.toLowerCase().replace(/[?.,!]/g, '').split(/\s+/).filter(w => w.length > 2);
            const overlap = qTokens.filter(t => tTokens.includes(t)).length;
            const score = overlap / Math.max(qTokens.length, tTokens.length);
            if (score > bestScore && score >= 0.65) {
                bestScore = score;
                bestItem = item;
            }
        }
        if (bestItem) {
            return `${bestItem.a} <span style="color:var(--amber);font-size:0.75rem;">[Trained Memory]</span>`;
        }
    }

    // 3. Match against pre-trained Agentic Datasets
    if (cleanQ.includes('what datasets are you trained on') || cleanQ.includes('list datasets') || cleanQ.includes('training data')) {
        const names = AGENTIC_DATASETS.map(d => `<strong>${d.name}</strong>`).join(', ');
        return `My neural architecture is trained and benchmarked on 15 agentic corpora: ${names}. Ask me about any specific dataset (e.g., <em>"Explain SWE-bench"</em>) for full telemetry.`;
    }

    for (const ds of AGENTIC_DATASETS) {
        const shortKey = ds.name.split(' ')[0].toLowerCase();
        if (cleanQ.includes(shortKey)) {
            return `<strong>${ds.name}</strong> [<em>${ds.type}</em>]: ${ds.summary} <br><a href="${ds.url}" target="_blank">[Inspect Corpus Source]</a>`;
        }
    }

    return null;
}

// 3. Training Modal UI Logic
const trainModal = document.getElementById('train-modal');
document.getElementById('train-modal-btn').addEventListener('click', () => {
    renderDatasetsTab();
    renderTrainedMemory();
    trainModal.classList.remove('hidden');
});
document.getElementById('close-train-modal').addEventListener('click', () => trainModal.classList.add('hidden'));

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
        btn.classList.add('active');
        document.getElementById(btn.getAttribute('data-tab')).classList.remove('hidden');
    });
});

document.getElementById('save-train-btn').addEventListener('click', () => {
    const q = document.getElementById('train-q').value.trim();
    const a = document.getElementById('train-a').value.trim();
    if (!q || !a) return;
    customMemory.unshift({ q, a, timestamp: new Date().toISOString() });
    saveMemory();
    document.getElementById('train-q').value = '';
    document.getElementById('train-a').value = '';
    trainModal.classList.add('hidden');
    appendMessage('jarvis', `Training complete, Sir. New neural pathway committed:<br><strong>Q:</strong> ${q}<br><strong>A:</strong> ${a}`);
    speak('Training complete, Sir. New knowledge committed to my neural matrix.');
});

function renderDatasetsTab() {
    const container = document.getElementById('dataset-list');
    container.innerHTML = AGENTIC_DATASETS.map(ds => `
        <div class="ds-card">
            <strong>${ds.name}</strong>
            <span>${ds.type} // INDEXED</span>
            <p style="margin-top:4px;color:#bfefff;">${ds.summary.slice(0, 88)}...</p>
        </div>
    `).join('');
}

function renderTrainedMemory() {
    const list = document.getElementById('trained-list');
    if (customMemory.length === 0) {
        list.innerHTML = '<p class="modal-desc">No custom Q&amp;A weights trained yet. Add one in Tab 1 or type <code>Train: question =&gt; answer</code> in chat.</p>';
        return;
    }
    list.innerHTML = customMemory.map((item, idx) => `
        <div class="trained-item">
            <div>
                <strong style="color:var(--cyan);">Q: ${item.q}</strong><br>
                <span>A: ${item.a}</span>
            </div>
            <button class="del-item-btn" onclick="deleteTrainedItem(${idx})">DEL</button>
        </div>
    `).join('');
}

window.deleteTrainedItem = function(idx) {
    customMemory.splice(idx, 1);
    saveMemory();
};

document.getElementById('export-memory-btn').addEventListener('click', () => {
    const payload = {
        model: "JARVIS-Custom-Weights-v1",
        builtin_datasets: AGENTIC_DATASETS.map(d => d.name),
        custom_trained_pairs: customMemory
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'jarvis-trained-weights.json';
    a.click();
});

document.getElementById('wipe-memory-btn').addEventListener('click', () => {
    customMemory = [];
    saveMemory();
});

// 4. Optional External LLM Configuration
let savedProvider = localStorage.getItem('jarvis_provider') || 'groq';
let savedApiKey = localStorage.getItem('jarvis_api_key') || '';

function updateModeBadge() {
    aiModeBadge.textContent = savedApiKey
        ? `SYS.ONLINE // ${savedProvider.toUpperCase()} + TRAINED CORE`
        : 'SYS.ONLINE // AGENT-TRAINED CORE';
}
updateModeBadge();

const apiModal = document.getElementById('api-modal');
document.getElementById('settings-btn').addEventListener('click', () => {
    document.getElementById('api-provider').value = savedProvider;
    document.getElementById('api-key-input').value = savedApiKey;
    apiModal.classList.remove('hidden');
});
document.getElementById('close-modal').addEventListener('click', () => apiModal.classList.add('hidden'));
document.getElementById('save-api-btn').addEventListener('click', () => {
    savedProvider = document.getElementById('api-provider').value;
    savedApiKey = document.getElementById('api-key-input').value.trim();
    localStorage.setItem('jarvis_provider', savedProvider);
    localStorage.setItem('jarvis_api_key', savedApiKey);
    updateModeBadge();
    apiModal.classList.add('hidden');
    appendMessage('jarvis', 'External neural link activated. Your custom trained facts will also be injected into every LLM prompt.');
});
document.getElementById('clear-api-btn').addEventListener('click', () => {
    savedApiKey = '';
    localStorage.removeItem('jarvis_api_key');
    updateModeBadge();
    apiModal.classList.add('hidden');
    appendMessage('jarvis', 'Switched to built-in zero-key Trained Knowledge Core.');
});

// Live Clock & Battery
function updateClock() {
    document.getElementById('clock').textContent = new Date().toTimeString().split(' ')[0];
}
setInterval(updateClock, 1000);
updateClock();

if ('getBattery' in navigator) {
    navigator.getBattery().then(battery => {
        const updateBat = () => {
            document.getElementById('battery-status').textContent = `PWR: ${Math.round(battery.level * 100)}%`;
        };
        updateBat();
        battery.addEventListener('levelchange', updateBat);
    });
}

// Terminal & Speech Helpers
function appendMessage(sender, htmlText) {
    const div = document.createElement('div');
    div.className = `msg ${sender}`;
    const tag = sender === 'user' ? '[USER]' : '[J.A.R.V.I.S.]';
    div.innerHTML = `<span class="tag">${tag}</span> ${htmlText}`;
    terminalLog.appendChild(div);
    terminalLog.scrollTop = terminalLog.scrollHeight;
}

function speak(plainText) {
    if (!ttsToggle.checked || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const cleanText = plainText.replace(/<[^>]*>/g, '').slice(0, 420);
    const utterance = new SpeechSynthesisUtterance(cleanText);
    const voices = window.speechSynthesis.getVoices();
    const britishVoice = voices.find(v => v.lang.includes('en-GB') && v.name.includes('Male'))
                      || voices.find(v => v.lang.includes('en-GB'))
                      || voices.find(v => v.lang.includes('en'));
    if (britishVoice) utterance.voice = britishVoice;
    utterance.pitch = 0.95;
    utterance.rate = 1.03;

    utterance.onstart = () => {
        reactorCore.classList.add('speaking');
        coreState.textContent = 'SPEAKING';
    };
    utterance.onend = () => {
        reactorCore.classList.remove('speaking');
        coreState.textContent = 'STANDBY';
    };
    window.speechSynthesis.speak(utterance);
}

// 5. BFCL-Style Tool Functions (Math, Weather, Dictionary, Wikipedia)
function tryMathTool(query) {
    const cleaned = query
        .replace(/^(what is|whats|what's|calculate|solve|compute|math)\s+/i, '')
        .replace(/×/g, '*').replace(/÷/g, '/').replace(/\^/g, '**').replace(/\?/g, '').trim();
    if (/^[0-9+\-*/().\s*]+$/.test(cleaned) && /[0-9]/.test(cleaned) && /[+\-*/]/.test(cleaned)) {
        try {
            const result = Function(`"use strict"; return (${cleaned})`)();
            if (typeof result === 'number' && !isNaN(result)) {
                return `The calculated result is <strong>${Number(result.toFixed(6))}</strong>, Sir. <span style="color:var(--cyan);font-size:0.75rem;">[Tool: MathEngine]</span>`;
            }
        } catch (e) {}
    }
    return null;
}

async function fetchWeatherTool(city) {
    const geoRes = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`);
    const geoData = await geoRes.json();
    if (!geoData.results || geoData.results.length === 0) return null;
    const { latitude, longitude, name, country } = geoData.results[0];
    const wRes = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code`);
    const wData = await wRes.json();
    const cur = wData.current;
    const codes = { 0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast', 45: 'Fog', 61: 'Light rain', 63: 'Moderate rain', 65: 'Heavy rain', 95: 'Thunderstorm' };
    return `Current telemetry for <strong>${name}, ${country}</strong>: ${codes[cur.weather_code] || 'Active conditions'}, <strong>${cur.temperature_2m}°C</strong>, humidity ${cur.relative_humidity_2m}%, wind ${cur.wind_speed_10m} km/h. <span style="color:var(--cyan);font-size:0.75rem;">[Tool: OpenMeteo]</span>`;
}

async function fetchDictionaryTool(word) {
    const res = await fetch(`https://api.dictionaryapi.dev/api/v2/entries/en/${encodeURIComponent(word)}`);
    if (!res.ok) return null;
    const data = await res.json();
    const meaning = data[0]?.meanings?.[0];
    const def = meaning?.definitions?.[0]?.definition;
    return def ? `<strong>${data[0].word}</strong> (<em>${meaning.partOfSpeech}</em>): ${def} <span style="color:var(--cyan);font-size:0.75rem;">[Tool: Lexicon]</span>` : null;
}

async function fetchWikipediaTool(question) {
    const topic = question
        .replace(/^(who is|who was|who are|what is|what are|what was|tell me about|explain|search for|search|define|where is|when was)\s+/i, '')
        .replace(/\?+$/, '').trim();
    if (!topic) return null;
    const searchRes = await fetch(`https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(topic)}&format=json&origin=*`);
    const searchData = await searchRes.json();
    const hits = searchData?.query?.search;
    if (!hits || hits.length === 0) return null;
    const sumRes = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(hits[0].title)}`);
    if (!sumRes.ok) return null;
    const sumData = await sumRes.json();
    if (sumData.extract) {
        const pageUrl = sumData.content_urls?.desktop?.page || `https://en.wikipedia.org/wiki/${encodeURIComponent(hits[0].title)}`;
        return `${sumData.extract} <br><a href="${pageUrl}" target="_blank">[Read Full Dossier]</a>`;
    }
    return null;
}

async function fetchLLMWithTrainedContext(userQuestion) {
    const memoryContext = customMemory.length > 0
        ? "User-trained custom facts you MUST prioritize:
" + customMemory.map(m => `- Q: ${m.q} -> A: ${m.a}`).join("
")
        : "";
    const endpoint = savedProvider === 'openai'
        ? 'https://api.openai.com/v1/chat/completions'
        : 'https://api.groq.com/openai/v1/chat/completions';
    const model = savedProvider === 'openai' ? 'gpt-4o-mini' : 'llama-3.3-70b-versatile';

    const res = await fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${savedApiKey}`
        },
        body: JSON.stringify({
            model,
            messages: [
                { role: 'system', content: `You are J.A.R.V.I.S., a sophisticated British AI assistant trained on agentic benchmarks (BFCL, ToolBench, GAIA, SWE-bench, WebArena, ALFRED). ${memoryContext}` },
                { role: 'user', content: userQuestion }
            ],
            temperature: 0.5
        })
    });
    if (!res.ok) throw new Error(`API HTTP ${res.status}`);
    const data = await res.json();
    return data.choices[0].message.content.replace(/\n/g, '<br>');
}

// 6. Master Command & Training Router
async function processCommand(rawCmd) {
    const trimmed = rawCmd.trim();
    if (!trimmed) return;
    const cmd = trimmed.toLowerCase();

    appendMessage('user', trimmed);
    reactorCore.classList.add('thinking');
    coreState.textContent = 'ROUTING';

    let response = '';

    try {
        // Inline Training Command: "Train: question => answer"
        if (/^train:\s*.+\s*=>\s*.+$/i.test(trimmed)) {
            const parts = trimmed.replace(/^train:\s*/i, '').split('=>');
            const q = parts[0].trim();
            const a = parts.slice(1).join('=>').trim();
            customMemory.unshift({ q, a, timestamp: new Date().toISOString() });
            saveMemory();
            response = `Neural weight updated, Sir. Whenever you ask <em>"${q}"</em>, I will respond: <strong>${a}</strong>`;
        }
        else if (cmd === 'clear' || cmd === 'clear terminal') {
            terminalLog.innerHTML = '';
            response = 'Terminal logs purged, Sir.';
        }
        else if (/^(what time|current time|time is it)/i.test(cmd)) {
            response = `The current local time is <strong>${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</strong>.`;
        }
        else if (/^(what is the date|today's date|what day is it)/i.test(cmd)) {
            response = `Today is <strong>${new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</strong>.`;
        }
        else if (cmd.startsWith('open ')) {
            const site = cmd.replace('open ', '').trim();
            const shortcuts = { github: 'https://github.com', youtube: 'https://www.youtube.com', google: 'https://www.google.com', huggingface: 'https://huggingface.co' };
            const url = shortcuts[site] || (site.startsWith('http') ? site : `https://www.${site}.com`);
            window.open(url, '_blank');
            response = `Launching <strong>${site}</strong> in a new secure viewport.`;
        }
        else {
            // Step A: Always check Custom Trained Memory & Agentic Datasets first!
            const trainedReply = findTrainedMatch(trimmed);
            if (trainedReply) {
                response = trainedReply;
            }
            // Step B: If External LLM API Key is active, use LLM + Trained Context
            else if (savedApiKey) {
                response = await fetchLLMWithTrainedContext(trimmed);
            }
            // Step C: Zero-Key Tool Routing (Math -> Weather -> Dictionary -> Wikipedia)
            else {
                const mathAns = tryMathTool(trimmed);
                if (mathAns) {
                    response = mathAns;
                } else if (cmd.includes('weather in ') || cmd.startsWith('weather ')) {
                    const city = trimmed.replace(/.*weather\s+(in\s+)?/i, '').replace(/\?+$/, '').trim();
                    response = (await fetchWeatherTool(city)) || `No meteorological data found for "${city}".`;
                } else if (cmd.startsWith('define ')) {
                    const word = trimmed.replace(/^define\s+/i, '').replace(/\?+$/, '').trim();
                    response = (await fetchDictionaryTool(word)) || (await fetchWikipediaTool(trimmed));
                } else {
                    const wikiAns = await fetchWikipediaTool(trimmed);
                    response = wikiAns || `No direct match found. You can train me on this topic anytime by typing:<br><code>Train: ${trimmed} =&gt; [your answer]</code>`;
                }
            }
        }
    } catch (err) {
        response = `System exception (${err.message}). Verify network or API key configuration.`;
    }

    reactorCore.classList.remove('thinking');
    coreState.textContent = 'STANDBY';
    appendMessage('jarvis', response);
    speak(response);
}

// Form & Quick Buttons
cmdForm.addEventListener('submit', (e) => {
    e.preventDefault();
    processCommand(cmdInput.value);
    cmdInput.value = '';
});

document.querySelectorAll('.cmd-btn[data-cmd]').forEach(btn => {
    btn.addEventListener('click', () => {
        processCommand(btn.getAttribute('data-cmd'));
    });
});

// Speech Recognition
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;

    micBtn.addEventListener('click', () => {
        try { recognition.start(); } catch (err) { recognition.stop(); }
    });
    recognition.onstart = () => {
        reactorCore.classList.add('listening');
        micBtn.classList.add('active');
        micLabel.textContent = 'LISTENING...';
        coreState.textContent = 'LISTENING';
    };
    recognition.onresult = (event) => processCommand(event.results[0][0].transcript);
    recognition.onerror = (event) => appendMessage('jarvis', `Audio sensor alert: ${event.error}.`);
    recognition.onend = () => {
        reactorCore.classList.remove('listening');
        micBtn.classList.remove('active');
        micLabel.textContent = 'INITIALIZE VOICE';
        if (coreState.textContent === 'LISTENING') coreState.textContent = 'STANDBY';
    };
} else {
    micBtn.style.opacity = '0.5';
    micLabel.textContent = 'VOICE UNSUPPORTED';
}
