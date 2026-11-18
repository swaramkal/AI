
// ============================================================
// J.A.R.V.I.S. — AGENT-TRAINED CORE
// Custom Memory + AI Integration + Live Tool Routing
// ============================================================

// 1. DOM ELEMENTS
const terminalLog = document.getElementById('terminal-log');
const cmdForm = document.getElementById('cmd-form');
const cmdInput = document.getElementById('cmd-input');
const micBtn = document.getElementById('mic-btn');
const micLabel = document.getElementById('mic-label');
const reactorCore = document.getElementById('reactor-core');
const coreState = document.getElementById('core-state');
const ttsToggle = document.getElementById('tts-toggle');
const aiModeBadge = document.getElementById('ai-mode-badge');

// ============================================================
// 2. AGENTIC DATASETS
// ============================================================

const AGENTIC_DATASETS = [
    {
        name: "arcee-ai/agent-data",
        type: "Training Corpus",
        url: "https://huggingface.co/datasets/arcee-ai/agent-data",
        summary: "Function-calling and multi-turn agent conversations for tool selection and dialogue planning."
    },
    {
        name: "ToolBench (ToolLLaMA)",
        type: "Training Corpus",
        url: "https://huggingface.co/datasets/tuandunghcmut/toolbench-v1",
        summary: "A supervised tool-use corpus for general API and tool interaction."
    },
    {
        name: "ToolAlpaca",
        type: "Training Corpus",
        url: "https://github.com/tangqiaoyu/ToolAlpaca",
        summary: "Synthetic tool-use examples spanning hundreds of tools."
    },
    {
        name: "API-Bank",
        type: "Training Corpus",
        url: "https://arxiv.org/abs/2304.08244",
        summary: "Tool-use dialogues paired with a runnable API ecosystem."
    },
    {
        name: "WebShop",
        type: "Training Environment",
        url: "https://github.com/princeton-nlp/WebShop",
        summary: "An environment for evaluating multi-step shopping and decision-making."
    },
    {
        name: "webshop_expert_trajectories",
        type: "Training Corpus",
        url: "https://huggingface.co/datasets/lclan/webshop_expert_trajectories",
        summary: "Expert agent trajectories for behavior learning and evaluation."
    },
    {
        name: "ALFRED",
        type: "Embodied Corpus",
        url: "https://github.com/askforalfred/alfred",
        summary: "An embodied AI dataset for natural-language household tasks."
    },
    {
        name: "WebArena",
        type: "Evaluation Benchmark",
        url: "https://webarena.dev/",
        summary: "A benchmark for evaluating realistic multi-step web tasks."
    },
    {
        name: "WebChoreArena",
        type: "Evaluation Benchmark",
        url: "https://github.com/WebChoreArena/WebChoreArena",
        summary: "A benchmark for long-horizon web workflow tasks."
    },
    {
        name: "MiniWoB++",
        type: "Evaluation Benchmark",
        url: "https://github.com/Farama-Foundation/miniwob-plusplus",
        summary: "Synthetic web environments for evaluating UI interaction."
    },
    {
        name: "GAIA",
        type: "Evaluation Benchmark",
        url: "https://huggingface.co/datasets/gaia-benchmark/GAIA",
        summary: "A benchmark for general AI assistants and multi-step reasoning."
    },
    {
        name: "AgentBench",
        type: "Evaluation Benchmark",
        url: "https://github.com/THUDM/AgentBench",
        summary: "A benchmark for evaluating AI agents across different environments."
    },
    {
        name: "SWE-bench",
        type: "Evaluation Benchmark",
        url: "https://github.com/SWE-bench/SWE-bench",
        summary: "A benchmark for software engineering agents."
    },
    {
        name: "OmniACT",
        type: "Evaluation Benchmark",
        url: "https://huggingface.co/datasets/Writer/omniact",
        summary: "A benchmark for desktop and web computer-use tasks."
    },
    {
        name: "BFCL (Berkeley Function Calling Leaderboard)",
        type: "Evaluation Benchmark",
        url: "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        summary: "A benchmark for evaluating function-calling accuracy."
    }
];

// ============================================================
// 3. CUSTOM TRAINED MEMORY
// ============================================================

let customMemory = [];

try {
    customMemory = JSON.parse(
        localStorage.getItem('jarvis_trained_memory') || '[]'
    );

    if (!Array.isArray(customMemory)) {
        customMemory = [];
    }
} catch (error) {
    customMemory = [];
}

function saveMemory() {
    localStorage.setItem(
        'jarvis_trained_memory',
        JSON.stringify(customMemory)
    );

    renderTrainedMemory();
    updateWeightsIndicator();
}

function updateWeightsIndicator() {
    const weightsVal = document.getElementById('weights-val');
    const memCount = document.getElementById('mem-count');

    if (weightsVal) {
        weightsVal.textContent =
            `${AGENTIC_DATASETS.length + customMemory.length} NODES`;
    }

    if (memCount) {
        memCount.textContent = customMemory.length;
    }
}

updateWeightsIndicator();

// ============================================================
// 4. TRAINED MEMORY MATCHING
// ============================================================


function findTrainedMatch(query) {
    const cleanQ = query
        .toLowerCase()
        .replace(/[?.,!]/g, '')
        .trim();

    if (!cleanQ) return null;

    for (const item of customMemory) {
        if (!item.q || !item.a) continue;

        const trainedQ = item.q
            .toLowerCase()
            .replace(/[?.,!]/g, '')
            .trim();

        if (
            cleanQ === trainedQ ||
            cleanQ.includes(trainedQ) ||
            trainedQ.includes(cleanQ)
        ) {
            return escapeHTML(item.a);
        }
    }

    const qTokens = cleanQ.split(/\s+/).filter(w => w.length > 2);

    if (qTokens.length > 0) {
        let bestScore = 0;
        let bestItem = null;

        for (const item of customMemory) {
            if (!item.q || !item.a) continue;

            const tTokens = item.q
                .toLowerCase()
                .replace(/[?.,!]/g, '')
                .split(/\s+/)
                .filter(w => w.length > 2);

            const overlap = qTokens.filter(
                token => tTokens.includes(token)
            ).length;

            const score = overlap /
                Math.max(qTokens.length, tTokens.length);

            if (score > bestScore && score >= 0.65) {
                bestScore = score;
                bestItem = item;
            }
        }

        if (bestItem) {
            return escapeHTML(bestItem.a);
        }
    }

    if (
        cleanQ.includes('what datasets are you trained on') ||
        cleanQ.includes('list datasets') ||
        cleanQ.includes('training data')
    ) {
        const names = AGENTIC_DATASETS
            .map(d => `<strong>${escapeHTML(d.name)}</strong>`)
            .join(', ');

        return `My indexed agentic dataset catalogue contains:
            ${names}. Ask me about a specific dataset for more information.`;
    }

    for (const ds of AGENTIC_DATASETS) {
        const shortKey = ds.name.split(' ')[0].toLowerCase();

        if (cleanQ.includes(shortKey)) {
            return `<strong>${escapeHTML(ds.name)}</strong>
                [<em>${escapeHTML(ds.type)}</em>]:
                ${escapeHTML(ds.summary)}
                <br><a href="${ds.url}" target="_blank"
                rel="noopener noreferrer">[Inspect Corpus Source]</a>`;
        }
    }

    return null;
}


// ============================================================
// 5. HTML ESCAPING
// ============================================================

function escapeHTML(value) {
    return String(value).replace(/[&<>"']/g, char => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    })[char]);
}

// ============================================================
// 6. TRAINING MODAL
// ============================================================

const trainModal = document.getElementById('train-modal');

document.getElementById('train-modal-btn')
    .addEventListener('click', () => {
        renderDatasetsTab();
        renderTrainedMemory();
        trainModal.classList.remove('hidden');
    });

document.getElementById('close-train-modal')
    .addEventListener('click', () => {
        trainModal.classList.add('hidden');
    });

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn')
            .forEach(b => b.classList.remove('active'));

        document.querySelectorAll('.tab-content')
            .forEach(c => c.classList.add('hidden'));

        btn.classList.add('active');

        const target = document.getElementById(
            btn.getAttribute('data-tab')
        );

        if (target) target.classList.remove('hidden');
    });
});

document.getElementById('save-train-btn')
    .addEventListener('click', () => {
        const q = document.getElementById('train-q').value.trim();
        const a = document.getElementById('train-a').value.trim();

        if (!q || !a) return;

        customMemory.unshift({
            q,
            a,
            timestamp: new Date().toISOString()
        });

        saveMemory();

        document.getElementById('train-q').value = '';
        document.getElementById('train-a').value = '';

        trainModal.classList.add('hidden');

        appendMessage(
            'jarvis',
            `Training complete. New knowledge saved:
            <br><strong>Q:</strong> ${escapeHTML(q)}
            <br><strong>A:</strong> ${escapeHTML(a)}`
        );

        speak('Training complete. New knowledge saved.');
    });

function renderDatasetsTab() {
    const container = document.getElementById('dataset-list');

    container.innerHTML = AGENTIC_DATASETS.map(ds => `
        <div class="ds-card">
            <strong>${escapeHTML(ds.name)}</strong>
            <span>${escapeHTML(ds.type)} // INDEXED</span>
            <p style="margin-top:4px;color:#bfefff;">
                ${escapeHTML(ds.summary.slice(0, 88))}...
            </p>
        </div>
    `).join('');
}

function renderTrainedMemory() {
    const list = document.getElementById('trained-list');

    if (!list) return;

    if (customMemory.length === 0) {
        list.innerHTML = `
            <p class="modal-desc">
                No custom Q&A knowledge saved yet.
                Add one in the training tab or type:
                <code>Train: question => answer</code>
            </p>
        `;
        return;
    }

    list.innerHTML = customMemory.map((item, idx) => `
        <div class="trained-item">
            <div>
                <strong style="color:var(--cyan);">
                    Q: ${escapeHTML(item.q)}
                </strong>
                <br>
                <span>A: ${escapeHTML(item.a)}</span>
            </div>
            <button class="del-item-btn"
                onclick="deleteTrainedItem(${idx})">
                DEL
            </button>
        </div>
    `).join('');
}

window.deleteTrainedItem = function(idx) {
    if (idx < 0 || idx >= customMemory.length) return;

    customMemory.splice(idx, 1);
    saveMemory();
};

// Export memory
document.getElementById('export-memory-btn')
    .addEventListener('click', () => {
        const payload = {
            model: 'JARVIS-Custom-Memory-v1',
            builtin_datasets: AGENTIC_DATASETS.map(d => d.name),
            custom_trained_pairs: customMemory
        };

        const blob = new Blob(
            [JSON.stringify(payload, null, 2)],
            { type: 'application/json' }
        );

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');

        a.href = url;
        a.download = 'jarvis-trained-memory.json';
        a.click();

        URL.revokeObjectURL(url);
    });

// Clear memory
document.getElementById('wipe-memory-btn')
    .addEventListener('click', () => {
        customMemory = [];
        saveMemory();
    });

// ============================================================
// 7. AI PROVIDER SETTINGS
// ============================================================

let savedProvider =
    localStorage.getItem('jarvis_provider') || 'groq';

let savedApiKey =
    localStorage.getItem('jarvis_api_key') || '';

function updateModeBadge() {
    if (!aiModeBadge) return;

    aiModeBadge.textContent = savedApiKey
        ? `SYS.ONLINE // ${savedProvider.toUpperCase()} + TRAINED CORE`
        : 'SYS.ONLINE // AGENT-TRAINED CORE';
}

updateModeBadge();

const apiModal = document.getElementById('api-modal');

document.getElementById('settings-btn')
    .addEventListener('click', () => {
        document.getElementById('api-provider').value = savedProvider;
        document.getElementById('api-key-input').value = savedApiKey;

        apiModal.classList.remove('hidden');
    });

document.getElementById('close-modal')
    .addEventListener('click', () => {
        apiModal.classList.add('hidden');
    });

document.getElementById('save-api-btn')
    .addEventListener('click', () => {
        savedProvider =
            document.getElementById('api-provider').value;

        savedApiKey =
            document.getElementById('api-key-input').value.trim();

        localStorage.setItem('jarvis_provider', savedProvider);
        localStorage.setItem('jarvis_api_key', savedApiKey);

        updateModeBadge();

        apiModal.classList.add('hidden');

        appendMessage(
            'jarvis',
            'External AI provider configured in this browser. Test the connection with a message.'
        );
    });

document.getElementById('clear-api-btn')
    .addEventListener('click', () => {
        savedApiKey = '';

        localStorage.removeItem('jarvis_api_key');

        updateModeBadge();

        apiModal.classList.add('hidden');

        appendMessage(
            'jarvis',
            'External AI connection disabled. Built-in tools remain available.'
        );
    });

// ============================================================
// 8. CLOCK AND BATTERY
// ============================================================

function updateClock() {
    const clock = document.getElementById('clock');

    if (clock) {
        clock.textContent =
            new Date().toTimeString().split(' ')[0];
    }
}

setInterval(updateClock, 1000);
updateClock();

if ('getBattery' in navigator) {
    navigator.getBattery()
        .then(battery => {
            const updateBat = () => {
                const status =
                    document.getElementById('battery-status');

                if (status) {
                    status.textContent =
                        `PWR: ${Math.round(battery.level * 100)}%`;
                }
            };

            updateBat();
            battery.addEventListener('levelchange', updateBat);
        })
        .catch(() => {});
}

// ============================================================
// 9. CHAT DISPLAY
// ============================================================

function appendMessage(sender, htmlText) {
    const div = document.createElement('div');

    div.className = `msg ${sender}`;

    const tag = sender === 'user'
        ? '[USER]'
        : '[J.A.R.V.I.S.]';

    const tagElement = document.createElement('span');
    tagElement.className = 'tag';
    tagElement.textContent = tag;

    div.appendChild(tagElement);

    const content = document.createElement('span');
    content.innerHTML = htmlText;

    div.appendChild(content);

    terminalLog.appendChild(div);
    terminalLog.scrollTop = terminalLog.scrollHeight;
}

// ============================================================
// 10. TEXT-TO-SPEECH
// ============================================================

function speak(plainText) {
    if (
        !ttsToggle ||
        !ttsToggle.checked ||
        !('speechSynthesis' in window)
    ) {
        return;
    }

    window.speechSynthesis.cancel();

    const cleanText = plainText
        .replace(/<[^>]*>/g, '')
        .slice(0, 420);

    const utterance = new SpeechSynthesisUtterance(cleanText);

    const voices = window.speechSynthesis.getVoices();

    const britishVoice =
        voices.find(v => v.lang.includes('en-GB') && v.name.includes('Male')) ||
        voices.find(v => v.lang.includes('en-GB')) ||
        voices.find(v => v.lang.includes('en'));

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

// ============================================================
// 11. MATH TOOL
// ============================================================

function tryMathTool(query) {
    const cleaned = query
        .replace(/^(what is|whats|what's|calculate|solve|compute|math)\s+/i, '')
        .replace(/×/g, '*')
        .replace(/÷/g, '/')
        .replace(/\^/g, '**')
        .replace(/\?/g, '')
        .trim();

    if (
        /^[0-9+\-*/().\s*]+$/.test(cleaned) &&
        /[0-9]/.test(cleaned) &&
        /[+\-*/]/.test(cleaned)
    ) {
        try {
            const result = Function(
                `"use strict"; return (${cleaned})`
            )();

            if (typeof result === 'number' && Number.isFinite(result)) {
                return `The calculated result is
                    <strong>${Number(result.toFixed(6))}</strong>.
                    <span style="color:var(--cyan);font-size:0.75rem;">
                    [Tool: MathEngine]</span>`;
            }
        } catch (error) {
            return null;
        }
    }

    return null;
}

// ============================================================
// 12. WEATHER TOOL
// ============================================================

async function fetchWeatherTool(city) {
    const geoRes = await fetch(
        `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`
    );

    if (!geoRes.ok) throw new Error('Weather location lookup failed.');

    const geoData = await geoRes.json();

    if (!geoData.results || geoData.results.length === 0) {
        return null;
    }

    const {
        latitude,
        longitude,
        name,
        country
    } = geoData.results[0];

    const wRes = await fetch(
        `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code`
    );

    if (!wRes.ok) throw new Error('Weather service unavailable.');

    const wData = await wRes.json();
    const cur = wData.current;

    const codes = {
        0: 'Clear sky',
        1: 'Mainly clear',
        2: 'Partly cloudy',
        3: 'Overcast',
        45: 'Fog',
        48: 'Depositing rime fog',
        51: 'Light drizzle',
        53: 'Moderate drizzle',
        55: 'Dense drizzle',
        61: 'Light rain',
        63: 'Moderate rain',
        65: 'Heavy rain',
        80: 'Rain showers',
        81: 'Moderate rain showers',
        82: 'Heavy rain showers',
        95: 'Thunderstorm'
    };

    return `Current weather for
        <strong>${escapeHTML(name)}, ${escapeHTML(country)}</strong>:
        ${codes[cur.weather_code] || 'Active conditions'},
        <strong>${cur.temperature_2m}°C</strong>,
        humidity ${cur.relative_humidity_2m}%,
        wind ${cur.wind_speed_10m} km/h.
        <span style="color:var(--cyan);font-size:0.75rem;">
        [Tool: OpenMeteo]</span>`;
}

// ============================================================
// 13. DICTIONARY TOOL
// ============================================================

async function fetchDictionaryTool(word) {
    const res = await fetch(
        `https://api.dictionaryapi.dev/api/v2/entries/en/${encodeURIComponent(word)}`
    );

    if (!res.ok) return null;

    const data = await res.json();

    const meaning = data[0]?.meanings?.[0];
    const def = meaning?.definitions?.[0]?.definition;

    return def
        ? `<strong>${escapeHTML(data[0].word)}</strong>
            (<em>${escapeHTML(meaning.partOfSpeech)}</em>):
            ${escapeHTML(def)}
            <span style="color:var(--cyan);font-size:0.75rem;">
            [Tool: Lexicon]</span>`
        : null;
}

// ============================================================
// 14. WIKIPEDIA TOOL
// ============================================================

async function fetchWikipediaTool(question) {
    const topic = question
        .replace(
            /^(who is|who was|who are|what is|what are|what was|tell me about|explain|search for|search|define|where is|when was)\s+/i,
            ''
        )
        .replace(/\?+$/, '')
        .trim();

    if (!topic) return null;

    const searchRes = await fetch(
        `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(topic)}&format=json&origin=*`
    );

    if (!searchRes.ok) return null;

    const searchData = await searchRes.json();
    const hits = searchData?.query?.search;

    if (!hits || hits.length === 0) return null;

    const sumRes = await fetch(
        `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(hits[0].title)}`
    );

    if (!sumRes.ok) return null;

    const sumData = await sumRes.json();

    if (sumData.extract) {
        const pageUrl =
            sumData.content_urls?.desktop?.page ||
            `https://en.wikipedia.org/wiki/${encodeURIComponent(hits[0].title)}`;

        return `${escapeHTML(sumData.extract)}
            <br><a href="${pageUrl}" target="_blank"
            rel="noopener noreferrer">[Read Full Dossier]</a>`;
    }

    return null;
}

// ============================================================
// 15. GROQ / OPENAI AI CONNECTION
// ============================================================

async function fetchLLMWithTrainedContext(userQuestion) {

    // FIXED: Correct newline handling in memory context.
    const memoryContext = customMemory.length > 0
        ? "User-trained custom facts you MUST prioritize:\n" +
          customMemory
              .map(m => `- Q: ${m.q} -> A: ${m.a}`)
              .join("\n")
        : "No additional user-trained facts are currently available.";

    const endpoint = savedProvider === 'openai'
        ? 'https://api.openai.com/v1/chat/completions'
        : 'https://api.groq.com/openai/v1/chat/completions';

    const model = savedProvider === 'openai'
        ? 'gpt-4o-mini'
        : 'llama-3.3-70b-versatile';

    const res = await fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${savedApiKey}`
        },
        body: JSON.stringify({
            model,
            messages: [
                {
                    role: 'system',
                    content: `You are J.A.R.V.I.S., a helpful, intelligent AI assistant.
                    Communicate clearly and professionally.
                    Use the user's trained facts when relevant.
                    Do not claim to have performed actions you have not performed.

                    USER-TRAINED MEMORY:
                    ${memoryContext}`
                },
                {
                    role: 'user',
                    content: userQuestion
                }
            ],
            temperature: 0.5
        })
    });

    const data = await res.json();

    if (!res.ok) {
        throw new Error(
            data.error?.message || `AI API error: HTTP ${res.status}`
        );
    }

    const answer = data.choices?.[0]?.message?.content;

    if (!answer) {
        throw new Error('The AI provider returned an empty response.');
    }

    // Escape model text before displaying it as HTML.
    return escapeHTML(answer).replace(/\n/g, '<br>');
}

// ============================================================
// 16. MASTER COMMAND ROUTER
// ============================================================

async function processCommand(rawCmd) {
    const trimmed = rawCmd.trim();

    if (!trimmed) return;

    const cmd = trimmed.toLowerCase();

    appendMessage('user', escapeHTML(trimmed));

    reactorCore.classList.add('thinking');
    coreState.textContent = 'ROUTING';

    let response = '';

    try {

        // INLINE TRAINING
        if (/^train:\s*.+\s*=>\s*.+$/i.test(trimmed)) {

            const parts = trimmed
                .replace(/^train:\s*/i, '')
                .split('=>');

            const q = parts[0].trim();
            const a = parts.slice(1).join('=>').trim();

            customMemory.unshift({
                q,
                a,
                timestamp: new Date().toISOString()
            });

            saveMemory();

            response = `Knowledge saved. When you ask
                <em>"${escapeHTML(q)}"</em>,
                I will use your trained answer:
                <strong>${escapeHTML(a)}</strong>`;
        }

        // CLEAR TERMINAL
        else if (
            cmd === 'clear' ||
            cmd === 'clear terminal'
        ) {
            terminalLog.innerHTML = '';
            response = 'Terminal logs cleared.';
        }

        // TIME
        else if (
            /^(what time|current time|time is it)/i.test(cmd)
        ) {
            response = `The current local time is
                <strong>${new Date().toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit'
                })}</strong>.`;
        }

        // DATE
        else if (
            /^(what is the date|today's date|what day is it)/i.test(cmd)
        ) {
            response = `Today is
                <strong>${new Date().toLocaleDateString('en-US', {
                    weekday: 'long',
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                })}</strong>.`;
        }

        // OPEN WEBSITE
        else if (cmd.startsWith('open ')) {

            const site = cmd.replace('open ', '').trim();

            const shortcuts = {
                github: 'https://github.com',
                youtube: 'https://www.youtube.com',
                google: 'https://www.google.com',
                huggingface: 'https://huggingface.co'
            };

            const url = shortcuts[site] ||
                (site.startsWith('http')
                    ? site
                    : `https://www.${site}.com`);

            window.open(url, '_blank', 'noopener');

            response = `Opening <strong>${escapeHTML(site)}</strong>.`;
        }

        else {

            // A. CUSTOM TRAINED MEMORY
            const trainedReply = findTrainedMatch(trimmed);

            if (trainedReply) {
                response = trainedReply;
            }

            // B. EXTERNAL AI PROVIDER
            else if (savedApiKey) {
                response = await fetchLLMWithTrainedContext(trimmed);
            }

            // C. BUILT-IN TOOLS
            else {

                const mathAns = tryMathTool(trimmed);

                if (mathAns) {
                    response = mathAns;
                }

                else if (
                    cmd.includes('weather in ') ||
                    cmd.startsWith('weather ')
                ) {
                    const city = trimmed
                        .replace(/.*weather\s+(in\s+)?/i, '')
                        .replace(/\?+$/, '')
                        .trim();

                    response = (await fetchWeatherTool(city)) ||
                        `No weather information found for ${escapeHTML(city)}.`;
                }

                else if (cmd.startsWith('define ')) {

                    const word = trimmed
                        .replace(/^define\s+/i, '')
                        .replace(/\?+$/, '')
                        .trim();

                    response =
                        (await fetchDictionaryTool(word)) ||
                        (await fetchWikipediaTool(trimmed)) ||
                        'No definition found.';
                }

                else {

                    const wikiAns =
                        await fetchWikipediaTool(trimmed);

                    response = wikiAns ||
                        `I could not find a direct match.
                        You can teach me using:
                        <br><code>Train: ${escapeHTML(trimmed)}
                        =&gt; [your answer]</code>`;
                }
            }
        }

    } catch (err) {

        console.error('JARVIS command error:', err);

        response = `System exception:
            <strong>${escapeHTML(err.message)}</strong>
            <br>Check your internet connection and AI configuration.`;
    }

    reactorCore.classList.remove('thinking');
    coreState.textContent = 'STANDBY';

    appendMessage('jarvis', response);
    speak(response);
}

// ============================================================
// 17. FORM SUBMISSION
// ============================================================

cmdForm.addEventListener('submit', async (event) => {

    event.preventDefault();

    const message = cmdInput.value.trim();

    if (!message) return;

    cmdInput.value = '';

    await processCommand(message);
});

// ============================================================
// 18. QUICK COMMAND BUTTONS
// ============================================================

document.querySelectorAll('.cmd-btn[data-cmd]').forEach(btn => {

    btn.addEventListener('click', () => {

        processCommand(
            btn.getAttribute('data-cmd')
        );

    });

});

// ============================================================
// 19. SPEECH RECOGNITION
// ============================================================

const SpeechRecognition =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;

if (SpeechRecognition) {

    const recognition = new SpeechRecognition();

    recognition.lang = 'en-US';
    recognition.interimResults = false;

    micBtn.addEventListener('click', () => {

        try {
            recognition.start();
        } catch (error) {
            console.warn('Speech recognition:', error);
        }

    });

    recognition.onstart = () => {

        reactorCore.classList.add('listening');
        micBtn.classList.add('active');

        micLabel.textContent = 'LISTENING...';
        coreState.textContent = 'LISTENING';

    };

    recognition.onresult = (event) => {

        processCommand(
            event.results[0][0].transcript
        );

    };

    recognition.onerror = (event) => {

        appendMessage(
            'jarvis',
            `Audio sensor alert: ${escapeHTML(event.error)}.`
        );

    };

    recognition.onend = () => {

        reactorCore.classList.remove('listening');
        micBtn.classList.remove('active');

        micLabel.textContent = 'INITIALIZE VOICE';

        if (coreState.textContent === 'LISTENING') {
            coreState.textContent = 'STANDBY';
        }

    };

} else {

    micBtn.style.opacity = '0.5';
    micLabel.textContent = 'VOICE UNSUPPORTED';

}

// ============================================================
// 20. SYSTEM INITIALIZATION
// ============================================================

console.log('J.A.R.V.I.S. system initialized.');
console.log('Agentic dataset catalogue loaded:', AGENTIC_DATASETS.length);
console.log('Custom memory entries:', customMemory.length);
console.log('AI provider:', savedProvider);
console.log('External AI configured:', Boolean(savedApiKey));
