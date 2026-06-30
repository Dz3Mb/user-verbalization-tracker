/**
 * Verbalization Tracker — frontend logic
 * Records audio via MediaRecorder, uploads to backend, displays results.
 */

const API_URL = "http://localhost:8000";

// DOM elements
const consentBanner = document.getElementById("consent-banner");
const consentBtn = document.getElementById("consent-btn");
const recorderSection = document.getElementById("recorder");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const languageSelect = document.getElementById("language-select");
const statusEl = document.getElementById("status");
const timerEl = document.getElementById("timer");
const dropZone = document.getElementById("drop-zone");
const browseBtn = document.getElementById("browse-btn");
const fileInput = document.getElementById("file-input");
const resultsSection = document.getElementById("results");
const transcriptionText = document.getElementById("transcription-text");
const entitiesList = document.getElementById("entities-list");
const relationsList = document.getElementById("relations-list");
const rawJson = document.getElementById("raw-json");

let mediaRecorder = null;
let audioChunks = [];
let timerInterval = null;
let recordingStart = 0;

// --- Consent ---

consentBtn.addEventListener("click", () => {
    consentBanner.classList.add("hidden");
    recorderSection.classList.remove("hidden");
});

// --- Recording ---

startBtn.addEventListener("click", async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });

        // Prefer webm; fall back to whatever the browser supports
        const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
            ? "audio/webm;codecs=opus"
            : "audio/webm";

        mediaRecorder = new MediaRecorder(stream, { mimeType });
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
            stream.getTracks().forEach((t) => t.stop());
            clearInterval(timerInterval);
            handleRecordingComplete();
        };

        mediaRecorder.start();
        recordingStart = Date.now();
        updateTimer();
        timerInterval = setInterval(updateTimer, 500);

        setStatus("Recording…", "");
        startBtn.disabled = true;
        stopBtn.disabled = false;
    } catch (err) {
        setStatus("Microphone access denied or unavailable.", "error");
        console.error(err);
    }
});

stopBtn.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;
});

function updateTimer() {
    const elapsed = Math.floor((Date.now() - recordingStart) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
    const secs = String(elapsed % 60).padStart(2, "0");
    timerEl.textContent = `${mins}:${secs}`;
}

// --- Upload & display ---

const MAX_FILE_SIZE = 25 * 1024 * 1024;

async function uploadAudio(blob, filename) {
    if (!blob || blob.size === 0) {
        setStatus("Empty audio. Try again.", "error");
        return;
    }
    if (blob.size > MAX_FILE_SIZE) {
        setStatus("File too large (max 25 MB).", "error");
        return;
    }

    setStatus(`Uploading and analyzing ${filename}… this may take a moment.`, "");
    startBtn.disabled = true;
    if (browseBtn) browseBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", blob, filename);
    formData.append("language", languageSelect ? languageSelect.value : "auto");

    try {
        const response = await fetch(`${API_URL}/analyze`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errBody = await response.json().catch(() => ({}));
            throw new Error(errBody.detail || `Server error ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);
        setStatus("Analysis complete.", "success");
    } catch (err) {
        setStatus(`Error: ${err.message}`, "error");
        console.error(err);
    } finally {
        startBtn.disabled = false;
        if (browseBtn) browseBtn.disabled = false;
    }
}

async function handleRecordingComplete() {
    const blob = new Blob(audioChunks, { type: "audio/webm" });
    await uploadAudio(blob, "recording.webm");
}

// --- File upload (drag & drop + browse) ---

function handleSelectedFile(file) {
    if (!file) return;
    // Be lenient: some files (e.g. .opus) have empty `type` from the OS.
    if (file.type && !file.type.startsWith("audio/")) {
        setStatus(`Unsupported file type: ${file.type}`, "error");
        return;
    }
    uploadAudio(file, file.name);
}

if (dropZone) {
    ["dragenter", "dragover"].forEach((ev) => {
        dropZone.addEventListener(ev, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        });
    });
    ["dragleave", "dragend", "drop"].forEach((ev) => {
        dropZone.addEventListener(ev, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        });
    });
    dropZone.addEventListener("drop", (e) => {
        const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        handleSelectedFile(file);
    });
}

if (browseBtn && fileInput) {
    browseBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
        handleSelectedFile(fileInput.files && fileInput.files[0]);
        fileInput.value = ""; // allow re-selecting the same file
    });
}

// Prevent the browser from navigating away if a file is dropped outside the zone.
["dragover", "drop"].forEach((ev) => {
    window.addEventListener(ev, (e) => e.preventDefault());
});

function displayResults(data) {
    resultsSection.classList.remove("hidden");

    // Transcription
    transcriptionText.textContent = data.transcription?.text || "(no text)";
    const detected = data.transcription?.language;
    const langProb = data.transcription?.language_probability;
    const model = data.transcription?.model;
    if (detected || model) {
        const note = document.createElement("div");
        note.className = "lang-note";
        const bits = [];
        if (detected) {
            const probStr = langProb != null ? ` (${Math.round(langProb * 100)}%)` : "";
            bits.push(`Detected language: ${detected}${probStr}`);
        }
        if (model) bits.push(`Whisper model: ${model}`);
        note.textContent = bits.join("  ·  ");
        transcriptionText.insertAdjacentElement("afterend", note);
        // Avoid stacking notes across runs
        const prev = transcriptionText.parentElement.querySelectorAll(".lang-note");
        prev.forEach((n, i) => { if (i < prev.length - 1) n.remove(); });
    }

    // Entities
    entitiesList.innerHTML = "";
    const entities = data.entities || [];
    if (entities.length === 0) {
        entitiesList.textContent = "No entities found.";
    } else {
        entities.forEach((ent) => {
            const tag = document.createElement("span");
            tag.className = "tag";

            let html =
                escapeHtml(ent.text) +
                `<span class="label">${escapeHtml(ent.label)}</span>`;

            // Knowledge-graph type (from Wikidata P31), independent of spaCy's
            // NER label. Shown as a separate badge so mismatches are visible.
            const wd = ent.wikidata;
            if (wd && wd.kg_type) {
                const cls = wd.label_mismatch ? "kg-type mismatch" : "kg-type";
                const title = wd.label_mismatch
                    ? `Wikidata says: ${wd.kg_type} (does not match spaCy label ${ent.label})`
                    : `Wikidata type: ${wd.kg_type}`;
                html += `<span class="${cls}" title="${escapeHtml(title)}">${escapeHtml(wd.kg_type)}</span>`;
            }

            // Knowledge-graph links (Wikidata / DBpedia), when available
            const links = [];
            if (wd && wd.url) {
                const tip = wd.description
                    ? `${wd.label} — ${wd.description}`
                    : wd.label || wd.id;
                links.push(
                    `<a class="kg wd" href="${escapeHtml(wd.url)}" target="_blank" rel="noopener" title="${escapeHtml(tip)}">${escapeHtml(wd.id)}</a>`
                );
            }
            if (ent.dbpedia && ent.dbpedia.uri) {
                links.push(
                    `<a class="kg db" href="${escapeHtml(ent.dbpedia.uri)}" target="_blank" rel="noopener" title="${escapeHtml(ent.dbpedia.uri)}">DBpedia</a>`
                );
            }
            if (links.length > 0) {
                html += `<span class="kg-links">${links.join("")}</span>`;
            }

            tag.innerHTML = html;
            entitiesList.appendChild(tag);
        });
    }

    // Relations
    relationsList.innerHTML = "";
    const relations = data.relations || [];
    if (relations.length === 0) {
        const li = document.createElement("li");
        li.textContent = "No relations found.";
        relationsList.appendChild(li);
    } else {
        relations.forEach((rel) => {
            const li = document.createElement("li");
            li.innerHTML =
                `<span class="triple">${escapeHtml(rel.subject)} → ${escapeHtml(rel.predicate)} → ${escapeHtml(rel.object)}</span>` +
                `<span class="sentence">"${escapeHtml(rel.sentence)}"</span>`;
            relationsList.appendChild(li);
        });
    }

    // Raw JSON
    rawJson.textContent = JSON.stringify(data, null, 2);
}

// --- Helpers ---

function setStatus(msg, type) {
    statusEl.textContent = msg;
    statusEl.className = "status" + (type ? ` ${type}` : "");
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}
