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
const statusEl = document.getElementById("status");
const timerEl = document.getElementById("timer");
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
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

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

async function handleRecordingComplete() {
    const blob = new Blob(audioChunks, { type: "audio/webm" });

    if (blob.size === 0) {
        setStatus("Recording was empty. Try again.", "error");
        return;
    }

    setStatus("Uploading and analyzing… this may take a moment.", "");
    startBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", blob, "recording.webm");

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
    }
}

function displayResults(data) {
    resultsSection.classList.remove("hidden");

    // Transcription
    transcriptionText.textContent = data.transcription?.text || "(no text)";

    // Entities
    entitiesList.innerHTML = "";
    const entities = data.entities || [];
    if (entities.length === 0) {
        entitiesList.textContent = "No entities found.";
    } else {
        entities.forEach((ent) => {
            const tag = document.createElement("span");
            tag.className = "tag";
            tag.innerHTML =
                escapeHtml(ent.text) +
                `<span class="label">${escapeHtml(ent.label)}</span>`;
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
