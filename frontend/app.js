/**
 * app.js — AI Study Tutor frontend logic
 *
 * All communication with the backend goes through the functions below.
 * The FastAPI server must be running at BACKEND_URL (default: localhost:8000).
 *
 * API calls made here:
 *   GET  /health        — check backend is alive on page load
 *   POST /upload        — upload the selected PDF
 *   POST /index         — embed the uploaded PDF into ChromaDB
 *   POST /ask           — ask a question using the RAG pipeline
 *   POST /quiz          — generate MCQ questions from the indexed document
 *
 * No API keys are stored or used here.
 * All AI work happens on the FastAPI backend.
 */

"use strict";

// ============================================================
// CONFIGURATION
// Change BACKEND_URL if the FastAPI server runs on a different port.
// ============================================================

const BACKEND_URL = "https://saviora-ai.onrender.com";


// ============================================================
// STATE
// Simple object — no framework needed.
// ============================================================

const state = {
  selectedFile:     null,   // File object chosen by the user
  uploadedFilename: null,   // filename returned by /upload
  questionsAsked: 0,
  docsIndexed: 0,
  quizBestScore: null
  
};


// ============================================================
// DOM REFERENCES
// ============================================================

const dropZone       = document.getElementById("drop-zone");
const dropZoneText   = document.getElementById("drop-zone-text");
const fileInput      = document.getElementById("file-input");
const chosenFile     = document.getElementById("chosen-file");
const btnUpload      = document.getElementById("btn-upload");
const btnIndex       = document.getElementById("btn-index");
const uploadStatus   = document.getElementById("upload-status");
const indexStatus    = document.getElementById("index-status");

const questionInput  = document.getElementById("question-input");
const btnAsk         = document.getElementById("btn-ask");
const answerArea     = document.getElementById("answer-area");
const answerText     = document.getElementById("answer-text");
const sourcesArea    = document.getElementById("sources-area");
const sourcesList    = document.getElementById("sources-list");
const askError       = document.getElementById("ask-error");

const backendStatus  = document.getElementById("backend-status");




// ============================================================
// HELPER — show a status box with a colour class
// type: "info" | "success" | "warning" | "error"
// ============================================================

function showStatus(element, message, type) {
  element.textContent = message;
  element.className = `status-box ${type}`;  // clears previous type
  element.classList.remove("hidden");
}

function hideStatus(element) {
  element.classList.add("hidden");
}


// ============================================================
// HELPER — update the progress counters
// ============================================================

function updateStats() {
  const statQuestions = document.getElementById("stat-questions");
  const statDocs = document.getElementById("stat-docs");
  const statQuizBest = document.getElementById("stat-quiz-best");

  if (statQuestions) {
    statQuestions.textContent = state.questionsAsked;
  }

  if (statDocs) {
    statDocs.textContent = state.docsIndexed;
  }

  if (statQuizBest) {
    statQuizBest.textContent = state.quizBestScore ?? "—";
  }
}


// ============================================================
// HEALTH CHECK
// Runs once on page load to show whether the backend is reachable.
// ============================================================

async function checkBackendHealth() {
  try {
    // GET /health → { "status": "ok" }
    const res = await fetch(`${BACKEND_URL}/health`);
    if (res.ok) {
      backendStatus.className = "status-dot status-ok";
      backendStatus.title = "Backend is running";
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    backendStatus.className = "status-dot status-error";
    backendStatus.title = `Backend unreachable: ${err.message}`;
    console.error("Backend health check failed:", err);
  }
}


// ============================================================
// FILE SELECTION
// Handles both the click-to-browse and drag-and-drop flows.
// ============================================================

function handleFileSelected(file) {
  if (!file) return;

  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showStatus(uploadStatus, "Only PDF files are supported.", "error");
    return;
  }

  state.selectedFile = file;

  // Show the filename below the drop zone
  chosenFile.textContent = `Selected: ${file.name}`;
  chosenFile.classList.remove("hidden");

  // Enable the Upload button; disable Index until upload succeeds
  btnUpload.disabled = false;
  btnIndex.disabled  = true;

  // Reset previous status messages when a new file is chosen
  hideStatus(uploadStatus);
  hideStatus(indexStatus);
}

// Click on the drop zone label opens the hidden file input
fileInput.addEventListener("change", () => {
  handleFileSelected(fileInput.files[0]);
});

// Drag-and-drop onto the drop zone
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  handleFileSelected(file);
});


// ============================================================
// UPLOAD
// POST /upload  multipart/form-data
// Response: { "filename": "...", "message": "...", "size_bytes": N }
// ============================================================

btnUpload.addEventListener("click", async () => {
  if (!state.selectedFile) return;

  // Visual feedback: disable button and show spinner text
  btnUpload.disabled = true;
  showStatus(uploadStatus, "Uploading…", "info");

  // Build the multipart form payload
  const formData = new FormData();
  formData.append("file", state.selectedFile);

  try {
    // POST /upload — send the PDF bytes to the backend
    const res = await fetch(`${BACKEND_URL}/upload`, {
      method: "POST",
      body: formData,
      // Note: do NOT set Content-Type header manually when using FormData.
      // The browser sets it automatically with the correct boundary.
    });

    const data = await res.json();

    if (!res.ok) {
      // The backend returns { "detail": "..." } on errors
      showStatus(uploadStatus, `Upload failed: ${data.detail || res.statusText}`, "error");
      return;
    }

    // Upload succeeded — store the filename for the /index call
    state.uploadedFilename = data.filename;

    showStatus(
      uploadStatus,
      `✓ Uploaded "${data.filename}" (${(data.size_bytes / 1024).toFixed(1)} KB)`,
      "success"
    );

    // Enable the Index button
    btnIndex.disabled = false;

  } catch (err) {
    // Network error or backend not running
    showStatus(uploadStatus, `Network error: ${err.message}`, "error");
    console.error("Upload error:", err);
    btnUpload.disabled = false;  // allow retry
  }
});


// ============================================================
// INDEX
// POST /index  { "filename": "..." }
// Response: { "status": "indexed"|"existing"|"empty", "chunks": N, "message": "..." }
//
// Note: indexing embeds all chunks via the Gemini API.
// On large documents this can take 1–3 minutes due to rate limiting.
// ============================================================

btnIndex.addEventListener("click", async () => {
  if (!state.uploadedFilename) return;

  btnIndex.disabled = true;
  showStatus(
    indexStatus,
    "Indexing document… This may take a minute or two for large PDFs.",
    "info"
  );

  try {
    // POST /index — triggers the RAG embedding pipeline on the backend
    const res = await fetch(`${BACKEND_URL}/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: state.uploadedFilename }),
    });

    const data = await res.json();

    if (res.status === 429) {
      // Quota exhausted — clear user message from backend
      showStatus(indexStatus, `⏳ ${data.detail}`, "warning");
      btnIndex.disabled = false;  // allow retry later
      return;
    }

    if (!res.ok) {
      showStatus(indexStatus, `Indexing failed: ${data.detail || res.statusText}`, "error");
      btnIndex.disabled = false;
      return;
    }

    // Success
    const type = data.status === "existing" ? "info" : "success";
    showStatus(indexStatus, `✓ ${data.message}`, type);

    if (data.status === "indexed") {
      state.docsIndexed += 1;
      updateStats();
    }

  } catch (err) {
    showStatus(indexStatus, `Network error: ${err.message}`, "error");
    console.error("Index error:", err);
    btnIndex.disabled = false;
  }
});


// ============================================================
// ASK
// POST /ask  { "question": "..." }
// Response: { "question": "...", "answer": "...", "sources": [...] }
// ============================================================

async function runAsk() {
  const question = questionInput.value.trim();
  if (!question) return;

  // Disable input and button while waiting
  btnAsk.disabled       = true;
  questionInput.disabled = true;

  // Clear previous answer and error
  answerArea.classList.add("hidden");
  askError.classList.add("hidden");

  // Show a loading indicator inside the answer box
  answerText.textContent = "Searching your notes…";
  answerArea.classList.remove("hidden");
  sourcesArea.classList.add("hidden");
  sourcesList.innerHTML = "";

  try {
    // POST /ask — sends the question to the RAG pipeline
    const res = await fetch(`${BACKEND_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();

    if (!res.ok) {
      answerArea.classList.add("hidden");
      askError.textContent = `Error: ${data.detail || res.statusText}`;
      askError.classList.remove("hidden");
      return;
    }

    // Show the answer
    answerText.textContent = data.answer;

    // Show sources if any were returned
    if (data.sources && data.sources.length > 0) {
      sourcesList.innerHTML = "";
      data.sources.forEach((src) => {
        const li = document.createElement("li");
        li.textContent = `${src.source}  —  p.${src.page}`;
        sourcesList.appendChild(li);
      });
      sourcesArea.classList.remove("hidden");
    }

    // Update stats
    state.questionsAsked += 1;
    updateStats();

  } catch (err) {
    answerArea.classList.add("hidden");
    askError.textContent = `Network error: ${err.message}`;
    askError.classList.remove("hidden");
    console.error("Ask error:", err);
  } finally {
    // Re-enable input regardless of outcome
    btnAsk.disabled        = false;
    questionInput.disabled = false;
    questionInput.focus();
  }
}

// Ask on button click
btnAsk.addEventListener("click", runAsk);

// Ask on Enter key in the input field
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") runAsk();
});


// ============================================================
// INIT — runs when the page loads
// ============================================================

(function init() {

  checkBackendHealth();
})();


// ============================================================
// QUIZ
//
// Flow:
//   1. User clicks "Start Quiz →" on the mode card → openQuiz()
//   2. User picks number of questions and clicks "Generate Quiz"
//      → POST /quiz  { "num_questions": N }
//      → backend returns { "questions": [...] }
//   3. renderQuestion() shows one question at a time
//   4. User picks an option → Submit Answer → feedback shown
//   5. Next → advances to next question
//   6. After last question → results screen
//   7. "Restart Quiz" → back to setup screen (same questions)
//   8. "← Back" → hides the quiz section, shows study modes
// ============================================================

// DOM refs — quiz section
const sectionQuiz        = document.getElementById("section-quiz");
const sectionModes       = document.getElementById("section-modes");
const quizSetup          = document.getElementById("quiz-setup");
const quizNumQuestionsEl = document.getElementById("quiz-num-questions");
const btnGenerateQuiz    = document.getElementById("btn-generate-quiz");
const quizGenStatus      = document.getElementById("quiz-gen-status");

const quizQuestionScreen = document.getElementById("quiz-question-screen");
const quizCounter        = document.getElementById("quiz-counter");
const quizScoreLive      = document.getElementById("quiz-score-live");
const quizProgressBar    = document.getElementById("quiz-progress-bar");
const quizQuestionText   = document.getElementById("quiz-question-text");
const quizOptions        = document.getElementById("quiz-options");
const btnSubmitAnswer    = document.getElementById("btn-submit-answer");
const btnNextQuestion    = document.getElementById("btn-next-question");
const quizFeedback       = document.getElementById("quiz-feedback");
const quizExplanation    = document.getElementById("quiz-explanation");

const quizResultsScreen  = document.getElementById("quiz-results-screen");
const quizFinalNumber    = document.getElementById("quiz-final-number");
const quizFinalMessage   = document.getElementById("quiz-final-message");
const btnRestartQuiz     = document.getElementById("btn-restart-quiz");


// Quiz runtime state — reset each time a new quiz is generated
let quizQuestions   = [];   // array of question objects from /quiz
let quizIndex       = 0;    // current question index
let quizScore       = 0;    // number of correct answers so far
let selectedOption  = null; // the option button the user clicked


// ── Open / close ────────────────────────────────────────────

function openQuiz() {
  // Hide the modes section, show the quiz section
  sectionModes.classList.add("hidden");
  sectionQuiz.classList.remove("hidden");

  // Always start at the setup screen
  showQuizScreen("setup");

  // Scroll to the quiz section
  sectionQuiz.scrollIntoView({ behavior: "smooth", block: "start" });
}

function closeQuiz() {
  sectionQuiz.classList.add("hidden");
  sectionModes.classList.remove("hidden");
  sectionModes.scrollIntoView({ behavior: "smooth", block: "start" });
}


// ── Screen switcher ──────────────────────────────────────────
// which: "setup" | "question" | "results"

function showQuizScreen(which) {
  quizSetup.classList.toggle("hidden",          which !== "setup");
  quizQuestionScreen.classList.toggle("hidden", which !== "question");
  quizResultsScreen.classList.toggle("hidden",  which !== "results");
}


// ── Generate quiz ────────────────────────────────────────────

btnGenerateQuiz.addEventListener("click", async () => {
  const numQ = parseInt(quizNumQuestionsEl.value, 10);

  btnGenerateQuiz.disabled = true;
  showStatus(quizGenStatus, `Generating ${numQ} questions from your document…`, "info");

  try {
    // POST /quiz — backend retrieves context from ChromaDB and
    // asks Gemini to generate MCQs grounded in that context.
    const res  = await fetch(`${BACKEND_URL}/quiz`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ num_questions: numQ }),
    });

    const data = await res.json();

    if (!res.ok) {
      showStatus(quizGenStatus, `Error: ${data.detail || res.statusText}`, "error");
      btnGenerateQuiz.disabled = false;
      return;
    }

    quizQuestions = data.questions;

    if (!quizQuestions || quizQuestions.length === 0) {
      showStatus(quizGenStatus, "No questions were returned. Try again.", "warning");
      btnGenerateQuiz.disabled = false;
      return;
    }

    hideStatus(quizGenStatus);
    btnGenerateQuiz.disabled = false;

    // Reset runtime state and show the first question
    quizIndex  = 0;
    quizScore  = 0;
    startQuiz();

  } catch (err) {
    showStatus(quizGenStatus, `Network error: ${err.message}`, "error");
    console.error("Quiz generate error:", err);
    btnGenerateQuiz.disabled = false;
  }
});


// ── Start / restart ──────────────────────────────────────────

function startQuiz() {
  quizIndex = 0;
  quizScore = 0;
  showQuizScreen("question");
  renderQuestion();
}

btnRestartQuiz.addEventListener("click", () => {
  showQuizScreen("setup");
});


// ── Render one question ──────────────────────────────────────

function renderQuestion() {
  const q     = quizQuestions[quizIndex];
  const total = quizQuestions.length;

  // Progress bar and counters
  quizCounter.textContent    = `Question ${quizIndex + 1} of ${total}`;
  quizScoreLive.textContent  = `Score: ${quizScore}/${quizIndex}`;
  quizProgressBar.style.width = `${((quizIndex) / total) * 100}%`;

  // Question text
  quizQuestionText.textContent = q.question;

  // Reset per-question UI
  selectedOption = null;
  btnSubmitAnswer.disabled = true;
  btnNextQuestion.classList.add("hidden");
  quizFeedback.classList.add("hidden");
  quizExplanation.classList.add("hidden");

  // Render option buttons
  const letters = ["A", "B", "C", "D"];
  quizOptions.innerHTML = "";

  q.options.forEach((optionText, i) => {
    const btn = document.createElement("button");
    btn.className = "quiz-option";
    btn.type      = "button";

    // Letter badge + text
    btn.innerHTML = `
      <span class="quiz-option-letter">${letters[i]}.</span>
      <span>${optionText}</span>
    `;

    btn.addEventListener("click", () => selectOption(btn, optionText));
    quizOptions.appendChild(btn);
  });
}


// ── Option selection ─────────────────────────────────────────

function selectOption(clickedBtn, optionText) {
  // Deselect previous selection
  quizOptions.querySelectorAll(".quiz-option").forEach(b => b.classList.remove("selected"));

  clickedBtn.classList.add("selected");
  selectedOption = optionText;
  btnSubmitAnswer.disabled = false;
}


// ── Submit answer ────────────────────────────────────────────

btnSubmitAnswer.addEventListener("click", () => {
  if (!selectedOption) return;

  const q          = quizQuestions[quizIndex];
  const isCorrect  = selectedOption === q.answer;
  const total      = quizQuestions.length;

  if (isCorrect) quizScore++;

  // Colour all option buttons: correct = green, selected-wrong = red
  quizOptions.querySelectorAll(".quiz-option").forEach((btn, i) => {
    const optText = q.options[i];
    btn.disabled  = true;

    if (optText === q.answer) {
      btn.classList.add("correct");
    } else if (btn.classList.contains("selected")) {
      btn.classList.add("wrong");
    }
  });

  // Feedback message
  showStatus(
    quizFeedback,
    isCorrect ? "✓ Correct!" : `✗ Incorrect. The correct answer is: ${q.answer}`,
    isCorrect ? "success" : "error"
  );

  // Explanation
  quizExplanation.innerHTML = `<strong>Explanation:</strong> ${q.explanation}`;
  quizExplanation.classList.remove("hidden");

  // Update live score display
  quizScoreLive.textContent = `Score: ${quizScore}/${quizIndex + 1}`;

  // Hide Submit, show Next (or Finish)
  btnSubmitAnswer.classList.add("hidden");

  if (quizIndex < total - 1) {
    btnNextQuestion.textContent = "Next →";
  } else {
    btnNextQuestion.textContent = "See Results";
  }
  btnNextQuestion.classList.remove("hidden");
});


// ── Next question / finish ────────────────────────────────────

btnNextQuestion.addEventListener("click", () => {
  const total = quizQuestions.length;

  if (quizIndex < total - 1) {
    quizIndex++;
    btnSubmitAnswer.classList.remove("hidden");
    renderQuestion();
  } else {
    showResults();
  }
});


// ── Results screen ────────────────────────────────────────────

function showResults() {
  const total   = quizQuestions.length;
  const percent = Math.round((quizScore / total) * 100);

  // Final progress bar = 100%
  quizProgressBar.style.width = "100%";

  // Score display
  quizFinalNumber.textContent = `${quizScore} / ${total}`;

  // Message based on percentage
  let message = "";
  if (percent === 100)       message = "🎉 Perfect score! Outstanding work.";
  else if (percent >= 80)    message = "🌟 Excellent! You really know this material.";
  else if (percent >= 60)    message = "👍 Good effort. Review the missed questions.";
  else if (percent >= 40)    message = "📖 Keep studying — you're making progress.";
  else                       message = "💪 Don't give up. Try again after reviewing your notes.";

  quizFinalMessage.textContent = message;

  // Update best score in progress section
  const scoreStr = `${quizScore}/${total}`;
  if (
    state.quizBestScore === null ||
    quizScore / total > parseFraction(state.quizBestScore)
  ) {
    state.quizBestScore = scoreStr;
    updateStats();
  }

  showQuizScreen("results");
}

// Helper: parse "3/5" → 0.6
function parseFraction(str) {
  const [a, b] = str.split("/").map(Number);
  return b > 0 ? a / b : 0;
}


// ============================================================
// PASSAGE RECALL
//
// Flow:
//   1. User clicks "Start Recall →" on the mode card → openRecall()
//   2. User picks difficulty and clicks "Start Recall"
//      → POST /recall/start { "difficulty": "medium" }
//      → backend retrieves context from ChromaDB, generates passage
//      → response: { session_id, passage, duration_seconds }
//   3. Reading screen: passage is shown + 30-second countdown timer
//      - Timer runs entirely in JS (no backend sleep)
//      - When timer hits 0: passage is hidden, recall input is shown
//   4. User types their recall → Submit Recall
//      → POST /recall/evaluate { "session_id": "...", "response": "..." }
//      → NOTE: passage is NOT sent — server retrieves it by session_id
//      → response: { score, summary, remembered_points, missed_points, feedback }
//   5. Results screen shows score gauge, remembered/missed points, feedback
//   6. "Try Again" → back to setup
//   7. "← Back" → hides recall section, shows study modes
// ============================================================

// ── DOM refs ─────────────────────────────────────────────────

const sectionRecall       = document.getElementById("section-recall");
const recallSetup         = document.getElementById("recall-setup");
const recallDifficultyEl  = document.getElementById("recall-difficulty");
const btnStartRecall      = document.getElementById("btn-start-recall");
const recallSetupStatus   = document.getElementById("recall-setup-status");

const recallReading       = document.getElementById("recall-reading");
const recallTimerDisplay  = document.getElementById("recall-timer-display");
const recallTimerNumber   = document.getElementById("recall-timer-number");
const recallTimerRing     = document.getElementById("recall-timer-ring");
const recallPassageBox    = document.getElementById("recall-passage-box");

const recallInput         = document.getElementById("recall-input");
const recallTextarea      = document.getElementById("recall-textarea");
const btnSubmitRecall     = document.getElementById("btn-submit-recall");
const recallEvalStatus    = document.getElementById("recall-eval-status");

const recallResults       = document.getElementById("recall-results");
const recallScoreRing     = document.getElementById("recall-score-ring");
const recallScoreNumber   = document.getElementById("recall-score-number");
const recallSummary       = document.getElementById("recall-summary");
const recallRememberedList= document.getElementById("recall-remembered-list");
const recallMissedList    = document.getElementById("recall-missed-list");
const recallFeedbackBox   = document.getElementById("recall-feedback-box");
const btnRecallTryAgain   = document.getElementById("btn-recall-try-again");


// ── Runtime state ─────────────────────────────────────────────

let recallSessionId    = null;   // returned by /recall/start
let recallTimerHandle  = null;   // setInterval handle
let recallSecondsLeft  = 30;     // countdown state

// Timer SVG circumference for r=34: 2π×34 ≈ 213.628
const RECALL_TIMER_CIRCUMFERENCE = 2 * Math.PI * 34;  // ≈ 213.6


// ── Screen switcher ───────────────────────────────────────────
// which: "setup" | "reading" | "input" | "results"

function showRecallScreen(which) {
  recallSetup.classList.toggle("hidden",   which !== "setup");
  recallReading.classList.toggle("hidden", which !== "reading");
  recallInput.classList.toggle("hidden",   which !== "input");
  recallResults.classList.toggle("hidden", which !== "results");
}


// ── Open / close ──────────────────────────────────────────────

function openRecall() {
  _stopRecallTimer();
  recallSessionId = null;

  // Reset textarea
  recallTextarea.value = "";

  sectionModes.classList.add("hidden");
  sectionRecall.classList.remove("hidden");
  showRecallScreen("setup");
  sectionRecall.scrollIntoView({ behavior: "smooth", block: "start" });
}

function closeRecall() {
  _stopRecallTimer();
  sectionRecall.classList.add("hidden");
  sectionModes.classList.remove("hidden");
  sectionModes.scrollIntoView({ behavior: "smooth", block: "start" });
}


// ── Start recall — fetch passage from backend ──────────────────

btnStartRecall.addEventListener("click", async () => {
  const difficulty = recallDifficultyEl.value;

  btnStartRecall.disabled = true;
  showStatus(recallSetupStatus, "Generating passage from your document…", "info");

  try {
    const res  = await fetch(`${BACKEND_URL}/recall/start`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ difficulty }),
    });

    const data = await res.json();

    if (res.status === 429) {
      showStatus(recallSetupStatus, `⏳ ${data.detail}`, "warning");
      btnStartRecall.disabled = false;
      return;
    }

    if (!res.ok) {
      showStatus(recallSetupStatus, `Error: ${data.detail || res.statusText}`, "error");
      btnStartRecall.disabled = false;
      return;
    }

    // Store session id (passage stays on the server after this point)
    recallSessionId = data.session_id;

    // Show passage in reading screen
    recallPassageBox.textContent = data.passage;
    hideStatus(recallSetupStatus);
    btnStartRecall.disabled = false;

    showRecallScreen("reading");
    _startRecallTimer(data.duration_seconds || 30);

  } catch (err) {
    showStatus(recallSetupStatus, `Network error: ${err.message}`, "error");
    console.error("Recall start error:", err);
    btnStartRecall.disabled = false;
  }
});


// ── Countdown timer ───────────────────────────────────────────

function _startRecallTimer(totalSeconds) {
  _stopRecallTimer();   // clear any previous timer

  recallSecondsLeft = totalSeconds;
  recallTimerDisplay.textContent = totalSeconds;
  recallTimerNumber.textContent  = totalSeconds;

  // Reset ring to full
  recallTimerRing.style.strokeDashoffset = "0";
  recallTimerRing.classList.remove("urgent");

  recallTimerHandle = setInterval(() => {
    recallSecondsLeft -= 1;

    // Update text counters
    recallTimerNumber.textContent  = recallSecondsLeft;
    recallTimerDisplay.textContent = recallSecondsLeft;

    // Update SVG ring: offset goes from 0 → circumference as time runs out
    const fraction  = recallSecondsLeft / totalSeconds;
    const offset    = RECALL_TIMER_CIRCUMFERENCE * (1 - fraction);
    recallTimerRing.style.strokeDashoffset = offset;

    // Turn ring red in last 10 seconds
    if (recallSecondsLeft <= 10) {
      recallTimerRing.classList.add("urgent");
    }

    if (recallSecondsLeft <= 0) {
      _stopRecallTimer();
      _onTimerExpired();
    }
  }, 1000);
}

function _stopRecallTimer() {
  if (recallTimerHandle !== null) {
    clearInterval(recallTimerHandle);
    recallTimerHandle = null;
  }
}

function _onTimerExpired() {
  // Hide the passage — user can no longer see it
  recallPassageBox.textContent = "";
  recallTextarea.value = "";
  showRecallScreen("input");
  recallTextarea.focus();
}


// ── Submit recall ─────────────────────────────────────────────

btnSubmitRecall.addEventListener("click", async () => {
  const response = recallTextarea.value.trim();

  if (!response) {
    showStatus(recallEvalStatus, "Please write something before submitting.", "warning");
    return;
  }

  if (!recallSessionId) {
    showStatus(recallEvalStatus, "Session expired. Please start a new recall.", "error");
    return;
  }

  btnSubmitRecall.disabled = true;
  showStatus(recallEvalStatus, "Evaluating your recall…", "info");

  try {
    // NOTE: the passage is NOT sent here — only the session_id and the user's response.
    const res  = await fetch(`${BACKEND_URL}/recall/evaluate`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        session_id: recallSessionId,
        response:   response,
      }),
    });

    const data = await res.json();

    if (res.status === 429) {
      showStatus(recallEvalStatus, `⏳ ${data.detail}`, "warning");
      btnSubmitRecall.disabled = false;
      return;
    }

    if (!res.ok) {
      showStatus(recallEvalStatus, `Error: ${data.detail || res.statusText}`, "error");
      btnSubmitRecall.disabled = false;
      return;
    }

    hideStatus(recallEvalStatus);
    btnSubmitRecall.disabled = false;

    _showRecallResults(data);

  } catch (err) {
    showStatus(recallEvalStatus, `Network error: ${err.message}`, "error");
    console.error("Recall evaluate error:", err);
    btnSubmitRecall.disabled = false;
  }
});


// ── Render results ────────────────────────────────────────────

function _showRecallResults(data) {
  const score = data.score || 0;

  // Animate score gauge
  // Circumference for r=50: 2π×50 ≈ 314.16
  const SCORE_CIRC = 314.16;
  const offset = SCORE_CIRC * (1 - score / 100);

  // Apply colour class based on score
  recallScoreRing.setAttribute("class", "recall-score-ring");
  recallScoreNumber.setAttribute("class", "recall-score-number");
  if (score >= 70) {
    recallScoreRing.classList.add("score-high");
    recallScoreNumber.classList.add("score-high");
  } else if (score >= 40) {
    recallScoreRing.classList.add("score-mid");
    recallScoreNumber.classList.add("score-mid");
  } else {
    recallScoreRing.classList.add("score-low");
    recallScoreNumber.classList.add("score-low");
  }

  // Use rAF so the transition triggers after element becomes visible
  showRecallScreen("results");
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      recallScoreRing.style.strokeDashoffset = offset;
    });
  });

  recallScoreNumber.textContent = score;
  recallSummary.textContent     = data.summary || "";

  // Remembered points
  recallRememberedList.innerHTML = "";
  const remembered = data.remembered_points || [];
  if (remembered.length > 0) {
    remembered.forEach((pt) => {
      const li = document.createElement("li");
      li.textContent = pt;
      recallRememberedList.appendChild(li);
    });
  } else {
    const li = document.createElement("li");
    li.textContent = "No specific points identified.";
    recallRememberedList.appendChild(li);
  }

  // Missed points
  recallMissedList.innerHTML = "";
  const missed = data.missed_points || [];
  if (missed.length > 0) {
    missed.forEach((pt) => {
      const li = document.createElement("li");
      li.textContent = pt;
      recallMissedList.appendChild(li);
    });
  } else {
    const li = document.createElement("li");
    li.textContent = "Nothing major missed — great job!";
    recallMissedList.appendChild(li);
  }

  // Feedback
  if (data.feedback) {
    recallFeedbackBox.textContent = data.feedback;
    recallFeedbackBox.classList.remove("hidden");
  } else {
    recallFeedbackBox.classList.add("hidden");
  }
}


// ── Try Again ─────────────────────────────────────────────────

btnRecallTryAgain.addEventListener("click", () => {
  _stopRecallTimer();
  recallSessionId  = null;
  recallTextarea.value = "";
  showRecallScreen("setup");
});


// ============================================================
// SENTENCE COMPLETION
//
// Flow:
//   1. User clicks "Start →" on the Sentence Completion card → openCompletion()
//   2. User selects difficulty → "Generate Question"
//      → POST /completion/start { "difficulty": "medium" }
//      → response: { session_id, sentence, context_hint, difficulty }
//      (expected answer stored server-side — never sent to frontend)
//   3. Question screen: sentence with styled blank + context hint + answer input
//   4. User types answer → Submit
//      → POST /completion/evaluate { session_id, answer }
//      → response: { score, status, expected_concept, explanation, feedback }
//   5. Result screen: badge, score, expected concept, explanation, feedback
//   6. "Next Question" → generate a new question (new POST /completion/start)
//   7. "See Final Score" → final screen with session totals + gauge
//   8. "Restart" → back to setup, session stats reset
//   9. "← Back" → hides section, shows study modes
//
// Session stats tracked in JS:
//   scStats: { attempted, correct, partial, wrong, totalScore }
// All evaluation is done server-side — frontend never judges correctness.
// ============================================================

// ── DOM refs ─────────────────────────────────────────────────

const sectionCompletion  = document.getElementById("section-completion");
const scSetup            = document.getElementById("sc-setup");
const scDifficultyEl     = document.getElementById("sc-difficulty");
const btnScStart         = document.getElementById("btn-sc-start");
const scSetupStatus      = document.getElementById("sc-setup-status");

const scQuestion         = document.getElementById("sc-question");
const scSessionLabel     = document.getElementById("sc-session-label");
const scSessionScore     = document.getElementById("sc-session-score");
const scSentenceBox      = document.getElementById("sc-sentence-box");
const scHintBox          = document.getElementById("sc-hint-box");
const scAnswerInput      = document.getElementById("sc-answer-input");
const btnScSubmit        = document.getElementById("btn-sc-submit");
const scSubmitStatus     = document.getElementById("sc-submit-status");

const scResult           = document.getElementById("sc-result");
const scStatusBadge      = document.getElementById("sc-status-badge");
const scResultScore      = document.getElementById("sc-result-score");
const scExpectedConcept  = document.getElementById("sc-expected-concept");
const scExplanation      = document.getElementById("sc-explanation");
const scFeedbackBox      = document.getElementById("sc-feedback-box");
const btnScNext          = document.getElementById("btn-sc-next");
const btnScFinish        = document.getElementById("btn-sc-finish");

const scFinal            = document.getElementById("sc-final");
const scFinalTotal       = document.getElementById("sc-final-total");
const scFinalCorrect     = document.getElementById("sc-final-correct");
const scFinalPartial     = document.getElementById("sc-final-partial");
const scFinalWrong       = document.getElementById("sc-final-wrong");
const scFinalAvg         = document.getElementById("sc-final-avg");
const scFinalRing        = document.getElementById("sc-final-ring");
const btnScRestart       = document.getElementById("btn-sc-restart");


// ── Runtime state ─────────────────────────────────────────────

let scSessionId = null;   // current question's session_id

// Cumulative session stats (reset on Restart)
const scStats = {
  attempted:  0,
  correct:    0,
  partial:    0,
  wrong:      0,
  totalScore: 0,
};

function _resetScStats() {
  scStats.attempted  = 0;
  scStats.correct    = 0;
  scStats.partial    = 0;
  scStats.wrong      = 0;
  scStats.totalScore = 0;
}


// ── Screen switcher ───────────────────────────────────────────
// which: "setup" | "question" | "result" | "final"

function showScScreen(which) {
  scSetup.classList.toggle("hidden",    which !== "setup");
  scQuestion.classList.toggle("hidden", which !== "question");
  scResult.classList.toggle("hidden",   which !== "result");
  scFinal.classList.toggle("hidden",    which !== "final");
}


// ── Open / close ──────────────────────────────────────────────

function openCompletion() {
  _resetScStats();
  scSessionId = null;
  scAnswerInput.value = "";

  sectionModes.classList.add("hidden");
  sectionCompletion.classList.remove("hidden");
  showScScreen("setup");
  sectionCompletion.scrollIntoView({ behavior: "smooth", block: "start" });
}

function closeCompletion() {
  sectionCompletion.classList.add("hidden");
  sectionModes.classList.remove("hidden");
  sectionModes.scrollIntoView({ behavior: "smooth", block: "start" });
}


// ── Update session score bar ──────────────────────────────────

function _updateScSessionBar() {
  const n = scStats.attempted + 1;
  scSessionLabel.textContent = `Question ${n}`;
  const avg = scStats.attempted > 0
    ? Math.round(scStats.totalScore / scStats.attempted)
    : 0;
  scSessionScore.textContent = `Avg score: ${avg}`;
}


// ── Render sentence with styled blank ─────────────────────────

function _renderSentence(sentence) {
  // Replace "______" with a styled <span class="sc-blank">
  const safeText = sentence.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const withBlank = safeText.replace(/______/g, '<span class="sc-blank">______</span>');
  scSentenceBox.innerHTML = withBlank;
}


// ── Fetch a new question ──────────────────────────────────────

async function _fetchNextQuestion() {
  const difficulty = scDifficultyEl.value;

  btnScStart.disabled  = true;
  btnScNext.disabled   = true;
  showStatus(scSetupStatus, "Generating question from your document…", "info");

  try {
    const res  = await fetch(`${BACKEND_URL}/completion/start`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ difficulty }),
    });

    const data = await res.json();

    if (res.status === 429) {
      showStatus(scSetupStatus, `⏳ ${data.detail}`, "warning");
      btnScStart.disabled = false;
      btnScNext.disabled  = false;
      return;
    }

    if (!res.ok) {
      showStatus(scSetupStatus, `Error: ${data.detail || res.statusText}`, "error");
      btnScStart.disabled = false;
      btnScNext.disabled  = false;
      return;
    }

    scSessionId = data.session_id;

    // Render question screen
    hideStatus(scSetupStatus);
    _updateScSessionBar();
    _renderSentence(data.sentence);

    if (data.context_hint) {
      scHintBox.textContent = `💡 Hint: ${data.context_hint}`;
      scHintBox.classList.remove("hidden");
    } else {
      scHintBox.classList.add("hidden");
    }

    // Reset answer input
    scAnswerInput.value = "";
    hideStatus(scSubmitStatus);
    btnScSubmit.disabled = false;

    btnScStart.disabled = false;
    btnScNext.disabled  = false;

    showScScreen("question");
    scAnswerInput.focus();

  } catch (err) {
    showStatus(scSetupStatus, `Network error: ${err.message}`, "error");
    console.error("Completion start error:", err);
    btnScStart.disabled = false;
    btnScNext.disabled  = false;
  }
}


// ── Generate Question button ──────────────────────────────────

btnScStart.addEventListener("click", () => {
  _resetScStats();
  _fetchNextQuestion();
});


// ── Submit answer ─────────────────────────────────────────────

async function _submitAnswer() {
  const answer = scAnswerInput.value.trim();

  if (!answer) {
    showStatus(scSubmitStatus, "Please type an answer before submitting.", "warning");
    return;
  }

  if (!scSessionId) {
    showStatus(scSubmitStatus, "Session expired. Please generate a new question.", "error");
    return;
  }

  btnScSubmit.disabled = true;
  showStatus(scSubmitStatus, "Evaluating your answer…", "info");

  try {
    // The expected concept is NOT sent — only session_id + answer
    const res  = await fetch(`${BACKEND_URL}/completion/evaluate`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        session_id: scSessionId,
        answer:     answer,
      }),
    });

    const data = await res.json();

    if (res.status === 429) {
      showStatus(scSubmitStatus, `⏳ ${data.detail}`, "warning");
      btnScSubmit.disabled = false;
      return;
    }

    if (!res.ok) {
      showStatus(scSubmitStatus, `Error: ${data.detail || res.statusText}`, "error");
      btnScSubmit.disabled = false;
      return;
    }

    // Update session stats
    scStats.attempted  += 1;
    scStats.totalScore += (data.score || 0);
    if (data.status === "correct")           scStats.correct += 1;
    else if (data.status === "partially_correct") scStats.partial += 1;
    else                                     scStats.wrong   += 1;

    hideStatus(scSubmitStatus);
    btnScSubmit.disabled = false;
    _showScResult(data);

  } catch (err) {
    showStatus(scSubmitStatus, `Network error: ${err.message}`, "error");
    console.error("Completion evaluate error:", err);
    btnScSubmit.disabled = false;
  }
}

btnScSubmit.addEventListener("click", _submitAnswer);

scAnswerInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") _submitAnswer();
});


// ── Show per-question result ───────────────────────────────────

function _showScResult(data) {
  const status = data.status || "incorrect";
  const score  = data.score  || 0;

  // Status badge
  scStatusBadge.className = "sc-status-badge";
  if (status === "correct") {
    scStatusBadge.classList.add("badge-correct");
    scStatusBadge.textContent = "✓ Correct";
  } else if (status === "partially_correct") {
    scStatusBadge.classList.add("badge-partial");
    scStatusBadge.textContent = "◑ Partially Correct";
  } else {
    scStatusBadge.classList.add("badge-incorrect");
    scStatusBadge.textContent = "✗ Incorrect";
  }

  scResultScore.textContent    = `${score} / 100`;
  scExpectedConcept.textContent = data.expected_concept || "—";
  scExplanation.textContent    = data.explanation || "—";

  if (data.feedback) {
    scFeedbackBox.textContent = data.feedback;
    scFeedbackBox.classList.remove("hidden");
  } else {
    scFeedbackBox.classList.add("hidden");
  }

  showScScreen("result");
}


// ── Next question ─────────────────────────────────────────────

btnScNext.addEventListener("click", () => {
  showScScreen("setup");   // show setup briefly while loading
  _fetchNextQuestion();
});


// ── See final score ───────────────────────────────────────────

btnScFinish.addEventListener("click", () => {
  _showScFinal();
});

function _showScFinal() {
  const { attempted, correct, partial, wrong, totalScore } = scStats;
  const avg = attempted > 0 ? Math.round(totalScore / attempted) : 0;

  scFinalTotal.textContent   = attempted;
  scFinalCorrect.textContent = correct;
  scFinalPartial.textContent = partial;
  scFinalWrong.textContent   = wrong;
  scFinalAvg.textContent     = avg;

  // Colour avg number
  scFinalAvg.className = "recall-score-number";
  if (avg >= 70)       scFinalAvg.classList.add("score-high");
  else if (avg >= 40)  scFinalAvg.classList.add("score-mid");
  else                 scFinalAvg.classList.add("score-low");

  // Colour gauge ring
  const SCORE_CIRC = 314.16;
  const offset = SCORE_CIRC * (1 - avg / 100);
  scFinalRing.className = "recall-score-ring";
  if (avg >= 70)       scFinalRing.classList.add("score-high");
  else if (avg >= 40)  scFinalRing.classList.add("score-mid");
  else                 scFinalRing.classList.add("score-low");

  showScScreen("final");

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      scFinalRing.style.strokeDashoffset = offset;
    });
  });
}


// ── Restart ───────────────────────────────────────────────────

btnScRestart.addEventListener("click", () => {
  _resetScStats();
  scSessionId = null;
  scAnswerInput.value = "";
  showScScreen("setup");
});
