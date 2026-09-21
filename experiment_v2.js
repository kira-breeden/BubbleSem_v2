// ============================================================
// BubbleSem v2 — Three-Part Experiment
//
// Part 1 (Baseline): some_masked + most_masked trials shuffled
//   - 10 target words shown with 20% or 40% of revealable words revealed
//   - No reveal interactivity; participant makes guess when ready
//
// Part 2 (Sampling): predetermined greedy-trajectory reveal
//   - The other 10 target words start fully masked
//   - "Reveal Next Word" reveals words one at a time in greedy-trajectory
//     order (most informative first) — the participant cannot choose which
//     word is revealed next, only when to stop and guess
//   - Points system: each trial starts at 100 points; each reveal costs
//     100 / (trajectory length for that passage), so fully revealing a
//     passage always costs 100 points regardless of passage length
//
// Part 3 (Open-Ended): longer passages loaded from open_ended_passages.csv
//   - Participant reads each passage and answers an open-ended question
//     about what the passage is about
//   - Response and timing recorded; no target-word guessing
//
// Every participant sees all 20 target words exactly once (10 in Part 1,
// 10 in Part 2), rotating through entropy levels so contexts vary in
// ambiguity across the study.
//
// CSV files required:
//   trial_lists/sublist_X.csv  — 20 Part 1 + Part 2 trials (varies by sublist)
//     columns: condition, target_word, real_passage, jabber_passage,
//              target_word_position, unmasked_word_indices (Part 1),
//              reveal_order (Part 2), entropy, target_probability, ...
//   open_ended_passages.csv  — Part 3 passages (same for all participants)
//     columns: passage_id, longer_passage
//
// URL parameters:
//   sublist=1..16  (default: 1)
//   subjCode=<string> (default: random ID)
// ============================================================

// ===== GLOBAL STATE =====

let baselineTrialData  = [];
let samplingTrialData  = [];
let openEndedTrialData = [];
let trialSequenceData = {};   // accumulates data across a trial's screens
let consolidatedTrials = [];  // all saved trial rows
let startTime = null;
let firstKeystrokeTime = null;  // time from passage appearing to first keypress in guess/response box

// Sampling-specific state (reset each sampling trial)
let revealQueue          = [];  // {wordPos, tokenIdx} entries remaining to reveal
let revealedTokenIndices = [];  // token indices revealed so far
let revealClickTimes     = [];  // [{word_position, revealed_word, time_from_start, num_revealed}]
let trialPoints          = 100;
let pointsPerReveal      = 0;

// Words that are always shown as real (never masked)
const ARTICLES = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                  'to', 'for', 'of', 'with', 'by'];

// ===== URL PARAMETERS + SUBJECT CODE =====

function getURLParameter(name) {
    return new URLSearchParams(window.location.search).get(name);
}

const subjCode = getURLParameter('subjCode') || jsPsych.randomization.randomID(10);

let sublistNumber = null;  // set in assignSublist() at timeline start

const seedParam  = getURLParameter('seed');
const randomSeed = seedParam ? parseInt(seedParam) : Math.floor(Math.random() * 1000000);
const filename   = `${subjCode}.csv`;

// ===== JSPSYCH INIT =====

const jsPsych = initJsPsych({});

// ===== SEEDED RNG =====

class SeededRandom {
    constructor(seed) { this.seed = seed; }

    next() {
        this.seed = (this.seed * 9301 + 49297) % 233280;
        return this.seed / 233280;
    }

    shuffle(array) {
        const shuffled = [...array];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(this.next() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }
}

// ===== UTILITY FUNCTIONS =====

// Split a sentence string into word and punctuation tokens.
// e.g. "Hello, world." => ["Hello", ",", "world", "."]
function tokenizeSentence(sentence) {
    const tokens = [];
    sentence.split(' ').forEach(word => {
        const match = word.match(/^([^.,!?;:'"]*)([.,!?;:'"]*)$/);
        if (match) {
            const [, wordPart, punctPart] = match;
            if (wordPart) tokens.push(wordPart);
            if (punctPart) punctPart.split('').forEach(p => tokens.push(p));
        } else {
            tokens.push(word);
        }
    });
    return tokens;
}

// Return true if a token is punctuation.
function isPunct(token) {
    return /^[.,!?;:'"]$/.test(token);
}

// Convert a 0-indexed word position (skipping punctuation tokens) to a token index.
function wordPosToTokenIndex(tokens, wordPos) {
    let wordCount = 0;
    for (let i = 0; i < tokens.length; i++) {
        if (!isPunct(tokens[i])) {
            if (wordCount === wordPos) return i;
            wordCount++;
        }
    }
    console.error(`wordPosToTokenIndex: position ${wordPos} not found (${wordCount} words in tokens)`);
    return -1;
}

// Build a Map from token index → word position (0-indexed, ignoring punctuation).
// Used to convert internal token indices back to word positions when saving data.
function buildTokenToWordPosMap(tokens) {
    const map = new Map();
    let wordPos = 0;
    for (let i = 0; i < tokens.length; i++) {
        if (!isPunct(tokens[i])) {
            map.set(i, wordPos);
            wordPos++;
        }
    }
    return map;
}

// Parse a JSON array column from CSV (PapaParse may leave it as a string).
function parseJSONColumn(value) {
    if (Array.isArray(value)) return value;
    if (typeof value === 'string') {
        try { return JSON.parse(value); }
        catch (e) {
            console.error('parseJSONColumn: failed to parse', value);
            return [];
        }
    }
    return [];
}

// Return true if this token should always be shown as its real English word
// (articles, function words, or words identical in jabberwocky and real versions).
function isAutoRevealed(jabberToken, realToken) {
    const cleanJ = jabberToken.toLowerCase().replace(/[.,!?;:'"]/g, '');
    const cleanR = realToken.toLowerCase().replace(/[.,!?;:'"]/g, '');
    return ARTICLES.includes(cleanJ) || ARTICLES.includes(cleanR) || cleanJ === cleanR;
}

// Return token indices of all maskable words in a passage:
// words that are nonce in jabber but real in the original — i.e. the two tokens differ.
// Excludes the target token index and punctuation.
function getMaskableTokenIndices(jabberTokens, realTokens, targetTokenIdx) {
    const maskable = [];
    for (let i = 0; i < jabberTokens.length; i++) {
        if (i === targetTokenIdx) continue;
        if (isPunct(jabberTokens[i])) continue;
        if (!isAutoRevealed(jabberTokens[i], realTokens[i])) {
            maskable.push(i);
        }
    }
    return maskable;
}

// Update the on-screen points counter (sampling trials only).
function updatePointsDisplay(points) {
    const el = document.getElementById('points-counter');
    if (el) {
        el.textContent = `Points: ${Math.round(points)}`;
        el.style.color = '#d32f2f';
        setTimeout(() => { el.style.color = '#333'; }, 300);
    }
}

// Convert an array of objects to a CSV string.
// Headers are the union of all keys across every row so that Phase 1 and
// Phase 2 columns all appear even though each phase has unique fields.
function arrayToCSV(data) {
    if (!data.length) return '';
    const headers = [...new Set(data.flatMap(row => Object.keys(row)))];
    const escape = val => {
        if (val === null || val === undefined) return '';
        const str = String(val);
        return (str.includes(',') || str.includes('"') || str.includes('\n'))
            ? `"${str.replace(/"/g, '""')}"` : str;
    };
    return [
        headers.join(','),
        ...data.map(row => headers.map(h => escape(row[h])).join(','))
    ].join('\n');
}

// ===== CONDITION ASSIGNMENT =====

const EXPERIMENT_ID = 'PYSjeESL3lfq';
const N_SUBLISTS    = 16;

// Returns a sublist number (1–16).
// If ?sublist= is in the URL, uses that.
// Otherwise calls the DataPipe condition assignment API for counterbalanced assignment.
async function assignSublist() {
    const param  = getURLParameter('sublist');
    const parsed = parseInt(param);
    if (param && parsed >= 1 && parsed <= N_SUBLISTS) {
        console.log(`Sublist from URL: ${parsed}`);
        return parsed;
    }

    try {
        const resp = await fetch('https://pipe.jspsych.org/api/condition/', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ experimentID: EXPERIMENT_ID })
        });
        console.log('DataPipe response status:', resp.status);
        if (!resp.ok) {
            console.warn(`DataPipe returned HTTP ${resp.status} — defaulting to sublist 1.`);
            return 1;
        }
        const data = await resp.json();
        console.log('DataPipe response body:', data);
        const conditionNum = parseInt(data.condition);
        if (isNaN(conditionNum)) {
            console.warn('DataPipe returned unexpected condition value:', data, '— defaulting to sublist 1.');
            return 1;
        }
        // DataPipe returns 0-indexed condition; map to 1–N_SUBLISTS
        const assigned = (conditionNum % N_SUBLISTS) + 1;
        console.log(`Sublist from DataPipe condition assignment: ${assigned}`);
        return assigned;
    } catch (err) {
        console.warn('DataPipe condition assignment failed, defaulting to sublist 1.', err);
        return 1;
    }
}

// ===== DATA LOADING =====

function loadCSV(csvFilename) {
    return new Promise((resolve, reject) => {
        Papa.parse(csvFilename, {
            download: true,
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true,
            complete: results => {
                if (!results.data.length) {
                    reject(new Error(`CSV file is empty: ${csvFilename}`));
                } else {
                    console.log(`Loaded ${results.data.length} rows from ${csvFilename}`);
                    console.log('Sample row:', results.data[0]);
                    resolve(results.data);
                }
            },
            error: err => reject(err)
        });
    });
}

async function loadAllTrialData() {
    const allTrials = await loadCSV(`trial_lists/sublist_${sublistNumber}.csv`);

    const baseline  = allTrials.filter(t => t.condition === 'some_masked' || t.condition === 'most_masked');
    const sampling  = allTrials.filter(t => t.condition === 'sampling');
    const openEnded = allTrials.filter(t => t.condition === 'open_ended');

    // Shuffle each part independently with seeded RNG
    const rng = new SeededRandom(randomSeed);
    baselineTrialData  = rng.shuffle(baseline);
    samplingTrialData  = rng.shuffle(sampling);
    openEndedTrialData = rng.shuffle(openEnded);

    console.log(`Baseline trials:    ${baselineTrialData.length}`);
    console.log(`Sampling trials:    ${samplingTrialData.length}`);
    console.log(`Open-ended trials:  ${openEndedTrialData.length}`);
}

// ===== HARDCODED PRACTICE + ATTENTION CHECK TRIALS =====

const PRACTICE_TRIAL_DATA = [
    {
        passageHtml: `The zirps kicked the <span class="word target">blorf</span> across the scrempf. It glashed high prof the deek.`,
        targetWord: 'ball',
    },
    {
        passageHtml: `She zop dake at the glimp and opened her <span class="word target">glorp</span>. She began to zap quietly.`,
        targetWord: 'book',
    },
];

const ATTENTION_CHECK_DATA = [
    {
        passageHtml: `The dog barked loudly at the <span class="word target">blorf</span> across the street. Everyone on the block could hear it.`,
        targetWord: 'cat',
    },
    {
        passageHtml: `She turned on the kitchen <span class="word target">glorp</span> to fill the pot with water. The sound of the cool tap water filling the pot echoed aorund the kitchen.`,
        targetWord: 'sink',
    },
];

function createHardcodedTrial(passageHtml, targetWord, trialType, trialNumber) {
    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: function () {
            startTime = Date.now();
            firstKeystrokeTime = null;
            trialSequenceData = {
                subjCode,
                sublist:     sublistNumber,
                random_seed: randomSeed,
                trial_type:  trialType,
                trial_number: trialNumber,
                target_word:  targetWord,
                condition:    trialType,
            };
            return `
                <div class="sentence-container baseline-passage" id="sentence-container">
                    ${passageHtml}
                </div>
                <div class="controls">
                    <button class="guess-button" id="guess-btn">Make Guess</button>
                </div>
            `;
        },
        choices: ['Make Guess'],
        button_html: '<button class="jspsych-btn" style="display:none;">%choice%</button>',
        on_load: function () {
            document.getElementById('guess-btn').addEventListener('click', function () {
                trialSequenceData.time_before_guess = Date.now() - startTime;
                jsPsych.finishTrial();
            });
        },
        trial_duration: null,
        response_ends_trial: false
    };
}

// ===== BASELINE TRIAL =====
// Shows the passage with predetermined masked/unmasked words.
// No reveal interactivity — participant clicks "Make Guess" when ready.

function createBaselineTrial(trial, sectionTrialIndex, totalBaseline, trialNumber) {
    const realSentence   = trial.real_passage    || '';
    const jabberSentence = trial.jabber_passage  || '';
    const realTokens     = tokenizeSentence(realSentence);
    const jabberTokens   = tokenizeSentence(jabberSentence);
    const targetTokenIdx = wordPosToTokenIndex(jabberTokens, trial.target_word_position);
    const maskingLevel   = trial.condition || 'some_masked';

    // Unmasked word positions come directly from the pre-computed trial list.
    const unmaskedWordPositions = parseJSONColumn(trial.unmasked_word_indices);
    const unmaskedTokenIdxSet = new Set(
        unmaskedWordPositions
            .map(pos => wordPosToTokenIndex(jabberTokens, pos))
            .filter(i => i >= 0)
    );

    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: function () {
            startTime = Date.now();

            trialSequenceData = {
                subjCode:               subjCode,
                sublist:                sublistNumber,
                random_seed:            randomSeed,
                trial_type:             'baseline',
                trial_number:           trialNumber,
                trial_list_index:       trial.trial_number,
                condition:              maskingLevel,
                section_trial_index:    sectionTrialIndex + 1,
                target_word:            trial.target_word,
                target_word_position:   trial.target_word_position,
                entropy:                trial.entropy,
                target_probability:     trial.target_probability,
                real_passage:           realSentence,
                jabber_passage:         jabberSentence,
                unmasked_word_indices:  JSON.stringify(unmaskedWordPositions),
            };

            let html = `
                <div class="sentence-container baseline-passage" id="sentence-container">
            `;

            for (let i = 0; i < jabberTokens.length; i++) {
                const token = jabberTokens[i];

                if (isPunct(token)) {
                    html += token;
                    if (/[.,!?;:]/.test(token) && i < jabberTokens.length - 1) html += ' ';
                    continue;
                }

                if (i === targetTokenIdx) {
                    // Target word: always show as jabberwocky, bold
                    html += `<span class="word target">${token}</span> `;
                } else if (
                    unmaskedTokenIdxSet.has(i) ||
                    isAutoRevealed(jabberTokens[i], realTokens[i])
                ) {
                    // Unmasked: show real English word
                    html += `<span class="word">${realTokens[i]}</span> `;
                } else {
                    // Masked: show jabberwocky word (styled as nonce)
                    html += `<span class="word nonce">${token}</span> `;
                }
            }

            html += `
                </div>
                <div class="controls">
                    <button class="guess-button" id="guess-btn">Make Guess</button>
                </div>
            `;

            return html;
        },
        choices: ['Make Guess'],
        button_html: '<button class="jspsych-btn" style="display:none;">%choice%</button>',
        on_load: function () {
            document.getElementById('guess-btn').addEventListener('click', function () {
                trialSequenceData.time_before_guess = Date.now() - startTime;
                jsPsych.finishTrial();
            });
        },
        trial_duration: null,
        response_ends_trial: false
    };
}

// ===== SAMPLING TRIAL (Part 2) =====
// All revealable words start masked. "Reveal Next Word" reveals them one at a
// time in greedy-trajectory order (most informative first) — the participant
// cannot choose which word is revealed, only when to stop and guess.

function createSamplingTrial(trial, sectionTrialIndex, totalSampling, trialNumber) {
    const realSentence   = trial.real_passage    || '';
    const jabberSentence = trial.jabber_passage  || '';
    const realTokens     = tokenizeSentence(realSentence);
    const jabberTokens   = tokenizeSentence(jabberSentence);
    const targetTokenIdx = wordPosToTokenIndex(jabberTokens, trial.target_word_position);

    // Reveal order comes from the passage's greedy trajectory (word positions,
    // most informative first). Convert to token indices for DOM manipulation.
    const revealOrderWordPositions = parseJSONColumn(trial.reveal_order);
    const revealOrderPairs = revealOrderWordPositions
        .map(wordPos => ({ wordPos, tokenIdx: wordPosToTokenIndex(jabberTokens, wordPos) }))
        .filter(({ tokenIdx }) => tokenIdx >= 0 && tokenIdx !== targetTokenIdx);

    // Token-index → word-position reverse map (for saving word positions in output data)
    const tokenToWordPos = buildTokenToWordPosMap(jabberTokens);

    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: function () {
            startTime            = Date.now();
            trialPoints          = 100;
            revealQueue          = [...revealOrderPairs]; // each entry: {wordPos, tokenIdx}
            revealedTokenIndices = [];
            revealClickTimes     = [];
            pointsPerReveal      = revealQueue.length > 0
                ? Math.round((100 / revealQueue.length) * 100) / 100
                : 0;

            trialSequenceData = {
                subjCode:             subjCode,
                sublist:              sublistNumber,
                random_seed:          randomSeed,
                trial_type:           'sampling',
                trial_number:         trialNumber,
                trial_list_index:     trial.trial_number,
                condition:            'sampling',
                section_trial_index:  sectionTrialIndex + 1,
                target_word:          trial.target_word,
                target_word_position: trial.target_word_position,
                entropy:              trial.entropy,
                target_probability:   trial.target_probability,
                real_passage:         realSentence,
                jabber_passage:       jabberSentence,
                reveal_order:         JSON.stringify(revealOrderWordPositions),
                points_per_reveal:    pointsPerReveal,
            };

            let html = `
                <div style="position: relative;">
                    <div class="trial-counter">
                        Part 2 &mdash; Trial ${sectionTrialIndex + 1} of ${totalSampling}
                    </div>
                    <div class="points-counter" id="points-counter">Points: ${trialPoints}</div>
                    <div class="sentence-container sampling-passage" id="sentence-container">
            `;

            for (let i = 0; i < jabberTokens.length; i++) {
                const token = jabberTokens[i];

                if (isPunct(token)) {
                    html += token;
                    if (/[.,!?;:]/.test(token) && i < jabberTokens.length - 1) html += ' ';
                    continue;
                }

                if (i === targetTokenIdx) {
                    html += `<span class="word target">${token}</span> `;
                } else if (isAutoRevealed(jabberTokens[i], realTokens[i])) {
                    html += `<span class="word">${realTokens[i]}</span> `;
                } else {
                    // All other words start masked; id used for DOM update on reveal
                    html += `<span class="word nonce" id="word-tok-${i}"
                                   data-real="${realTokens[i]}">${token}</span> `;
                }
            }

            html += `
                    </div>
                    <div class="controls">
                        <button class="reveal-button" id="reveal-btn">Reveal Next Word</button>
                        <button class="guess-button"  id="guess-btn">Make Guess</button>
                    </div>
                </div>
            `;

            return html;
        },
        choices: ['Make Guess'],
        button_html: '<button class="jspsych-btn" style="display:none;">%choice%</button>',
        on_load: function () {
            const revealBtn = document.getElementById('reveal-btn');
            const guessBtn  = document.getElementById('guess-btn');

            if (revealQueue.length === 0) revealBtn.disabled = true;

            revealBtn.addEventListener('click', function () {
                if (revealQueue.length === 0) return;

                const { wordPos, tokenIdx } = revealQueue.shift();
                revealedTokenIndices.push(tokenIdx);

                // Deduct points
                trialPoints = Math.max(0, trialPoints - pointsPerReveal);
                updatePointsDisplay(trialPoints);

                revealClickTimes.push({
                    word_position:   wordPos,
                    revealed_word:   realTokens[tokenIdx],
                    time_from_start: Date.now() - startTime,
                    num_revealed:    revealedTokenIndices.length
                });

                // Update word in DOM
                const wordEl = document.getElementById(`word-tok-${tokenIdx}`);
                if (wordEl) {
                    wordEl.textContent = realTokens[tokenIdx];
                    wordEl.classList.remove('nonce');
                    wordEl.classList.add('revealed');
                }

                if (revealQueue.length === 0) revealBtn.disabled = true;
            });

            guessBtn.addEventListener('click', function () {
                // Save word positions (0-indexed, ignoring punctuation) — not token indices
                const revealedWordPositions = revealedTokenIndices.map(
                    ti => tokenToWordPos.get(ti)
                );
                trialSequenceData.num_words_revealed    = revealedWordPositions.length;
                trialSequenceData.revealed_word_indices = JSON.stringify(revealedWordPositions);
                trialSequenceData.revealed_words        =
                    JSON.stringify(revealedTokenIndices.map(ti => realTokens[ti]));
                trialSequenceData.click_times           = JSON.stringify(revealClickTimes);
                trialSequenceData.time_before_guess     = Date.now() - startTime;
                trialSequenceData.points_remaining      = Math.round(trialPoints * 100) / 100;
                jsPsych.finishTrial();
            });
        },
        trial_duration: null,
        response_ends_trial: false
    };
}

// ===== OPEN-ENDED TRIAL (Part 3) =====
// Shows the full real passage and collects an open-ended response
// about what the participant thinks the passage is about.
// Response and timing are saved directly here (no separate guess/confidence screens).

function createOpenEndedTrial(trial, sectionTrialIndex, totalOpenEnded, trialNumber) {
    const passage = trial.jabber_passage || '';

    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: function () {
            startTime = Date.now();

            trialSequenceData = {
                subjCode:            subjCode,
                sublist:             sublistNumber,
                random_seed:         randomSeed,
                trial_type:          'open_ended',
                trial_number:        trialNumber,
                trial_list_index:    trial.trial_number,
                condition:           'open_ended',
                section_trial_index: sectionTrialIndex + 1,
                passage_id:          trial.passage_id,
                real_passage:        trial.real_passage || '',
                jabber_passage:      passage,
            };

            return `
                <div class="sentence-container open-ended-passage" id="sentence-container">
                    ${passage}
                </div>
                <div class="open-ended-question">
                    <label for="open-ended-response">
                        <strong>What do you think this passage is about?</strong>
                        Write a few sentences describing your interpretation.
                    </label>
                    <textarea
                        id="open-ended-response"
                        rows="5"
                        placeholder="Write your response here..."
                    ></textarea>
                </div>
                <div class="controls">
                    <button class="guess-button" id="submit-btn" disabled>Submit</button>
                </div>
            `;
        },
        choices: ['Submit'],
        button_html: '<button class="jspsych-btn" style="display:none;">%choice%</button>',
        on_load: function () {
            firstKeystrokeTime = null;
            const textarea  = document.getElementById('open-ended-response');
            const submitBtn = document.getElementById('submit-btn');

            textarea.addEventListener('keydown', function () {
                if (firstKeystrokeTime === null) {
                    firstKeystrokeTime = Date.now() - startTime;
                }
            }, { once: true });

            textarea.addEventListener('input', function () {
                submitBtn.disabled = textarea.value.trim().length === 0;
            });

            submitBtn.addEventListener('click', function () {
                trialSequenceData.open_ended_response     = textarea.value.trim();
                trialSequenceData.time_before_submit      = Date.now() - startTime;
                trialSequenceData.time_to_first_keystroke = firstKeystrokeTime;
                consolidatedTrials.push({ ...trialSequenceData });
                console.log('Phase 2 trial saved:', trialSequenceData);
                jsPsych.finishTrial();
            });
        },
        trial_duration: null,
        response_ends_trial: false
    };
}

// ===== SHARED TRIAL TYPES =====

// Guess input — same for both sections
function createGuessInputTrial() {
    return {
        type: jsPsychSurveyText,
        questions: [{
            prompt: `
                <div class="instructions">
                    <p>What do you think the <strong>bolded word</strong> was in the sentence?</p>
                    <p><strong>Type ONE WORD for your guess:</strong></p>
                </div>
            `,
            name: 'target_word_guess',
            required: true,
            rows: 1,
            columns: 40
        }],
        on_load: function () {
            firstKeystrokeTime = null;
            const input     = document.querySelector('[data-name="target_word_guess"]');
            const submitBtn = document.querySelector('input[type="submit"].jspsych-btn');

            if (submitBtn) submitBtn.disabled = true;

            if (input) {
                input.addEventListener('keydown', function () {
                    if (firstKeystrokeTime === null) {
                        firstKeystrokeTime = Date.now() - startTime;
                    }
                }, { once: true });

                input.addEventListener('input', function () {
                    if (submitBtn) submitBtn.disabled = input.value.trim().length === 0;
                });
            }
        },
        on_finish: function (data) {
            trialSequenceData.guess                  = data.response.target_word_guess;
            trialSequenceData.rt_guess               = data.rt;
            trialSequenceData.time_to_first_keystroke = firstKeystrokeTime;
        }
    };
}

// Confidence rating — same for both sections.
// Pushes the completed trial object to consolidatedTrials.
function createConfidenceRatingTrial() {
    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: `
            <div style="text-align: center;">
                <p>How confident are you in your guess?</p>
            </div>
        `,
        choices: [
            'Not at all confident',
            'Slightly confident',
            'Moderately confident',
            'Very confident',
            'Extremely confident'
        ],
        on_finish: function (data) {
            trialSequenceData.confidence_rating = data.response + 1; // 0-4 → 1-5
            trialSequenceData.confidence_rt     = data.rt;
            consolidatedTrials.push({ ...trialSequenceData });
            console.log('Trial saved:', trialSequenceData);
        }
    };
}

// Feedback — shows the correct target word. Same for both sections.
function createFeedbackTrial(trial) {
    return {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: function () {
            return `
                <div style="text-align: center; max-width: 600px; margin: 0 auto; padding: 40px;">
                    <h2>The target word was:</h2>
                    <p style="font-size: 36px; font-weight: bold; margin: 30px 0;">
                        ${trial.target_word}
                    </p>
                    <p style="font-size: 14px; color: #666;">
                        <em>Press any key to continue</em>
                    </p>
                </div>
            `;
        },
        trial_duration: null
    };
}

// ===== SCREENS =====

const consent = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
        <div style="width: 800px; margin: 0 auto; text-align: left">
            <h3>Consent to Participate in Research</h3>

            <p>The task you are about to do is sponsored by University of Wisconsin-Madison.
            It is part of a protocol titled "What are we learning from language?"</p>

            <p>The task you are asked to do involves making simple responses to words and
            sentences. More detailed instructions for this specific task will be provided
            on the next screen.</p>

            <p>This task has no direct benefits. We do not anticipate any psychosocial
            risks. There is a risk of a confidentiality breach. Participants may become
            fatigued or frustrated due to the length of the study.</p>

            <p>The responses you submit as part of this task will be stored on a secure
            server and accessible only to researchers who have been approved by
            UW-Madison. Processed data with all identifiers removed could be used for
            future research studies or distributed to another investigator for future
            research studies without additional informed consent.</p>

            <p>You are free to decline to participate, to end participation at any time
            for any reason, or to refuse to answer any individual question without penalty
            or loss of earned compensation. We will not retain data from partial
            responses.</p>

            <p>If you have any questions or concerns about this task please contact the
            principal investigator: Prof. Gary Lupyan at lupyan@wisc.edu.</p>

            <p>If you are not satisfied with the response of the research team, have more
            questions, or want to talk with someone about your rights as a research
            participant, you should contact University of Wisconsin's Education Research
            and Social &amp; Behavioral Science IRB Office at 608-263-2320.</p>

            <p><strong>By clicking the box below, I consent to participate in this task
            and affirm that I am at least 18 years old.</strong></p>
        </div>
    `,
    choices: ['I Agree', 'I Do Not Agree'],
    on_finish: function (data) {
        if (data.response === 1) {
            jsPsych.endExperiment('Thank you for your time. The experiment has been ended.');
        }
    }
};

// Browsers require a user gesture to enter fullscreen — this click is that
// gesture. Fullscreen is not exited automatically; it persists through the
// final redirect to the Qualtrics survey.
const enterFullscreen = {
    type: jsPsychFullscreen,
    fullscreen_mode: true,
    message: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <p>This experiment works best in fullscreen mode.</p>
        </div>
    `,
    button_label: 'Continue in Fullscreen'
};

const welcome = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h1>Welcome!</h1>
            <p>In this experiment you will read passages and answer questions about them.
            The experiment has three parts. You will receive instructions for each part
            before it begins.</p>
            <p><em>Press any key to continue</em></p>
        </div>
    `
};

// --- Part 1 instructions ---

const baselineInstructions1 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Part 1 Instructions</h2>
            <p>In this task you will read a passage of text and try to guess the meaning of a word in bold. The majority of 
            the words in the passage will be nonsense words. These words are totally random and have no relationship to the 
            real English words they have replaced. </p>
            <p>On each trial:</p>
            <ol>
                <li>You'll see a sentence with one <strong>bolded word</strong> - this is your target word to guess</li>
                <li>Read the sentence carefully to understand the context</li>
                <li>When you think you know the meaning of the bolded word, click "Make Guess"</li>
                <li>Type your guess for the bolded word</li>
                <li>Rate your confidence in your guess</li>
                <li>You'll see feedback showing the correct answer</li>
            </ol>
            <p><strong>Important: Try to be as specific as possible in your guesses. Your guess should be ONE WORD!</strong></p>
            <p><em>Press any key to move on to the next page </em></p>
        </div> 
    `
};

const baselineInstructions2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>When are you ready to guess?</h2>
            <p><strong>This will sometimes be quite difficult, but just do your best!</strong></p>
            <p>For example, you might see something like this:</p>
            <p style="margin-left: 20px; font-style: italic;">
                "The glorp tafed in the deng zirp <strong>glosh</strong>."
            </p>
            <p>Take some time to think about what "glosh" might mean and once you have your best ONE WORD GUESS, you can move forward.</p>
            <p style="margin-top: 30px;"><em>Press any key to continue</em></p>
        </div>
    `
};

// --- Practice trial instructions ---

const practiceInstructions = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Let's Practice</h2>
            <p>Before we begin, let's do <strong>2 practice trials</strong> so you can
            get comfortable with the task.</p>
            <p>Remember: the bolded nonsense word is what you're guessing. Some words will be masked 
            and some won't but try your best to use all the context available to you to guess the meaning.</p>
            <p><em>Press any key to start the practice</em></p>
        </div>
    `
};

// --- Post-practice transition ---

const practiceCompleteScreen = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Practice complete!</h2>
            <p><strong>Final Reminder: Please use ONE WORD GUESSES!</strong></p>
            <p><strong>Some will be harder than others, so just do your best and take your time!</strong></p>
            <p><em>Press any key to start Part 1</em></p>
        </div>
    `
};

// --- Transition between Part 1 and Part 2 ---

const transitionScreen = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Great work — Part 1 complete!</h2>
            <p>Now we will move on to <strong>Part 2</strong>, which works differently.</p>
            <p><em>Press any key to read the Part 2 instructions</em></p>
        </div>
    `
};

// --- Part 2 instructions ---

const samplingInstructions1 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Part 2 Instructions</h2>
            <p>In this part, passages start with <strong>all</strong> non-target words
            replaced by nonsense words. You can reveal the real words one at a time by
            clicking <strong>Reveal Next Word</strong>.</p>
            <p>Your job:</p>
            <ol>
                <li>Read the passage (initially all nonsense except articles).</li>
                <li>Click <strong>Reveal Next Word</strong> to reveal another word.
                    Each revealed word will stay visible.</li>
                <li>Click <strong>Make Guess</strong> whenever you feel ready — you do not
                    need to reveal every word first!</li>
                <li>Type your best ONE-WORD guess and rate your confidence.</li>
            </ol>
            <p><em>Press any key to continue</em></p>
        </div>
    `
};

const samplingInstructions2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Part 2 — Scoring</h2>
            <p>Each trial in Part 2 starts with <strong>100 points</strong>. Every word
            you reveal costs you some points.</p>
            <p>Try to guess the target word with as few reveals as possible to keep your
            score high!</p>
            <p>You cannot choose which word is revealed next — only when to stop and
            guess.</p>
            <p><strong>Please use ONE WORD guesses only.</strong></p>
            <p><em>Press any key to start Part 2</em></p>
        </div>
    `
};

// --- Transition between Part 2 and Part 3 ---

const transitionScreen2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Great work — Part 2 complete!</h2>
            <p>Now we will move on to <strong>Part 3</strong>, the final part.</p>
            <p><em>Press any key to read the Part 3 instructions</em></p>
        </div>
    `
};

// --- Part 3 instructions ---

const OPEN_ENDED_EXAMPLE_PASSAGE = [
    "Ghoc and splync . Splync he had gwob gwob . We 're doing a dwoque neight down .",
    "Knurt sneese to to to dwoque Neight Down dwazz down . It 's scis when you shroosh",
    "the throck wherg about this maunch in and out gheint . Knurt . The prerk wrudd is a",
    "dwazz . The prerk gheathe is a blalf which is fuite . Knurt . Why you wherg I did",
    "that ? I do n't ghegging greash because they twieve more sweil . Sneese to grong the",
    "dwazz dwazz but the twoofs off have scuthed about this phu plaiths the blalf Blaint",
    "in-and-out Gheint is the rirm in-and-out Gheint . Thweil , you greash what knime 's",
    "threrb a plause . Knurt , you greash sweil threrb plauses for gwal these drorbs Nalc",
    "gwalph whuile of the Dwazz-Dwazz . Knurt . No , just no do n't gwalph whuile of it",
    "crolt the shroosh crolt . The shroosh prerk wrudd is a blalf to is a dwazz . Knurt .",
    "That 's dwoll a flurl vewn flurl vewn dwaul threrb it Now threrb an grune shreight .",
    "That 's brulf . That 's sprate . We do n't we do n't twieve to yalt , you greash ,",
    "fru . So , knurt , it was brulf uzz . Knurt , it was girchs . So yipe plaith we did",
    "a dwoque splusk of dwoss . So this plaith we phleethed to do wrudd again and we did",
    "it for yisque yisque The Screrf and whadd cloop of dwoan strilges , which you will",
    "phiv why Thweil , we 've scuthed about it a thwipe whealt a cralph here. ",
].join(' ');

const openEndedInstructions1 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 650px; margin: 0 auto; text-align: left;">
            <h2>Part 3 Instructions</h2>
            <p>In this part you will read longer passages where <strong>most words have been
            replaced with nonsense</strong>. 
            <p>After reading each passage you will answer:</p>
            <p style="margin: 16px 30px; font-size: 17px;">
                <em>"What do you think this passage is about?"</em>
            </p>
            <p>We know this might seem pretty difficult with most of words masked with nonsense. 
            But, there are no right or wrong answers! Do your best to understand and make a guess 
            about what the passage is about.</p>
            <p><strong>Important:</strong> the nonsense words are randomly assigned
            and are not secretly related to the real words.</p>
            <p><em>Press any key to see examples</em></p>
        </div>
    `
};

const openEndedInstructions2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 700px; margin: 0 auto; text-align: left;">
            <h2>Part 3 — Examples</h2>

            <p>Here is an example of what a passage will look like:</p>

            <div style="background: #fafafa; border: 1px solid #ddd; border-radius: 6px;
                        padding: 18px 22px; margin: 14px 0 24px 0;
                        font-size: 18px; line-height: 1.8;">
                ${OPEN_ENDED_EXAMPLE_PASSAGE}
            </div>

            <p>Remember, you will be asked what you think this passage is about. 
            Below are examples of <strong>amazing</strong>, <strong>acceptable</strong>,
            and <strong>bad</strong> responses.</p>

            <div style="background: #e8f5e9; border-left: 4px solid #1b5e20;
                        padding: 14px 18px; margin: 14px 0; border-radius: 3px;">
                <p style="margin: 0 0 6px 0; font-weight: bold; color: #1b5e20;">
                    Amazing response -- detailed and specific
                </p>
                <p style="margin: 0;">
                    "Someone is doing a cooking or how-to demonstration, going through
                    steps of a process and making corrections along the way. They keep
                    restarting and emphasizing getting the right technique down,
                    mentioning names of people or things involved, and wrapping up by
                    noting they've discussed this topic before."
                </p>
            </div>

            <div style="background: #f1f8f1; border-left: 4px solid #2e7d32;
                        padding: 14px 18px; margin: 14px 0; border-radius: 3px;">
                <p style="margin: 0 0 6px 0; font-weight: bold; color: #2e7d32;">
                    Acceptable response
                </p>
                <p style="margin: 0;">
                    "The speaker is walking someone through a process or set of instructions,
                    emphasizing how to perform actions correctly and what to watch out for."
                </p>
            </div>

            <div style="background: #f1f8f1; border-left: 4px solid #2e7d32;
                        padding: 14px 18px; margin: 14px 0; border-radius: 3px;">
                <p style="margin: 0 0 6px 0; font-weight: bold; color: #2e7d32;">
                    Acceptable response
                </p>
                <p style="margin: 0;">
                    "A person talking to themselves about something they're trying
                    to convince themselves they're sure of (they're not)."
                </p>
            </div>

            <div style="background: #fff4f4; border-left: 4px solid #c62828;
                        padding: 14px 18px; margin: 14px 0; border-radius: 3px;">
                <p style="margin: 0 0 6px 0; font-weight: bold; color: #c62828;">
                    Bad response — too vague, no attempt at interpretation
                </p>
                <p style="margin: 0; color: #555;">
                    "It's a story."
                </p>
            </div>

            <div style="background: #fff4f4; border-left: 4px solid #c62828;
                        padding: 14px 18px; margin: 14px 0; border-radius: 3px;">
                <p style="margin: 0 0 6px 0; font-weight: bold; color: #c62828;">
                    Bad response — too vague, no attempt at interpretation
                </p>
                <p style="margin: 0; color: #555;">
                    "It's a conversation."
                </p>
            </div>

            <p style="margin-top: 20px;">Do your best with the information you have!
            Even an uncertain interpretation is extremely valuable to us.</p>
            <p><em>Press any key to start Part 3</em></p>
        </div>
    `
};

// ===== SAVING + END =====

const savingScreen = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="text-align: center; padding: 50px;">
            <h2>Saving your data...</h2>
            <p style="font-size: 18px; margin-top: 30px;">
                Please wait — do not close this window. 
                You will be redirected to the final phase of the study.
            </p>
            <div style="margin-top: 30px;">
                <div style="display: inline-block; width: 50px; height: 50px;
                     border: 5px solid #f3f3f3; border-top: 5px solid #2196f3;
                     border-radius: 50%; animation: spin 1s linear infinite;"></div>
            </div>
            <style>
                @keyframes spin {
                    0%   { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        </div>
    `,
    choices: 'NO_KEYS',
    trial_duration: 1000
};

// ===== TIMELINE BLOCK BUILDERS =====
// Each builder returns an array of timeline nodes for one part. Pulled out
// of createTimeline() so demo mode (see below) can assemble a run starting
// from any part, not just the full Part 1 -> 2 -> 3 sequence.

// counter is a mutable { n } ref so trial_number stays contiguous however
// the blocks get assembled.

function buildPart1Block(counter) {
    const events = [
        baselineInstructions1,
        baselineInstructions2,
        practiceInstructions,
    ];

    PRACTICE_TRIAL_DATA.forEach((p) => {
        events.push(createHardcodedTrial(p.passageHtml, p.targetWord, 'practice', counter.n++));
        events.push(createGuessInputTrial());
        events.push(createConfidenceRatingTrial());
        events.push(createFeedbackTrial({ target_word: p.targetWord }));
    });

    events.push(practiceCompleteScreen);

    // Baseline trials with attention checks at ~1/3 and ~2/3
    const totalBaseline = baselineTrialData.length;
    let attnCheckIdx = 0;
    const attnInsertAfter = new Set([
        Math.floor(totalBaseline / 3) - 1,
        Math.floor(2 * totalBaseline / 3) - 1,
    ]);

    baselineTrialData.forEach((trial, i) => {
        events.push(createBaselineTrial(trial, i, totalBaseline, counter.n++));
        events.push(createGuessInputTrial());
        events.push(createConfidenceRatingTrial());
        events.push(createFeedbackTrial(trial));

        if (attnInsertAfter.has(i) && attnCheckIdx < ATTENTION_CHECK_DATA.length) {
            const check = ATTENTION_CHECK_DATA[attnCheckIdx];
            events.push(createHardcodedTrial(check.passageHtml, check.targetWord, 'attention_check', counter.n++));
            events.push(createGuessInputTrial());
            events.push(createConfidenceRatingTrial());
            events.push(createFeedbackTrial({ target_word: check.targetWord }));
            attnCheckIdx++;
        }
    });

    return events;
}

// showTransitionIn: include the "Part N complete!" transition screen that
// leads into this part. Only meaningful when the previous part was actually
// played earlier in this same run — demo mode starting here skips it.
function buildPart2Block(counter, { showTransitionIn }) {
    const events = [];
    if (showTransitionIn) events.push(transitionScreen);
    events.push(samplingInstructions1, samplingInstructions2);

    const totalSampling = samplingTrialData.length;
    samplingTrialData.forEach((trial, i) => {
        events.push(createSamplingTrial(trial, i, totalSampling, counter.n++));
        events.push(createGuessInputTrial());
        events.push(createConfidenceRatingTrial());
        events.push(createFeedbackTrial(trial));
    });

    return events;
}

function buildPart3Block(counter, { showTransitionIn }) {
    const events = [];
    if (showTransitionIn) events.push(transitionScreen2);
    events.push(openEndedInstructions1, openEndedInstructions2);

    const totalOpenEnded = openEndedTrialData.length;
    openEndedTrialData.forEach((trial, i) => {
        events.push(createOpenEndedTrial(trial, i, totalOpenEnded, counter.n++));
    });

    return events;
}

function buildEndingBlock() {
    return [
        savingScreen,
        {
            type: jsPsychPipe,
            action: 'save',
            experiment_id: 'PYSjeESL3lfq',
            filename: `${subjCode}.csv`,
            data_string: () => {
                console.log(`Saving ${consolidatedTrials.length} trials...`);
                if (consolidatedTrials.length > 0) {
                    console.log('Columns:', Object.keys(consolidatedTrials[0]));
                }
                return arrayToCSV(consolidatedTrials);
            },
            on_finish: function (data) {
                if (data.success === false) {
                    console.error('Data upload failed:', data);
                } else {
                    console.log('Data upload successful.');
                }
            }
        },
        {
            type: jsPsychHtmlKeyboardResponse,
            stimulus: function () {
                const surveyURL = getURLParameter('survey_url')
                    || 'https://uwmadison.co1.qualtrics.com/jfe/form/SV_2gBjgNQpFFwXvhQ';
                const surveyWithId = `${surveyURL}${surveyURL.includes('?') ? '&' : '?'}subjCode=${subjCode}`;

                setTimeout(() => { window.location.href = surveyWithId; }, 2000);

                return `
                    <div style="text-align: center; padding: 50px;">
                        <h2>Thank you! You have one more step! </h2>
                        <p style="font-size: 18px; margin: 30px 0;">
                            Your data has been saved successfully.
                        </p>
                        <p style="font-size: 18px; margin: 30px 0;">
                            You will be redirected to the final survey shortly...
                        </p>
                        <p style="font-size: 14px; color: #666; margin-top: 40px;">
                            If you are not redirected automatically,
                            <a href="${surveyWithId}" style="color: #2196f3;">click here</a>.
                        </p>
                    </div>
                `;
            },
            choices: 'NO_KEYS',
            trial_duration: null
        }
    ];
}

// ===== DEMO MODE =====
// ?demo=true lets you click which part to start at, skipping consent and
// fullscreen. The run still saves to DataPipe and redirects to Qualtrics
// at the end, same as a real run — it just starts partway through.

function isDemoMode() {
    const demoParam = getURLParameter('demo');
    return demoParam === 'true' || demoParam === '1';
}

function buildDemoPickerTrial(counter) {
    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: `
            <div style="max-width: 600px; margin: 0 auto; text-align: left;">
                <h2>Demo Mode</h2>
                <p>Choose which part of the experiment to start at:</p>
            </div>
        `,
        choices: ['Part 1 (Baseline)', 'Part 2 (Sampling)', 'Part 3 (Open-Ended)'],
        on_finish: function (data) {
            let block;
            if (data.response === 0) {
                block = [
                    ...buildPart1Block(counter),
                    ...buildPart2Block(counter, { showTransitionIn: true }),
                    ...buildPart3Block(counter, { showTransitionIn: true }),
                    ...buildEndingBlock(),
                ];
            } else if (data.response === 1) {
                block = [
                    ...buildPart2Block(counter, { showTransitionIn: false }),
                    ...buildPart3Block(counter, { showTransitionIn: true }),
                    ...buildEndingBlock(),
                ];
            } else {
                block = [
                    ...buildPart3Block(counter, { showTransitionIn: false }),
                    ...buildEndingBlock(),
                ];
            }
            jsPsych.addNodeToEndOfTimeline({ timeline: block });
        }
    };
}

// ===== TIMELINE =====

async function createTimeline() {
    sublistNumber = await assignSublist();
    console.log(`SubjectCode: ${subjCode} | Sublist: ${sublistNumber} | Seed: ${randomSeed}`);
    await loadAllTrialData();

    const counter = { n: 1 }; // shared trial_number counter across all blocks

    if (isDemoMode()) {
        console.log('Demo mode active — showing part picker.');
        return [buildDemoPickerTrial(counter)];
    }

    return [
        consent,
        enterFullscreen,
        welcome,
        ...buildPart1Block(counter),
        ...buildPart2Block(counter, { showTransitionIn: true }),
        ...buildPart3Block(counter, { showTransitionIn: true }),
        ...buildEndingBlock(),
    ];
}

// ===== ENTRY POINT =====

createTimeline()
    .then(timeline => jsPsych.run(timeline))
    .catch(error => {
        console.error('Error loading experiment:', error);
        document.body.innerHTML = `
            <div style="text-align: center; padding: 50px;">
                <h2>Error Loading Experiment</h2>
                <p>Could not load <code>trial_lists/sublist_${sublistNumber}.csv</code>.</p>
                <p style="color: red;">Error: ${error.message}</p>
            </div>
        `;
    });
