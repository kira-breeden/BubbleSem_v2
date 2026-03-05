// ============================================================
// BubbleSem v2 — Two-Section Experiment
//
// Section 1 (Baseline): some_masked + most_masked trials shuffled
//   - Passage shown with predetermined unmasked words
//   - No reveal interactivity; participant makes guess when ready
//
// Section 2 (Sampling): predetermined word-by-word reveal
//   - All nonce words start masked
//   - "Reveal Next Word" reveals words in a fixed order
//   - Points system active (100 pts per trial, deducted per reveal)
//   - within-subjects: 5 optimal + 5 suboptimal trajectories
//
// CSV files required (per sublist):
//   baseline_sublist_X.csv  — 10 trials (5 some_masked + 5 most_masked)
//     columns: target_word, passage_variant, jabber_passage,
//              target_word_position, masking_level, unmasked_word_indices,
//              entropy, target_probability, [trial_number, ...]
//
//   sampling_sublist_X.csv  — 10 trials (5 optimal + 5 suboptimal)
//     columns: target_word, passage_variant, jabber_passage,
//              target_word_position, reveal_order, sampling_order_type,
//              entropy, target_probability, [trial_number, ...]
//
// URL parameters:
//   sublist=1|2|3|4   (default: 1)
//   subjCode=<string> (default: random ID)
// ============================================================

// ===== GLOBAL STATE =====

let baselineTrialData = [];
let samplingTrialData = [];
let trialSequenceData = {};   // accumulates data across a trial's screens
let consolidatedTrials = [];  // all saved trial rows (pushed at confidence step)
let startTime = null;

// Sampling-specific state (reset each sampling trial)
let revealQueue = [];         // token indices remaining to reveal, in order
let revealedTokenIndices = []; // token indices revealed so far
let revealClickTimes = [];    // [{word_index, revealed_word, time_from_start, num_revealed}]
let trialPoints = 100;
let pointsPerReveal = 0;

// Words that are always shown as real (never masked)
const ARTICLES = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                  'to', 'for', 'of', 'with', 'by'];

// ===== URL PARAMETERS + SUBJECT CODE =====

function getURLParameter(name) {
    return new URLSearchParams(window.location.search).get(name);
}

const subjCode = getURLParameter('subjCode') || jsPsych.randomization.randomID(10);

const sublistParam = getURLParameter('sublist');
let sublistNumber = 1;
if (sublistParam) {
    const parsed = parseInt(sublistParam);
    if (parsed >= 1 && parsed <= 4) {
        sublistNumber = parsed;
    } else {
        console.warn(`Invalid sublist parameter "${sublistParam}", defaulting to 1.`);
    }
}

const randomSeed = Math.floor(Math.random() * 1000000);
const filename = `${subjCode}.csv`;

console.log(`SubjectCode: ${subjCode} | Sublist: ${sublistNumber} | Seed: ${randomSeed}`);

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

// Update the on-screen points counter.
function updatePointsDisplay(points) {
    const el = document.getElementById('points-counter');
    if (el) {
        el.textContent = `Points: ${Math.round(points)}`;
        el.style.color = '#d32f2f';
        setTimeout(() => { el.style.color = '#333'; }, 300);
    }
}

// Convert an array of objects to a CSV string.
function arrayToCSV(data) {
    if (!data.length) return '';
    const headers = Object.keys(data[0]);
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
    const baselineFile = `baseline_sublist_${sublistNumber}.csv`;
    const samplingFile  = `sampling_sublist_${sublistNumber}.csv`;

    const [baseline, sampling] = await Promise.all([
        loadCSV(baselineFile),
        loadCSV(samplingFile)
    ]);

    // Shuffle both independently with seeded RNG so order is reproducible
    const rng = new SeededRandom(randomSeed);
    baselineTrialData = rng.shuffle(baseline);
    samplingTrialData = rng.shuffle(sampling);

    console.log(`Baseline trials: ${baselineTrialData.length}`);
    console.log(`Sampling trials: ${samplingTrialData.length}`);
}

// ===== BASELINE TRIAL =====
// Shows the passage with predetermined masked/unmasked words.
// No reveal interactivity — participant clicks "Make Guess" when ready.

function createBaselineTrial(trial, sectionTrialIndex, totalBaseline) {
    const realSentence   = trial.passage_variant || '';
    const jabberSentence = trial.jabber_passage  || '';
    const realTokens     = tokenizeSentence(realSentence);
    const jabberTokens   = tokenizeSentence(jabberSentence);
    // Use jabberTokens for targetTokenIdx — the render loop iterates jabberTokens,
    // so the index must come from the same tokenization.
    const targetTokenIdx = wordPosToTokenIndex(jabberTokens, trial.target_word_position);
    const maskingLevel   = trial.masking_level || 'some';

    // Convert unmasked word positions (0-indexed, punctuation-ignored) to token indices.
    const unmaskedWordPositions = parseJSONColumn(trial.unmasked_word_indices);
    const unmaskedTokenIdxSet   = new Set(
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
                section:                'baseline',
                section_trial_index:    sectionTrialIndex + 1,
                masking_level:          maskingLevel,
                target_word:            trial.target_word,
                target_word_position:   trial.target_word_position,
                entropy:                trial.entropy,
                target_probability:     trial.target_probability,
                real_passage:           realSentence,
                jabber_passage:         jabberSentence,
                unmasked_word_indices:  JSON.stringify(unmaskedWordPositions),
            };

            let html = `
                <div class="trial-counter">
                    Section 1 &mdash; Trial ${sectionTrialIndex + 1} of ${totalBaseline}
                </div>
                <div class="sentence-container" id="sentence-container">
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

// ===== SAMPLING TRIAL =====
// All nonce words start masked. "Reveal Next Word" reveals them one at a time
// in a predetermined order. Points are deducted per reveal.

function createSamplingTrial(trial, sectionTrialIndex, totalSampling) {
    const realSentence   = trial.passage_variant || '';
    const jabberSentence = trial.jabber_passage  || '';
    const realTokens     = tokenizeSentence(realSentence);
    const jabberTokens   = tokenizeSentence(jabberSentence);
    const targetTokenIdx = wordPosToTokenIndex(jabberTokens, trial.target_word_position);

    // Parse reveal order: CSV stores 0-indexed word positions (ignoring punctuation).
    // Convert each to a token index for DOM manipulation, filtering out invalid entries
    // and the target word. Keep the word-position alongside for data saving.
    const revealOrderWordPositions = parseJSONColumn(trial.reveal_order);
    const revealOrderPairs = revealOrderWordPositions
        .map(wordPos => ({ wordPos, tokenIdx: wordPosToTokenIndex(jabberTokens, wordPos) }))
        .filter(({ tokenIdx }) => tokenIdx >= 0 && tokenIdx !== targetTokenIdx);

    // Token-index → word-position reverse map (for saving word positions in output data)
    const tokenToWordPos = buildTokenToWordPosMap(jabberTokens);

    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: function () {
            startTime = Date.now();
            trialPoints     = 100;
            revealQueue     = [...revealOrderPairs]; // each entry: {wordPos, tokenIdx}
            revealedTokenIndices = [];
            revealClickTimes = [];
            pointsPerReveal = revealQueue.length > 0
                ? Math.round((100 / revealQueue.length) * 100) / 100
                : 0;

            trialSequenceData = {
                subjCode:             subjCode,
                sublist:              sublistNumber,
                random_seed:          randomSeed,
                section:              'sampling',
                section_trial_index:  sectionTrialIndex + 1,
                sampling_order_type:  trial.sampling_order_type || '',
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
                        Section 2 &mdash; Trial ${sectionTrialIndex + 1} of ${totalSampling}
                    </div>
                    <div class="points-counter" id="points-counter">Points: ${trialPoints}</div>
                    <div class="sentence-container" id="sentence-container">
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
                    html += `<span class="word sampling-masked" id="word-tok-${i}"
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
                    word_position:   wordPos,          // 0-indexed word pos, ignoring punct
                    revealed_word:   realTokens[tokenIdx],
                    time_from_start: Date.now() - startTime,
                    num_revealed:    revealedTokenIndices.length
                });

                // Update word in DOM
                const wordEl = document.getElementById(`word-tok-${tokenIdx}`);
                if (wordEl) {
                    wordEl.textContent = realTokens[tokenIdx];
                    wordEl.classList.remove('sampling-masked');
                    wordEl.classList.add('sampling-revealed');
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
        on_finish: function (data) {
            trialSequenceData.guess    = data.response.target_word_guess;
            trialSequenceData.rt_guess = data.rt;
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

const welcome = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h1>Word Guessing Experiment</h1>
            <p>In this experiment you will read passages containing made-up nonsense words
            and try to guess the meaning of one <strong>bolded</strong> target word.</p>
            <p>The experiment has two parts. You will receive instructions for each part
            before it begins.</p>
            <p><em>Press any key to continue</em></p>
        </div>
    `
};

// --- Section 1 instructions ---

const baselineInstructions1 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Part 1 Instructions</h2>
            <p>In this part you will see passages where <strong>some words have been
            replaced with made-up nonsense words</strong>. The real words you can still
            see will give you context clues.</p>
            <p>Your job:</p>
            <ol>
                <li>Read the passage carefully.</li>
                <li>Try to figure out what the <strong>bolded word</strong> means.</li>
                <li>When you are ready to guess, click <strong>Make Guess</strong>.</li>
                <li>Type your best ONE-WORD guess.</li>
                <li>Rate your confidence.</li>
                <li>You will see the correct answer before moving on.</li>
            </ol>
            <p><em>Press any key to continue</em></p>
        </div>
    `
};

const baselineInstructions2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Part 1 — Tips</h2>
            <p>Some passages will have more nonsense words than others — that is intentional.</p>
            <p>Use every real word you can see as a clue. Even if you are unsure, make your
            best one-word guess before moving on.</p>
            <p><strong>Example:</strong></p>
            <p style="margin-left: 20px; font-style: italic;">
                "The glorp gleamed in the deng morning <strong>glosh</strong>."
            </p>
            <p>You might guess <em>sun</em>, <em>light</em>, or <em>air</em> — all
            reasonable one-word guesses based on context. That is the right level of
            specificity.</p>
            <p><strong>Please use ONE WORD guesses only.</strong></p>
            <p><em>Press any key to start Part 1</em></p>
        </div>
    `
};

// --- Transition between sections ---

const transitionScreen = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Great work — Part 1 complete!</h2>
            <p>Now we will move on to <strong>Part 2</strong>, which works a little
            differently.</p>
            <p><em>Press any key to read the Part 2 instructions</em></p>
        </div>
    `
};

// --- Section 2 instructions ---

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
                <li>Click <strong>Reveal Next Word</strong> to reveal words one at a time.
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
            <p>The words will be revealed in a fixed order — you cannot choose which word
            is revealed next, only when to stop and guess.</p>
            <p><strong>Please use ONE WORD guesses only.</strong></p>
            <p><em>Press any key to start Part 2</em></p>
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

// ===== TIMELINE =====

async function createTimeline() {
    await loadAllTrialData();

    const timeline = [
        consent,
        welcome,
        baselineInstructions1,
        baselineInstructions2
    ];

    // --- Section 1: baseline trials (some_masked + most_masked, shuffled) ---
    const totalBaseline = baselineTrialData.length;
    baselineTrialData.forEach((trial, i) => {
        timeline.push(createBaselineTrial(trial, i, totalBaseline));
        timeline.push(createGuessInputTrial());
        timeline.push(createConfidenceRatingTrial());
        timeline.push(createFeedbackTrial(trial));
    });

    // --- Transition ---
    timeline.push(transitionScreen);
    timeline.push(samplingInstructions1);
    timeline.push(samplingInstructions2);

    // --- Section 2: sampling trials ---
    const totalSampling = samplingTrialData.length;
    samplingTrialData.forEach((trial, i) => {
        timeline.push(createSamplingTrial(trial, i, totalSampling));
        timeline.push(createGuessInputTrial());
        timeline.push(createConfidenceRatingTrial());
        timeline.push(createFeedbackTrial(trial));
    });

    // --- Saving screen + data pipe save ---
    timeline.push(savingScreen);

    timeline.push({
        type: jsPsychPipe,
        action: 'save',
        experiment_id: 'YOUR_EXPERIMENT_ID_HERE', // replace with your DataPipe experiment ID
        filename: filename,
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
    });

    // --- Thank-you / redirect ---
    timeline.push({
        type: jsPsychHtmlKeyboardResponse,
        stimulus: function () {
            const surveyURL = getURLParameter('survey_url')
                || 'https://uwmadison.co1.qualtrics.com/jfe/form/SV_2hiSFCTKI8N4Wbk';
            const surveyWithId = `${surveyURL}${surveyURL.includes('?') ? '&' : '?'}subjCode=${subjCode}`;

            setTimeout(() => { window.location.href = surveyWithId; }, 2000);

            return `
                <div style="text-align: center; padding: 50px;">
                    <h2>Thank you!</h2>
                    <p style="font-size: 18px; margin: 30px 0;">
                        Your data has been saved successfully.
                    </p>
                    <p style="font-size: 16px; margin: 40px 0;">
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
    });

    return timeline;
}

// ===== ENTRY POINT =====

createTimeline()
    .then(timeline => jsPsych.run(timeline))
    .catch(error => {
        console.error('Error loading experiment:', error);
        document.body.innerHTML = `
            <div style="text-align: center; padding: 50px;">
                <h2>Error Loading Experiment</h2>
                <p>Could not load trial lists for sublist ${sublistNumber}.</p>
                <p>Make sure <code>baseline_sublist_${sublistNumber}.csv</code> and
                   <code>sampling_sublist_${sublistNumber}.csv</code> exist.</p>
                <p style="color: red;">Error: ${error.message}</p>
            </div>
        `;
    });
