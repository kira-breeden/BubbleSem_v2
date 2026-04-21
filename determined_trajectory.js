// ============================================================
// determined_trajectory.js
// Saved from experiment_v2.js for later use.
//
// This file contains the Phase 2 "sampling" trial logic:
// all nonce words start masked and are revealed one at a time
// in a predetermined optimal or suboptimal order, with a
// points system that deducts per reveal.
//
// Dependencies (must be present in the host experiment file):
//   - jsPsych, jsPsychHtmlButtonResponse
//   - Global state: trialSequenceData, consolidatedTrials, startTime
//   - Utility functions: tokenizeSentence, isPunct, wordPosToTokenIndex,
//     buildTokenToWordPosMap, parseJSONColumn, isAutoRevealed
//   - URL/config: subjCode, sublistNumber, randomSeed
// ============================================================

// ===== SAMPLING-SPECIFIC GLOBAL STATE =====
// (Reset at the start of each sampling trial)

let revealQueue          = [];  // {wordPos, tokenIdx} entries remaining to reveal
let revealedTokenIndices = [];  // token indices revealed so far
let revealClickTimes     = [];  // [{word_position, revealed_word, time_from_start, num_revealed}]
let trialPoints          = 100;
let pointsPerReveal      = 0;

// Update the on-screen points counter.
function updatePointsDisplay(points) {
    const el = document.getElementById('points-counter');
    if (el) {
        el.textContent = `Points: ${Math.round(points)}`;
        el.style.color = '#d32f2f';
        setTimeout(() => { el.style.color = '#333'; }, 300);
    }
}

// ===== SAMPLING TRIAL =====
// All nonce words start masked. "Reveal Next Word" reveals them one at a time
// in a predetermined order (optimal or suboptimal). Points are deducted per reveal.

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
                    html += `<span class="word clickable" id="word-tok-${i}"
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
                    wordEl.classList.remove('clickable');
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

// ===== SAMPLING INSTRUCTION SCREENS =====

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

// ===== TIMELINE SNIPPET =====
// To use this in a timeline, replace the Phase 2 block with:
//
//   timeline.push(transitionScreen);
//   timeline.push(samplingInstructions1);
//   timeline.push(samplingInstructions2);
//
//   const totalSampling = samplingTrialData.length;
//   samplingTrialData.forEach((trial, i) => {
//       timeline.push(createSamplingTrial(trial, i, totalSampling));
//       timeline.push(createGuessInputTrial());
//       timeline.push(createConfidenceRatingTrial());
//       timeline.push(createFeedbackTrial(trial));
//   });
