"""
generate_v2_sublists.py
=======================
Generates baseline and sampling trial-list CSV files for BubbleSem v2.

Input
-----
final_pilot_stimuli.csv  (same format as before, with 4 new columns added)

Required columns (pre-existing):
    og_passage_seed_number  -- groups the 4 entropy variants of each target passage
    passage_seed_num        -- unique ID for each row
    target_word
    passage_variant         -- real English passage
    jabber_passage
    target_word_position    -- 0-indexed word position of target (ignoring punctuation)
    entropy
    target_probability
    target_log_probability
    target_pos

New columns you must add (fill with [] placeholders until ready):
    unmasked_word_indices_some  -- JSON list of word positions to show as real in some_masked
    unmasked_word_indices_most  -- JSON list of word positions to show as real in most_masked
    reveal_order_optimal        -- JSON list of word positions in optimal reveal order
    reveal_order_suboptimal     -- JSON list of word positions in suboptimal reveal order

Output
------
trial_lists/sublist_1.csv ... trial_lists/sublist_16.csv
    columns: phase, target_word, passage_variant, jabber_passage, target_word_position,
             masking_level, unmasked_word_indices, sampling_order_type, reveal_order,
             entropy, target_probability, target_log_probability, target_pos,
             og_passage_seed_number

    Rows 1–10  have phase='baseline'  (masking_level + unmasked_word_indices populated)
    Rows 11–20 have phase='sampling'  (sampling_order_type + reveal_order populated)

Design
------
20 target concepts (unique og_passage_seed_numbers), each with 4 entropy variants.
4 conditions: some_masked | most_masked | sampling_optimal | sampling_suboptimal
16 sublists (4 condition rotations x 4 entropy rotations), each containing 20 trials
(5 per condition).

Each sublist is indexed by a pair (condition_rotation in 0..3, entropy_rotation in 0..3):
    sublist_num = condition_rotation * 4 + entropy_rotation + 1

Latin square — condition per group per condition_rotation:
    Concepts are divided into 4 fixed groups of 5 (G0..G3).
    For condition_rotation CR, group Gk gets condition[(k + CR) % 4].

    CR=0: G0=some  G1=most  G2=opt   G3=sub
    CR=1: G0=most  G1=opt   G2=sub   G3=some
    CR=2: G0=opt   G1=sub   G2=some  G3=most
    CR=3: G0=sub   G1=some  G2=most  G3=opt

Entropy-level assignment per concept per sublist:
    Variants for each concept are sorted low->high entropy (levels 0..3).
    Concept with global_concept_idx i, entropy_rotation ER uses entropy level (i + ER) % 4.

Full coverage guarantee:
    Each concept appears in all 16 sublists, covering all 4 conditions x 4 entropy levels.
    Across 16 participants (one per sublist), every passage is measured in every condition.
    Total: 20 concepts x 4 entropy variants x 4 conditions = 320 passage x condition observations.
    Minimum participants for full coverage: 16 (one per sublist); scale as multiples of 16.
"""

import pandas as pd
import numpy as np
import json
import sys
import re

# ── Configuration ────────────────────────────────────────────────────────────

INPUT_CSV          = 'all_stimuli.csv'
SEED               = 42          # for reproducible concept→group assignment
N_CONCEPTS         = 20          # expected unique target concepts
N_CONDITIONS       = 4
N_ENTROPY_LEVELS   = 4
N_SUBLISTS         = N_CONDITIONS * N_ENTROPY_LEVELS   # 16
CONCEPTS_PER_GROUP = 5    # N_CONCEPTS / N_CONDITIONS

CONDITIONS = ['some_masked', 'most_masked', 'sampling_optimal', 'sampling_suboptimal']

# Columns to carry through to output files
SHARED_COLS = [
    'target_word', 'passage_variant', 'jabber_passage',
    'target_word_position', 'entropy', 'target_probability',
    'target_log_probability', 'target_pos', 'og_passage_seed_number'
]

# ── Load data ─────────────────────────────────────────────────────────────────

print(f'Loading {INPUT_CSV}...')
df = pd.read_csv(INPUT_CSV)

# ── Compute target_word_position if missing ───────────────────────────────────
# Mirrors JS tokenizeSentence + isPunct + wordPosToTokenIndex exactly.
# Punctuation characters (.,!?;:'") are split off as separate tokens and
# excluded from the 0-indexed word position count.

_PUNCT = set('.,!?;:\'"')

def _tokenize(sentence):
    tokens = []
    for word in str(sentence).split(' '):
        m = re.match(r'^([^.,!?;:\'"]*)([ .,!?;:\'"]*)', word)
        if m:
            word_part = m.group(1)
            punct_part = m.group(2).strip()  # remove any trailing space
            if word_part:
                tokens.append(word_part)
            for ch in punct_part:
                if ch in _PUNCT:
                    tokens.append(ch)
        else:
            tokens.append(word)
    return tokens

def _find_target_word_position(passage, target_word):
    """Return 0-indexed word position of target_word in passage, ignoring punctuation tokens.
    Handles possessives (e.g. "defendant's" matches target "defendant")."""
    target_lower = target_word.lower()
    word_pos = 0
    for token in _tokenize(passage):
        if len(token) == 1 and token in _PUNCT:
            continue
        token_lower = token.lower()
        # Match exact form or possessive with straight (U+0027) or curly (U+2019) apostrophe.
        # Curly-apostrophe possessives (e.g. "defendant\u2019s") are kept as one token by
        # the tokenizer (U+2019 is not in the punct set) and should map to one word index.
        if (token_lower == target_lower
                or token_lower.startswith(target_lower + "'")
                or token_lower.startswith(target_lower + '\u2019')):
            return word_pos
        word_pos += 1
    return -1  # target not found

if 'target_word_position' not in df.columns or df['target_word_position'].isna().all():
    df['target_word_position'] = df.apply(
        lambda r: _find_target_word_position(r['passage_variant'], r['target_word']), axis=1
    )
    n_missing = (df['target_word_position'] == -1).sum()
    print(f'  Computed target_word_position (0-indexed, punct-excluded).')
    if n_missing > 0:
        print(f'  WARNING: {n_missing} rows where target word was not found in passage:')
        bad = df[df['target_word_position'] == -1][['passage_seed_num', 'target_word', 'passage_variant']]
        for _, row in bad.iterrows():
            print(f'    seed={row["passage_seed_num"]} target="{row["target_word"]}"  passage: {row["passage_variant"][:80]}')

# Add placeholder columns if they are not yet present
placeholder_cols = [
    'unmasked_word_indices_some',
    'unmasked_word_indices_most',
    'reveal_order_optimal',
    'reveal_order_suboptimal',
]
for col in placeholder_cols:
    if col not in df.columns:
        df[col] = '[]'
        print(f'  Added placeholder column: {col}')

# ── Group by target_word ─────────────────────────────────────────────────────

groups = {}
for target_word, group_df in df.groupby('target_word'):
    sorted_group = group_df.sort_values('entropy').reset_index(drop=True)
    groups[target_word] = sorted_group

n_concepts_found = len(groups)
print(f'Found {n_concepts_found} unique target words.')

# Warn if any target word has fewer than 4 entropy variants
for target_word, g in groups.items():
    if len(g) < N_ENTROPY_LEVELS:
        print(f'  WARNING: target_word="{target_word}" has only {len(g)} variants '
              f'(need {N_ENTROPY_LEVELS}). Some sublists may reuse entropy levels.')

if n_concepts_found != N_CONCEPTS:
    print(f'\nWARNING: Expected {N_CONCEPTS} target words but found {n_concepts_found}. '
          f'Proceeding anyway — group sizes will be adjusted.')

# ── Assign concepts to 4 fixed groups ────────────────────────────────────────

concept_keys = list(groups.keys())            # list of target words
rng = np.random.default_rng(SEED)
shuffled_keys = rng.permutation(concept_keys).tolist()

# Split into N_CONDITIONS groups as evenly as possible
concept_groups = []
n = len(shuffled_keys)
base_size, remainder = divmod(n, N_CONDITIONS)
start = 0
for i in range(N_CONDITIONS):
    size = base_size + (1 if i < remainder else 0)
    concept_groups.append(shuffled_keys[start:start + size])
    start += size

print('\nTarget-word-to-group assignment:')
for gi, group in enumerate(concept_groups):
    print(f'  G{gi}: {group}')

# ── 16-sublist generation (condition_rotation x entropy_rotation) ─────────────

print('\nGenerating sublists...')

for condition_rotation in range(N_CONDITIONS):
    for entropy_rotation in range(N_ENTROPY_LEVELS):
        sublist_num = condition_rotation * N_ENTROPY_LEVELS + entropy_rotation + 1
        baseline_rows = []
        sampling_rows = []

        for group_idx, target_list in enumerate(concept_groups):
            # Which condition does this group get for this condition_rotation?
            condition = CONDITIONS[(group_idx + condition_rotation) % N_CONDITIONS]

            for target_word in target_list:
                concept_df = groups[target_word]
                n_variants = len(concept_df)

                # Entropy level rotates independently of condition rotation.
                # global_concept_idx spreads entropy levels across concepts within
                # the same sublist so no single condition sees all the same level.
                global_concept_idx = shuffled_keys.index(target_word)
                entropy_level = (global_concept_idx + entropy_rotation) % min(n_variants, N_ENTROPY_LEVELS)

                row = concept_df.iloc[entropy_level]
                base = {col: row.get(col, '') for col in SHARED_COLS}

                if condition == 'some_masked':
                    baseline_rows.append({
                        **base,
                        'phase':                'baseline',
                        'masking_level':        'some',
                        'unmasked_word_indices': row['unmasked_word_indices_some'],
                        'sampling_order_type':  '',
                        'reveal_order':         '',
                    })

                elif condition == 'most_masked':
                    baseline_rows.append({
                        **base,
                        'phase':                'baseline',
                        'masking_level':        'most',
                        'unmasked_word_indices': row['unmasked_word_indices_most'],
                        'sampling_order_type':  '',
                        'reveal_order':         '',
                    })

                elif condition == 'sampling_optimal':
                    sampling_rows.append({
                        **base,
                        'phase':                'sampling',
                        'masking_level':        '',
                        'unmasked_word_indices': '',
                        'sampling_order_type': 'optimal',
                        'reveal_order':         row['reveal_order_optimal'],
                    })

                elif condition == 'sampling_suboptimal':
                    sampling_rows.append({
                        **base,
                        'phase':                'sampling',
                        'masking_level':        '',
                        'unmasked_word_indices': '',
                        'sampling_order_type': 'suboptimal',
                        'reveal_order':         row['reveal_order_suboptimal'],
                    })

        # Combine: baseline rows first, then sampling rows
        combined_df = pd.DataFrame(baseline_rows + sampling_rows)

        # Canonical column order
        col_order = [
            'phase', 'target_word', 'passage_variant', 'jabber_passage',
            'target_word_position', 'masking_level', 'unmasked_word_indices',
            'sampling_order_type', 'reveal_order',
            'entropy', 'target_probability', 'target_log_probability',
            'target_pos', 'og_passage_seed_number'
        ]
        combined_df = combined_df[[c for c in col_order if c in combined_df.columns]]

        sublist_file = f'trial_lists/sublist_{sublist_num}.csv'
        combined_df.to_csv(sublist_file, index=False)

        n_baseline = len(baseline_rows)
        n_sampling = len(sampling_rows)
        n_some = sum(1 for r in baseline_rows if r['masking_level'] == 'some')
        n_most = sum(1 for r in baseline_rows if r['masking_level'] == 'most')
        n_opt  = sum(1 for r in sampling_rows if r['sampling_order_type'] == 'optimal')
        n_sub  = sum(1 for r in sampling_rows if r['sampling_order_type'] == 'suboptimal')

        print(f'\n  Sublist {sublist_num:2d} (CR={condition_rotation}, ER={entropy_rotation}):')
        print(f'    {sublist_file}  — {len(combined_df)} trials total')
        print(f'      baseline ({n_baseline}): {n_some} some_masked, {n_most} most_masked')
        print(f'      sampling ({n_sampling}): {n_opt} optimal, {n_sub} suboptimal')

# ── Validation ────────────────────────────────────────────────────────────────

print('\nValidation:')

# 1. Each target word appears exactly once per sublist (no repeated concepts)
all_ok = True
for sublist_num in range(1, N_SUBLISTS + 1):
    df_sl = pd.read_csv(f'trial_lists/sublist_{sublist_num}.csv')
    all_targets = list(df_sl['target_word'])
    dupes = [t for t in set(all_targets) if all_targets.count(t) > 1]
    if dupes:
        print(f'  FAIL sublist {sublist_num:2d}: duplicate target_words: {dupes}')
        all_ok = False
if all_ok:
    print(f'  OK   all {N_SUBLISTS} sublists: no duplicate target_words within any sublist')

# 2. Every passage (target_word x entropy level) appears in every condition exactly once
# Build a dict: target_word -> {(entropy_rounded, condition): count}
coverage = {}
for sublist_num in range(1, N_SUBLISTS + 1):
    df_sl = pd.read_csv(f'trial_lists/sublist_{sublist_num}.csv')

    for _, row in df_sl.iterrows():
        tw = row['target_word']
        if row['phase'] == 'baseline':
            key = (round(float(row['entropy']), 6), row['masking_level'])
        else:
            key = (round(float(row['entropy']), 6), row['sampling_order_type'])
        coverage.setdefault(tw, {})
        coverage[tw][key] = coverage[tw].get(key, 0) + 1

coverage_ok = True
for tw, cell_counts in coverage.items():
    if len(cell_counts) != N_SUBLISTS:
        # Duplicate entropy levels within a target word cause this — treat as acceptable.
        print(f'  NOTE target_word "{tw}": {len(cell_counts)} unique (entropy, condition) '
              f'pairings (expected {N_SUBLISTS}) — likely duplicate entropy variants in source data')
    for cell, count in cell_counts.items():
        if count != 1:
            print(f'  NOTE target_word "{tw}": cell {cell} appears {count} times '
                  f'— duplicate entropy variant treated as distinct passage')
if coverage_ok:
    print(f'  OK   coverage check complete (see NOTEs above for any duplicate entropy variants)')

# 3. Each participant sees all 4 conditions and correct phase ordering (baseline first)
conditions_ok = True
for sublist_num in range(1, N_SUBLISTS + 1):
    df_sl = pd.read_csv(f'trial_lists/sublist_{sublist_num}.csv')
    b = df_sl[df_sl['phase'] == 'baseline']
    s = df_sl[df_sl['phase'] == 'sampling']
    seen_conditions = set(b['masking_level']).union(set(s['sampling_order_type']))
    expected = {'some', 'most', 'optimal', 'suboptimal'}
    if seen_conditions != expected:
        print(f'  FAIL sublist {sublist_num:2d}: conditions present = {seen_conditions}')
        conditions_ok = False
    # Verify baseline rows come before sampling rows
    phases = list(df_sl['phase'])
    if phases != sorted(phases, key=lambda p: 0 if p == 'baseline' else 1):
        print(f'  FAIL sublist {sublist_num:2d}: sampling rows appear before baseline rows')
        conditions_ok = False
    if len(b) != 10 or len(s) != 10:
        print(f'  FAIL sublist {sublist_num:2d}: expected 10 baseline + 10 sampling, '
              f'got {len(b)} + {len(s)}')
        conditions_ok = False
if conditions_ok:
    print(f'  OK   all {N_SUBLISTS} sublists: 10 baseline + 10 sampling, correct ordering')

print('\nDone.')