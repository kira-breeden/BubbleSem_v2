"""
generate_v2_sublists.py
=======================
Generates Part 1 (baseline) + Part 2 (sampling) trial-list CSV files for
BubbleSem v2.

Part 3 (open-ended longer passages) is a separate stimulus set loaded from
open_ended_passages.csv — it does not vary by sublist.

Input
-----
all_stimuli.csv  (with the following columns)

Required columns (pre-existing):
    og_passage_seed_number  -- groups the 4 entropy variants of each target passage
    passage_seed_num        -- unique ID for each row
    target_word
    passage_variant         -- real English passage shown in Part 1/2
    jabber_passage          -- jabberwocky version of passage_variant
    target_word_position    -- 0-indexed word position of target (ignoring punctuation)
    entropy
    target_probability
    target_log_probability
    target_pos

Columns to add before running (placeholders added automatically if absent):
    unmasked_word_indices_some  -- JSON list of word positions to reveal in some_masked (40%)
    unmasked_word_indices_most  -- JSON list of word positions to reveal in most_masked (20%)

Output
------
trial_lists/sublist_1.csv ... trial_lists/sublist_16.csv
    columns: trial_number, condition, real_passage, jabber_passage,
             target_word, target_word_position, unmasked_word_indices,
             reveal_order, entropy, target_probability, target_log_probability,
             target_pos, og_passage_seed_number, passage_id

    All 20 concept rows per sublist have condition in {some_masked, most_masked, sampling}.
    5  rows have condition='some_masked'  (40% of greedy-trajectory words revealed)
    5  rows have condition='most_masked'  (20% of greedy-trajectory words revealed)
    10 rows have condition='sampling'     (all words start masked; revealed one at a
                                            time, in greedy-trajectory order, during
                                            the task — see reveal_order)

Design
------
20 target concepts, each with 4 entropy variants.
3 Part 1/2 conditions: some_masked | most_masked | sampling
16 sublists = 4 condition rotations x 4 entropy rotations

    sublist_num = condition_rotation * 4 + entropy_rotation + 1   (1..16)

Concepts split into 4 groups of 5 (G0..G3). Each condition rotation (CR)
assigns every group a role from a fixed role sequence, cycled by group index:

    ROLE_SEQUENCE = ['some_masked', 'most_masked', 'sampling', 'sampling']
    role(group, CR) = ROLE_SEQUENCE[(group - CR) % 4]

so that across CR=0..3, every group takes each position in ROLE_SEQUENCE
exactly once: some_masked once, most_masked once, sampling twice.

Each participant sees all 20 concepts: 5 some_masked + 5 most_masked + 10 sampling.

Entropy-level assignment per concept per sublist:
    Variants sorted low->high entropy (levels 0..3).
    Concept with global index i, entropy_rotation ER uses level (i + ER) % 4.
    This spreads entropy levels across concepts within each sublist so that
    no single condition disproportionately samples high or low entropy.

Full coverage across 16 sublists:
    Each concept appears in some_masked once and most_masked once per entropy
    level (4 CRs x 4 ERs = 16 sublists, but a concept's role is fixed for a
    given CR across all 4 ERs, so its 4 some_masked/most_masked appearances —
    whichever CR lands on that role — sweep all 4 entropy levels exactly once).
    Because sampling occupies 2 of the 4 role slots per concept, each concept's
    (target_word, entropy) pair appears in sampling exactly twice across the
    16 sublists — twice the replication of the masking conditions, matching
    the 5:5:10 per-sublist ratio.
    Minimum participants for full coverage: 16 (one per sublist).
"""

import json
import os
import pandas as pd
import numpy as np
import re

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_CSV             = 'all_stimuli.csv'
SEED                  = 42
N_CONCEPTS            = 20
N_ROLE_GROUPS         = 4   # concepts split into 4 groups of 5
N_ENTROPY_LEVELS      = 4
N_CONDITION_ROTATIONS = 4
N_SUBLISTS            = N_CONDITION_ROTATIONS * N_ENTROPY_LEVELS  # 16

ROLE_SEQUENCE = ['some_masked', 'most_masked', 'sampling', 'sampling']

SHARED_COLS = [
    'target_word', 'passage_variant', 'jabber_passage',
    'target_word_position', 'entropy', 'target_probability',
    'target_log_probability', 'target_pos', 'og_passage_seed_number'
]

OPEN_ENDED_CSV = 'open_ended_passages.csv'

COL_ORDER = [
    'trial_number', 'condition',
    'real_passage', 'jabber_passage',
    'target_word', 'target_word_position',
    'unmasked_word_indices', 'reveal_order',
    'entropy', 'target_probability', 'target_log_probability',
    'target_pos', 'og_passage_seed_number',
    'passage_id',
]

# ── Load data ──────────────────────────────────────────────────────────────────

print(f'Loading {INPUT_CSV}...')
df = pd.read_csv(INPUT_CSV)

# ── Compute target_word_position if missing ────────────────────────────────────
# Mirrors JS tokenizeSentence + isPunct + wordPosToTokenIndex exactly.

_PUNCT = set('.,!?;:\'"')

def _tokenize(sentence):
    tokens = []
    for word in str(sentence).split(' '):
        m = re.match(r'^([^.,!?;:\'"]*)([ .,!?;:\'"]*)', word)
        if m:
            word_part  = m.group(1)
            punct_part = m.group(2).strip()
            if word_part:
                tokens.append(word_part)
            for ch in punct_part:
                if ch in _PUNCT:
                    tokens.append(ch)
        else:
            tokens.append(word)
    return tokens

def _find_target_word_position(passage, target_word):
    """0-indexed word position of target_word in passage, ignoring punctuation tokens.
    Handles possessives (e.g. "defendant's" matches target "defendant")."""
    target_lower = target_word.lower()
    word_pos = 0
    for token in _tokenize(passage):
        if len(token) == 1 and token in _PUNCT:
            continue
        token_lower = token.lower()
        if (token_lower == target_lower
                or token_lower.startswith(target_lower + "'")
                or token_lower.startswith(target_lower + '’')):
            return word_pos
        word_pos += 1
    return -1

if 'target_word_position' not in df.columns or df['target_word_position'].isna().all():
    df['target_word_position'] = df.apply(
        lambda r: _find_target_word_position(r['passage_variant'], r['target_word']), axis=1
    )
    n_missing = (df['target_word_position'] == -1).sum()
    print('  Computed target_word_position (0-indexed, punct-excluded).')
    if n_missing > 0:
        print(f'  WARNING: {n_missing} rows where target word was not found:')
        bad = df[df['target_word_position'] == -1][['passage_seed_num', 'target_word', 'passage_variant']]
        for _, row in bad.iterrows():
            print(f'    seed={row["passage_seed_num"]} target="{row["target_word"]}"  '
                  f'passage: {row["passage_variant"][:80]}')

# ── Load greedy trajectories and compute unmasked/reveal-order columns ────────
# Words are pre-revealed from the START of the (uncapped) greedy trajectory,
# most informative first. Matches the case study's condition_trajectories.py:
#   some_masked -> top 40% of the trajectory revealed (more context shown)
#   most_masked -> top 20% of the trajectory revealed (fewer words shown)
#   sampling    -> the full trajectory, revealed one word at a time in order

GREEDY_CSV = 'passage_greedy_trajectories.csv'
print(f'Loading {GREEDY_CSV}...')
greedy_df = pd.read_csv(GREEDY_CSV, usecols=['real_passage', 'greedy_words', 'greedy_indices'])

SOME_MASKED_PCT = 0.40   # some_masked: 40% revealed  -> some words remain masked
MOST_MASKED_PCT = 0.20   # most_masked: 20% revealed  -> most words remain masked

def _first_pct(indices_json, pct):
    """Return first pct% (by count) of a JSON list of indices, as a JSON string."""
    indices = json.loads(indices_json) if isinstance(indices_json, str) else []
    n = max(1, round(len(indices) * pct))
    return json.dumps(indices[:n])

greedy_df['unmasked_word_indices_some'] = greedy_df['greedy_indices'].apply(
    lambda x: _first_pct(x, SOME_MASKED_PCT)
)
greedy_df['unmasked_word_indices_most'] = greedy_df['greedy_indices'].apply(
    lambda x: _first_pct(x, MOST_MASKED_PCT)
)

df = df.merge(
    greedy_df[['real_passage', 'unmasked_word_indices_some', 'unmasked_word_indices_most', 'greedy_indices']],
    left_on='passage_variant',
    right_on='real_passage',
    how='left'
).drop(columns='real_passage')
df = df.rename(columns={'greedy_indices': 'reveal_order'})

n_missing = df['unmasked_word_indices_some'].isna().sum()
if n_missing > 0:
    print(f'  WARNING: {n_missing} rows had no match in {GREEDY_CSV} — defaulting to []')
    df['unmasked_word_indices_some'] = df['unmasked_word_indices_some'].fillna('[]')
    df['unmasked_word_indices_most'] = df['unmasked_word_indices_most'].fillna('[]')
    df['reveal_order']               = df['reveal_order'].fillna('[]')
else:
    print(f'  OK   greedy indices merged for all {len(df)} rows')

# ── Load open-ended passages (same for all sublists) ──────────────────────────

print(f'Loading {OPEN_ENDED_CSV}...')
oe_df = pd.read_csv(OPEN_ENDED_CSV)
for required in ['id', 'jabber_text_short', 'original_text_short']:
    if required not in oe_df.columns:
        raise ValueError(f'{OPEN_ENDED_CSV} must have a "{required}" column.')
open_ended_rows = [
    {
        'condition':    'open_ended',
        'passage_id':   row['id'],
        'real_passage': row['original_text_short'],
        'jabber_passage': row['jabber_text_short'],
    }
    for _, row in oe_df.iterrows()
]
print(f'  {len(open_ended_rows)} open-ended passages loaded.')

# ── Group concepts by target_word ─────────────────────────────────────────────

groups = {}
for target_word, group_df in df.groupby('target_word'):
    groups[target_word] = group_df.sort_values('entropy').reset_index(drop=True)

n_found = len(groups)
print(f'Found {n_found} unique target words.')
if n_found != N_CONCEPTS:
    print(f'  WARNING: expected {N_CONCEPTS}, found {n_found}. Proceeding.')

for tw, g in groups.items():
    if len(g) < N_ENTROPY_LEVELS:
        print(f'  WARNING: "{tw}" has only {len(g)} entropy variants (need {N_ENTROPY_LEVELS}).')

# ── Assign concepts to 4 fixed groups of 5 ────────────────────────────────────

rng = np.random.default_rng(SEED)
shuffled_keys = rng.permutation(list(groups.keys())).tolist()

# Split as evenly as possible into N_ROLE_GROUPS groups
concept_groups = []
n = len(shuffled_keys)
base_size, remainder = divmod(n, N_ROLE_GROUPS)
start = 0
for i in range(N_ROLE_GROUPS):
    size = base_size + (1 if i < remainder else 0)
    concept_groups.append(shuffled_keys[start:start + size])
    start += size

print('\nConcept-to-group assignment:')
for gi, group in enumerate(concept_groups):
    print(f'  G{gi} ({len(group)} concepts): {group}')

# ── Generate 16 sublists ───────────────────────────────────────────────────────

os.makedirs('trial_lists', exist_ok=True)
print('\nGenerating sublists...')

for cr in range(N_CONDITION_ROTATIONS):
    for er in range(N_ENTROPY_LEVELS):
        sublist_num = cr * N_ENTROPY_LEVELS + er + 1
        rows = []

        for group_idx, target_list in enumerate(concept_groups):
            role = ROLE_SEQUENCE[(group_idx - cr) % N_ROLE_GROUPS]

            for target_word in target_list:
                concept_df = groups[target_word]
                n_variants  = len(concept_df)

                # Spread entropy levels across concepts within the sublist
                global_idx  = shuffled_keys.index(target_word)
                entropy_lvl = (global_idx + er) % min(n_variants, N_ENTROPY_LEVELS)
                row         = concept_df.iloc[entropy_lvl]
                base        = {col: row.get(col, '') for col in SHARED_COLS}

                if role == 'some_masked':
                    rows.append({
                        **base,
                        'condition':             'some_masked',
                        'real_passage':           row['passage_variant'],
                        'unmasked_word_indices':  row['unmasked_word_indices_some'],
                    })
                elif role == 'most_masked':
                    rows.append({
                        **base,
                        'condition':             'most_masked',
                        'real_passage':           row['passage_variant'],
                        'unmasked_word_indices':  row['unmasked_word_indices_most'],
                    })
                else:  # sampling
                    rows.append({
                        **base,
                        'condition':    'sampling',
                        'real_passage': row['passage_variant'],
                        'reveal_order': row['reveal_order'],
                    })

        all_rows = rows + open_ended_rows
        out_df = pd.DataFrame(all_rows)
        out_df.insert(0, 'trial_number', range(1, len(out_df) + 1))
        out_df = out_df[[c for c in COL_ORDER if c in out_df.columns]]

        sublist_file = f'trial_lists/sublist_{sublist_num}.csv'
        out_df.to_csv(sublist_file, index=False)

        n_some     = sum(1 for r in rows if r['condition'] == 'some_masked')
        n_most     = sum(1 for r in rows if r['condition'] == 'most_masked')
        n_sampling = sum(1 for r in rows if r['condition'] == 'sampling')
        print(f'  Sublist {sublist_num:2d} (CR={cr}, ER={er}):  '
              f'{len(all_rows)} trials — {n_some} some_masked, {n_most} most_masked, '
              f'{n_sampling} sampling, {len(open_ended_rows)} open_ended  →  {sublist_file}')

# ── Validation ─────────────────────────────────────────────────────────────────

print('\nValidation:')

CONCEPT_CONDITIONS = ['some_masked', 'most_masked', 'sampling']

# 1. No duplicate target words within any sublist's concept rows
all_ok = True
for sl in range(1, N_SUBLISTS + 1):
    df_sl   = pd.read_csv(f'trial_lists/sublist_{sl}.csv')
    concept_rows = df_sl[df_sl['condition'].isin(CONCEPT_CONDITIONS)]
    targets  = list(concept_rows['target_word'])
    dupes    = [t for t in set(targets) if targets.count(t) > 1]
    if dupes:
        print(f'  FAIL sublist {sl:2d}: duplicate target_words: {dupes}')
        all_ok = False
if all_ok:
    print(f'  OK   no duplicate target_words in any sublist')

# 2. Each sublist has 5 some_masked + 5 most_masked + 10 sampling + N open_ended
n_oe = len(open_ended_rows)
counts_ok = True
for sl in range(1, N_SUBLISTS + 1):
    df_sl      = pd.read_csv(f'trial_lists/sublist_{sl}.csv')
    n_some     = len(df_sl[df_sl['condition'] == 'some_masked'])
    n_most     = len(df_sl[df_sl['condition'] == 'most_masked'])
    n_sampling = len(df_sl[df_sl['condition'] == 'sampling'])
    n_oe_sl    = len(df_sl[df_sl['condition'] == 'open_ended'])
    if n_some != 5 or n_most != 5 or n_sampling != 10 or n_oe_sl != n_oe:
        print(f'  FAIL sublist {sl:2d}: expected 5 some_masked + 5 most_masked + 10 sampling '
              f'+ {n_oe} open_ended, got {n_some} + {n_most} + {n_sampling} + {n_oe_sl}')
        counts_ok = False
if counts_ok:
    print(f'  OK   all {N_SUBLISTS} sublists: 5 some_masked + 5 most_masked + 10 sampling '
          f'+ {n_oe} open_ended')

# 3. Coverage: each (target_word, entropy) cell appears exactly once in some_masked,
#    exactly once in most_masked, and exactly twice in sampling, across all sublists.
coverage = {}
for sl in range(1, N_SUBLISTS + 1):
    df_sl = pd.read_csv(f'trial_lists/sublist_{sl}.csv')
    concept_rows = df_sl[df_sl['condition'].isin(CONCEPT_CONDITIONS)]
    for _, row in concept_rows.iterrows():
        key = (row['target_word'], round(float(row['entropy']), 6), row['condition'])
        coverage[key] = coverage.get(key, 0) + 1

EXPECTED_COUNT = {'some_masked': 1, 'most_masked': 1, 'sampling': 2}
coverage_ok = True
for key, count in coverage.items():
    _, _, condition = key
    if count != EXPECTED_COUNT[condition]:
        print(f'  NOTE cell {key} appears {count}x, expected {EXPECTED_COUNT[condition]}x')
        coverage_ok = False
if coverage_ok:
    print(f'  OK   every (target_word, entropy) cell appears once in some_masked, once in '
          f'most_masked, and twice in sampling')

print('\nDone.')
