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

# ── Group by og_passage_seed_number ──────────────────────────────────────────

groups = {}
for seed_num, group_df in df.groupby('og_passage_seed_number'):
    sorted_group = group_df.sort_values('entropy').reset_index(drop=True)
    groups[seed_num] = sorted_group

n_concepts_found = len(groups)
print(f'Found {n_concepts_found} unique og_passage_seed_numbers (target concepts).')

# Warn if any concept has fewer than 4 entropy variants
for seed_num, g in groups.items():
    if len(g) < N_ENTROPY_LEVELS:
        print(f'  WARNING: og_passage_seed_number={seed_num} has only {len(g)} variants '
              f'(need {N_ENTROPY_LEVELS}). Some sublists may reuse entropy levels.')

if n_concepts_found != N_CONCEPTS:
    print(f'\nWARNING: Expected {N_CONCEPTS} concepts but found {n_concepts_found}. '
          f'Proceeding anyway — group sizes will be adjusted.')

# ── Assign concepts to 4 fixed groups ────────────────────────────────────────

concept_keys = list(groups.keys())            # list of og_passage_seed_numbers
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

print('\nConcept-to-group assignment (seed):')
for gi, group in enumerate(concept_groups):
    print(f'  G{gi}: {group}')

# ── 16-sublist generation (condition_rotation x entropy_rotation) ─────────────

print('\nGenerating sublists...')

for condition_rotation in range(N_CONDITIONS):
    for entropy_rotation in range(N_ENTROPY_LEVELS):
        sublist_num = condition_rotation * N_ENTROPY_LEVELS + entropy_rotation + 1
        baseline_rows = []
        sampling_rows = []

        for group_idx, seed_list in enumerate(concept_groups):
            # Which condition does this group get for this condition_rotation?
            condition = CONDITIONS[(group_idx + condition_rotation) % N_CONDITIONS]

            for seed_num in seed_list:
                concept_df = groups[seed_num]
                n_variants = len(concept_df)

                # Entropy level rotates independently of condition rotation.
                # global_concept_idx spreads entropy levels across concepts within
                # the same sublist so no single condition sees all the same level.
                global_concept_idx = shuffled_keys.index(seed_num)
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

# 2. Every passage (og_passage_seed_number x entropy level) appears in every condition exactly once
# Build a dict: seed -> {(entropy_rounded, condition): count}
coverage = {}
for sublist_num in range(1, N_SUBLISTS + 1):
    df_sl = pd.read_csv(f'trial_lists/sublist_{sublist_num}.csv')

    for _, row in df_sl.iterrows():
        seed = row['og_passage_seed_number']
        if row['phase'] == 'baseline':
            key = (round(float(row['entropy']), 6), row['masking_level'])
        else:
            key = (round(float(row['entropy']), 6), row['sampling_order_type'])
        coverage.setdefault(seed, {})
        coverage[seed][key] = coverage[seed].get(key, 0) + 1

coverage_ok = True
for seed, cell_counts in coverage.items():
    if len(cell_counts) != N_SUBLISTS:
        print(f'  FAIL seed {seed}: only {len(cell_counts)} unique (entropy, condition) '
              f'pairings (expected {N_SUBLISTS})')
        coverage_ok = False
    for cell, count in cell_counts.items():
        if count != 1:
            print(f'  FAIL seed {seed}: cell {cell} appears {count} times (expected 1)')
            coverage_ok = False
if coverage_ok:
    print(f'  OK   every passage x condition pairing appears exactly once across all sublists')

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