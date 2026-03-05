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
baseline_sublist_1.csv ... baseline_sublist_4.csv
    columns: target_word, passage_variant, jabber_passage, target_word_position,
             masking_level, unmasked_word_indices, entropy, target_probability,
             target_log_probability, target_pos, og_passage_seed_number

sampling_sublist_1.csv ... sampling_sublist_4.csv
    columns: target_word, passage_variant, jabber_passage, target_word_position,
             sampling_order_type, reveal_order, entropy, target_probability,
             target_log_probability, target_pos, og_passage_seed_number

Design
------
20 target concepts (unique og_passage_seed_numbers), each with 4 entropy variants.
4 conditions: some_masked | most_masked | sampling_optimal | sampling_suboptimal
4 sublists, each containing 20 trials (5 per condition).

Latin square — condition per group per sublist:
  Concepts are divided into 4 fixed groups of 5 (G0..G3).
  In sublist N (0-indexed), group Gk gets condition[(k + N) % 4].

  Sublist 1: G0=some  G1=most  G2=opt   G3=sub
  Sublist 2: G0=most  G1=opt   G2=sub   G3=some
  Sublist 3: G0=opt   G1=sub   G2=some  G3=most
  Sublist 4: G0=sub   G1=some  G2=most  G3=opt

Entropy-level assignment per concept per sublist:
  Variants for each concept are sorted low→high entropy (levels 0..3).
  Concept c, Sublist N (0-indexed) uses entropy level (c + N) % 4.
  This ensures:
    - Within a sublist, different concepts use different entropy levels.
    - Across all 4 sublists, every concept is seen at every entropy level exactly once.
"""

import pandas as pd
import numpy as np
import json
import sys

# ── Configuration ────────────────────────────────────────────────────────────

INPUT_CSV       = 'final_pilot_stimuli.csv'
SEED            = 42          # for reproducible concept→group assignment
N_CONCEPTS      = 20          # expected unique target concepts
N_CONDITIONS    = 4
N_SUBLISTS      = 4
CONCEPTS_PER_CONDITION = 5    # N_CONCEPTS / N_CONDITIONS

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
    if len(g) < N_SUBLISTS:
        print(f'  WARNING: og_passage_seed_number={seed_num} has only {len(g)} variants '
              f'(need {N_SUBLISTS}). Some sublists may reuse entropy levels.')

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

# ── Latin-square sublist generation ──────────────────────────────────────────

print('\nGenerating sublists...')

for sublist_idx in range(N_SUBLISTS):
    sublist_num = sublist_idx + 1
    baseline_rows = []
    sampling_rows = []

    for group_idx, seed_list in enumerate(concept_groups):
        # Which condition does this group get in this sublist?
        condition = CONDITIONS[(group_idx + sublist_idx) % N_CONDITIONS]

        for concept_rank, seed_num in enumerate(seed_list):
            concept_df = groups[seed_num]
            n_variants = len(concept_df)

            # Entropy level: rotate concept_rank + sublist_idx to spread across sublists
            # concept_rank is the index of this concept within its group (0..4)
            # We want a global concept index for the rotation. Use its position
            # in the full shuffled list so that different concepts get different
            # levels within the same sublist.
            global_concept_idx = shuffled_keys.index(seed_num)
            entropy_level = (global_concept_idx + sublist_idx) % min(n_variants, N_SUBLISTS)

            row = concept_df.iloc[entropy_level]

            base = {col: row.get(col, '') for col in SHARED_COLS}

            if condition == 'some_masked':
                baseline_rows.append({
                    **base,
                    'masking_level':         'some',
                    'unmasked_word_indices':  row['unmasked_word_indices_some'],
                })

            elif condition == 'most_masked':
                baseline_rows.append({
                    **base,
                    'masking_level':         'most',
                    'unmasked_word_indices':  row['unmasked_word_indices_most'],
                })

            elif condition == 'sampling_optimal':
                sampling_rows.append({
                    **base,
                    'sampling_order_type': 'optimal',
                    'reveal_order':         row['reveal_order_optimal'],
                })

            elif condition == 'sampling_suboptimal':
                sampling_rows.append({
                    **base,
                    'sampling_order_type': 'suboptimal',
                    'reveal_order':         row['reveal_order_suboptimal'],
                })

    baseline_df = pd.DataFrame(baseline_rows)
    sampling_df = pd.DataFrame(sampling_rows)

    baseline_file = f'baseline_sublist_{sublist_num}.csv'
    sampling_file = f'sampling_sublist_{sublist_num}.csv'

    baseline_df.to_csv(baseline_file, index=False)
    sampling_df.to_csv(sampling_file, index=False)

    print(f'\n  Sublist {sublist_num}:')
    print(f'    {baseline_file}  — {len(baseline_df)} trials '
          f'({(baseline_df["masking_level"] == "some").sum()} some_masked, '
          f'{(baseline_df["masking_level"] == "most").sum()} most_masked)')
    print(f'    {sampling_file}  — {len(sampling_df)} trials '
          f'({(sampling_df["sampling_order_type"] == "optimal").sum()} optimal, '
          f'{(sampling_df["sampling_order_type"] == "suboptimal").sum()} suboptimal)')

# ── Validation ────────────────────────────────────────────────────────────────

print('\nValidation:')

# 1. Each target word appears exactly once per sublist
for sublist_num in range(1, N_SUBLISTS + 1):
    b = pd.read_csv(f'baseline_sublist_{sublist_num}.csv')
    s = pd.read_csv(f'sampling_sublist_{sublist_num}.csv')
    all_targets = list(b['target_word']) + list(s['target_word'])
    dupes = [t for t in set(all_targets) if all_targets.count(t) > 1]
    if dupes:
        print(f'  FAIL sublist {sublist_num}: duplicate target_words: {dupes}')
    else:
        print(f'  OK   sublist {sublist_num}: all {len(all_targets)} target_words unique')

# 2. Across all 4 sublists, each og_passage_seed_number uses each entropy level exactly once
entropy_check = {}  # seed_num -> list of entropy values used across sublists
for sublist_num in range(1, N_SUBLISTS + 1):
    for fname in [f'baseline_sublist_{sublist_num}.csv', f'sampling_sublist_{sublist_num}.csv']:
        sub_df = pd.read_csv(fname)
        for _, row in sub_df.iterrows():
            seed = row['og_passage_seed_number']
            if seed not in entropy_check:
                entropy_check[seed] = []
            entropy_check[seed].append(round(float(row['entropy']), 6))

entropy_ok = True
for seed, entropy_vals in entropy_check.items():
    if len(entropy_vals) != N_SUBLISTS:
        print(f'  FAIL seed {seed}: appeared in {len(entropy_vals)} sublists (expected {N_SUBLISTS})')
        entropy_ok = False
    elif len(set(entropy_vals)) != len(entropy_vals):
        print(f'  FAIL seed {seed}: same entropy level reused across sublists: {entropy_vals}')
        entropy_ok = False
if entropy_ok:
    print(f'  OK   each concept uses a unique entropy level across all 4 sublists')

print('\nDone.')
