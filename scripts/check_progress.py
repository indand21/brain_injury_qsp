"""
check_progress.py
Quick progress checker for brain_injury_bayes_drug.py run.
Usage: python check_progress.py
"""
import os, sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUTDIR = r"C:\Users\indan\OneDrive - aiimsbhubaneswar.edu.in\Brain_Injury_AI_Nitasha"
LOG    = os.path.join(OUTDIR, 'run_bayes_drug.log')

# ── Log tail ──────────────────────────────────────────────────────────────────
if os.path.exists(LOG):
    with open(LOG, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    print("=== LOG (last 30 lines) ===")
    print(''.join(lines[-30:]))
else:
    print("Log file not found — script may not have started yet.\n")

# ── Output file status ────────────────────────────────────────────────────────
FILES = [
    ('icp_summaries_n2000.npy',          'Part 2 — ICP waveforms'),
    ('bayes_ode_parameters_n2000.csv',   'Part 3 — Bayesian MAP params'),
    ('drug_response_features_n2000.csv', 'Part 5/6 — Drug response features'),
    ('extended_model_comparison.csv',    'Part 7 — ML comparison'),
    ('fig20_posterior_parameters.png',   'fig20'),
    ('fig21_bayesian_outcome_heatmap.png','fig21'),
    ('fig22_extended_ml_comparison.png', 'fig22'),
    ('fig23_icp_drug_trajectories.png',  'fig23'),
    ('fig24_osmolality_icp_coupling.png','fig24'),
    ('fig25_drug_response_phenotypes.png','fig25'),
]

print("=== OUTPUT FILES ===")
done = 0
for fname, label in FILES:
    path = os.path.join(OUTDIR, fname)
    if os.path.exists(path):
        kb = os.path.getsize(path) / 1024
        print(f"  [DONE]    {label:42s} {kb:7.1f} KB  ({fname})")
        done += 1
    else:
        print(f"  [pending] {label:42s}          ({fname})")

print(f"\n{done}/{len(FILES)} outputs complete.")
