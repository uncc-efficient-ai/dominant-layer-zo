"""Plot one calibration condition's activation magnitude and record jumps."""
import argparse
import json
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('report')
p.add_argument('--condition',type=int,default=0)
p.add_argument('--output',required=True)
a = p.parse_args()
d = json.load(open(a.report))['conditions'][a.condition]
rows = d['selection']['layers']
fig, axes = plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
x = [r['layer_idx_zero_based'] for r in rows]
for key in ['abs_mean','abs_max']:
    axes[0].plot(x,[r[key] for r in rows],label=key)
axes[0].set_yscale('log'); axes[0].set_ylabel('Absolute activation'); axes[0].legend()
axes[1].plot(x[1:],[r['record_jump_ratio'] for r in rows[1:]])
axes[1].axhline(d['selection']['jump_threshold'],color='gray',linestyle='--')
axes[1].set_yscale('log'); axes[1].set_ylabel('Record jump ratio')
for ax in axes:
    ax.set_xlabel('Transformer block (zero-based)')
    selected = d['selection']['selected_layer_zero_based']
    if selected is not None: ax.axvline(selected,color='tab:red',linestyle=':')
fig.savefig(a.output,dpi=200)
