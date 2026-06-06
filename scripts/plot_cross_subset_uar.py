import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

subsets = ['RAVDESS', 'CASIA', 'ESD', 'EMODB', 'IEMOCAP']
baseline = [0.2865, 0.3633, 0.4838, 0.4286, 0.3214]
derf     = [0.7708, 0.7100, 0.6176, 0.7032, 0.5348]
dyt      = [0.7639, 0.6933, 0.6167, 0.6921, 0.5641]

x = np.arange(len(subsets))
width = 0.27

fig, ax = plt.subplots(figsize=(9, 5.2))
b1 = ax.bar(x - width, baseline, width, label='CNN+BiLSTM+Attention', color='#9aa0a6')
b2 = ax.bar(x,         derf,     width, label='Derf (Main)',          color='#1f77b4')
b3 = ax.bar(x + width, dyt,      width, label='DyT (Comparison)',     color='#ff7f0e')

for bars in (b1, b2, b3):
    for b in bars:
        ax.annotate(f'{b.get_height():.3f}',
                    xy=(b.get_x() + b.get_width()/2, b.get_height()),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7.5)

ax.set_ylabel('Test UAR')
ax.set_xlabel('Subset')
ax.set_title('Cross-Subset Test UAR Comparison')
ax.set_xticks(x)
ax.set_xticklabels(subsets)
ax.set_ylim(0, 0.95)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.4)

iemocap_idx = subsets.index('IEMOCAP')
ax.annotate('Conversational corpus\n(main difficulty)',
            xy=(iemocap_idx, 0.5641), xytext=(iemocap_idx - 0.6, 0.85),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

out = Path(__file__).resolve().parent.parent / 'checkpoints' / 'cross_subset_uar_comparison.png'
fig.tight_layout()
fig.savefig(out, dpi=200)
print(f'saved: {out}')
