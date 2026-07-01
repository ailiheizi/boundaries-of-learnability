"""Generate figures for Boundaries of Learnability paper"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('paper/figures', exist_ok=True)
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# ============ Figure 1: The Dissociation (core result) ============
# Grouped bar: Storage(weight/in-ctx) × Procedure(direct/CoT) × Model scale
fig, ax = plt.subplots(figsize=(9, 5))

models = ['1.5B', '3B', '7B*']
# In-context CoT (success)
ctx_cot = [40, 65, 99]  # scales up
# Weight CoT (failure)
wt_cot = [2, 8, 0]  # stays at floor
# Weight single-hop (memory works)
wt_single = [98, 98, 98]  # high

x = np.arange(len(models))
w = 0.25

bars1 = ax.bar(x - w, wt_single, w, label='Weight: Single-hop recall', color='#90CAF9', edgecolor='#333')
bars2 = ax.bar(x, ctx_cot, w, label='In-context: CoT composition', color='#4CAF50', edgecolor='#333')
bars3 = ax.bar(x + w, wt_cot, w, label='Weight: CoT composition', color='#F44336', edgecolor='#333')

ax.set_xlabel('Model Scale (LoRA fine-tuning)')
ax.set_ylabel('Accuracy (%)')
ax.set_title('The Storage-Reasoning Dissociation\nFacts in Weights: Recalled Perfectly, Composed Never')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='upper left')
ax.set_ylim(0, 110)

# Annotation
ax.annotate('98% recall\nbut 0% composition', xy=(2, 5), xytext=(2.3, 40),
            fontsize=9, color='#D32F2F', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#D32F2F'))

ax.text(0.5, 105, '*7B: 3-hop results (single=98%)', fontsize=8, color='#666')
plt.tight_layout()
plt.savefig('paper/figures/fig1_dissociation.png', bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 1: Dissociation saved')

# ============ Figure 2: RAG Positive Control ============
fig, ax = plt.subplots(figsize=(8, 4.5))

conditions = ['Weight\n(closed-book)', 'In-context\n(direct)', 'In-context\n(CoT)', 'RAG\n(step-by-step)']
hop2 = [0, 27, 99, 100]
hop3 = [0, 5, 100, 100]
hop4 = [0, 0, 100, 100]

x = np.arange(len(conditions))
w = 0.22
ax.bar(x - w, hop2, w, label='2-hop', color='#42A5F5', edgecolor='#333')
ax.bar(x, hop3, w, label='3-hop', color='#FFA726', edgecolor='#333')
ax.bar(x + w, hop4, w, label='4-hop', color='#66BB6A', edgecolor='#333')

ax.set_ylabel('Composition Accuracy (%)')
ax.set_title('Composition by Condition (DeepSeek, n=100/hop)\nRAG Restores What Weights Cannot')
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.legend()
ax.set_ylim(0, 115)

# Highlight the contrast
ax.axhline(y=100, color='#999', linestyle=':', linewidth=0.8)
ax.annotate('Ceiling: 100%', xy=(3, 101), fontsize=8, color='#666')

plt.tight_layout()
plt.savefig('paper/figures/fig2_rag_control.png', bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 2: RAG Control saved')

# ============ Figure 3: Study 2 - Narrative Structure ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: keyword signal vs semantic
labels = ['Keyword\nannotations', 'Semantic\n(no keywords)']
vals = [4.5, 1.05]
colors = ['#4CAF50', '#9E9E9E']
bars = ax1.bar(labels, vals, color=colors, edgecolor='#333', width=0.5)
ax1.axhline(y=1.0, color='#F44336', linestyle='--', linewidth=1.5, label='Baseline (arousal-matched)')
ax1.set_ylabel('Signal ratio (vs baseline)')
ax1.set_title('Structural Position Signal')
ax1.legend()
ax1.set_ylim(0, 5.5)
ax1.text(0, 4.7, '4.5x', ha='center', fontsize=12, fontweight='bold', color='#2E7D32')
ax1.text(1, 1.25, '1.05x (= noise)', ha='center', fontsize=10, color='#666')

# Right: coarse vs fine classification
labels2 = ['Coarse-grained\n(cross-scene)', 'Fine-grained\n(within-scene)']
vals2 = [100, 44]
colors2 = ['#4CAF50', '#F44336']
bars2 = ax2.bar(labels2, vals2, color=colors2, edgecolor='#333', width=0.5)
ax2.axhline(y=50, color='#999', linestyle='--', linewidth=1.5, label='Chance (50%)')
ax2.set_ylabel('Classification Accuracy (%)')
ax2.set_title('Foreshadowing Detection')
ax2.legend()
ax2.set_ylim(0, 115)
ax2.text(0, 103, '100%', ha='center', fontsize=12, fontweight='bold', color='#2E7D32')
ax2.text(1, 47, '44% = chance', ha='center', fontsize=10, color='#D32F2F')

plt.tight_layout()
plt.savefig('paper/figures/fig3_narrative.png', bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 3: Narrative saved')

print('\nAll 3 figures saved to paper/figures/')
