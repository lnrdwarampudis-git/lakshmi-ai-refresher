"""
=============================================================================
COMPLETE MATPLOTLIB & SEABORN PLOT ENCYCLOPEDIA
Univariate · Bivariate · Multivariate · ML/DL/AI Production Plots
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                              precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Global Style ─────────────────────────────────────────────────────────────
BG    = '#FAFAF8'
BLUE  = '#1A5FA8'; RED   = '#C0392B'; GREEN = '#1A7A4A'
AMBER = '#B5770D'; PUR   = '#6A3D9A'; TEAL  = '#0B7A75'
DARK  = '#1A1A1A'; MID   = '#555555'; LITE  = '#AAAAAA'
PALETTE = [BLUE, RED, GREEN, AMBER, PUR, TEAL, '#E67E22', '#2ECC71']

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#E5E3DC', 'grid.linewidth': 0.5,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'figure.dpi': 130,
})
sns.set_theme(style='whitegrid', palette=PALETTE)

# ── Sample Data ───────────────────────────────────────────────────────────────
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=['SepalL','SepalW','PetalL','PetalW'])
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

n = 300
scores    = np.clip(np.random.normal(72, 14, n), 20, 100)
hours     = np.clip(scores/10 + np.random.normal(0, 1.2, n), 1, 12)
income    = np.random.lognormal(10.5, 0.8, n)
category  = np.random.choice(['A','B','C','D'], n, p=[0.3,0.25,0.25,0.2])
binary_y  = (scores > 72).astype(int)

# ML datasets
X_clf, y_clf = make_classification(n_samples=500, n_features=10, n_informative=6,
                                    n_classes=3, n_clusters_per_class=1, random_state=42)
X_reg, y_reg = make_regression(n_samples=300, n_features=5, noise=20, random_state=42)

def section_header(ax, title, color, subtitle=''):
    ax.axis('off')
    bg = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02", lw=0,
                         facecolor=color+'22', transform=ax.transAxes)
    ax.add_patch(bg)
    ax.text(0.5, 0.65, title, transform=ax.transAxes,
            ha='center', va='center', fontsize=13, fontweight='bold', color=color)
    if subtitle:
        ax.text(0.5, 0.25, subtitle, transform=ax.transAxes,
                ha='center', va='center', fontsize=8.5, color=MID, style='italic')

def add_code(ax, code, color=BLUE):
    ax.text(0.5, -0.18, code, transform=ax.transAxes, ha='center', fontsize=7.5,
            fontfamily='monospace', color=color,
            bbox=dict(boxstyle='round,pad=0.3', fc='#F0F0EB', ec=color, alpha=0.8))

def add_label(ax, text, color=MID):
    ax.text(0.5, 1.02, text, transform=ax.transAxes, ha='center', fontsize=7.5,
            color=color, style='italic')


# =============================================================================
# FIGURE 1 — UNIVARIATE PLOTS (9 plots)
# =============================================================================
fig1 = plt.figure(figsize=(22, 18))
fig1.suptitle('UNIVARIATE PLOTS — One Variable at a Time',
              fontsize=17, fontweight='bold', y=0.985, color=DARK)
gs1 = gridspec.GridSpec(3, 4, figure=fig1, hspace=0.55, wspace=0.38,
                        top=0.955, bottom=0.05, left=0.04, right=0.97)

# ── Header ─────────────────────────────────────────────────────────────────
ax_hdr = fig1.add_subplot(gs1[0, :1])
section_header(ax_hdr, 'UNIVARIATE', BLUE, 'Explore ONE variable\nDistribution · Shape · Spread')

# ── 1. Histogram ──────────────────────────────────────────────────────────
ax = fig1.add_subplot(gs1[0, 1])
ax.hist(scores, bins=28, color=BLUE, alpha=0.70, edgecolor='white', lw=0.4, density=True)
ax.axvline(scores.mean(), color=RED, lw=2, ls='--', label=f'μ={scores.mean():.1f}')
ax.axvline(np.median(scores), color=GREEN, lw=2, ls='-.', label=f'med={np.median(scores):.1f}')
ax.legend(fontsize=7); ax.set_xlabel('Score'); ax.set_ylabel('Density')
ax.set_title('① Histogram\n(frequency distribution)')
add_code(ax, 'plt.hist(data, bins=28, density=True)\nax.axvline(mean, color="red")')

# ── 2. KDE Plot ────────────────────────────────────────────────────────────
ax = fig1.add_subplot(gs1[0, 2])
kde_x = np.linspace(scores.min()-5, scores.max()+5, 300)
kde   = stats.gaussian_kde(scores, bw_method=0.3)
ax.plot(kde_x, kde(kde_x), color=BLUE, lw=2.5)
ax.fill_between(kde_x, kde(kde_x), alpha=0.18, color=BLUE)
ax.set_xlabel('Score'); ax.set_ylabel('Density')
ax.set_title('② KDE Plot\n(smooth distribution estimate)')
add_code(ax, 'sns.kdeplot(data, fill=True, bw_adjust=0.8)')

# ── 3. Histogram + KDE ────────────────────────────────────────────────────
ax = fig1.add_subplot(gs1[0, 3])
sns.histplot(scores, bins=22, kde=True, color=PUR, alpha=0.55, ax=ax)
ax.set_xlabel('Score'); ax.set_title('③ Histogram + KDE\n(combined — most common)')
add_code(ax, 'sns.histplot(data, bins=22, kde=True)')

# ── 4. Box Plot ────────────────────────────────────────────────────────────
ax = fig1.add_subplot(gs1[1, 0])
bp = ax.boxplot(scores, patch_artist=True, widths=0.5, vert=True,
                medianprops=dict(color='white', lw=2.5),
                boxprops=dict(facecolor=BLUE, alpha=0.55),
                flierprops=dict(marker='o', ms=4, color=RED, alpha=0.6))
ax.set_xticklabels(['Score']); ax.set_ylabel('Value')
ax.set_title('④ Box Plot\n(5-number summary + outliers)')
add_code(ax, 'ax.boxplot(data, patch_artist=True)\nsns.boxplot(y=data)')

# ── 5. Violin Plot ─────────────────────────────────────────────────────────
ax = fig1.add_subplot(gs1[1, 1])
parts = ax.violinplot([scores], showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor(GREEN); pc.set_alpha(0.55)
parts['cmeans'].set_color(RED); parts['cmedians'].set_color(AMBER)
ax.set_xticks([1]); ax.set_xticklabels(['Score'])
ax.set_title('⑤ Violin Plot\n(KDE + box combined)')
add_code(ax, 'ax.violinplot(data, showmeans=True)\nsns.violinplot(y=data)')

# ── 6. ECDF ────────────────────────────────────────────────────────────────
ax = fig1.add_subplot(gs1[1, 2])
sorted_s = np.sort(scores)
ecdf_y   = np.arange(1, n+1) / n
ax.step(sorted_s, ecdf_y, color=TEAL, lw=2.2)
ax.fill_between(sorted_s, ecdf_y, step='pre', alpha=0.12, color=TEAL)
for pct in [0.25, 0.50, 0.75]:
    val = np.percentile(scores, pct*100)
    ax.plot([val, val], [0, pct], color=LITE, lw=1, ls='--')
    ax.plot([sorted_s.min(), val], [pct, pct], color=LITE, lw=1, ls='--')
    ax.text(val, pct+0.02, f'Q{int(pct*4)}={val:.0f}', fontsize=7.5, ha='center')
ax.set_xlabel('Score'); ax.set_ylabel('Cumulative Probability')
ax.set_title('⑥ ECDF\n(empirical cumulative dist)')
add_code(ax, 'ax.step(np.sort(x), np.arange(1,n+1)/n)\nsns.ecdfplot(data)')

# ── 7. Q-Q Plot ────────────────────────────────────────────────────────────
ax = fig1.add_subplot(gs1[1, 3])
(osm, osr), (slope, intercept, _) = stats.probplot(scores, dist='norm')
ax.scatter(osm, osr, color=BLUE, s=20, alpha=0.65, edgecolor='none')
lx = np.array([osm.min(), osm.max()])
ax.plot(lx, slope*lx+intercept, color=RED, lw=2)
ax.fill_between(lx, (slope*lx+intercept)-4, (slope*lx+intercept)+4, alpha=0.08, color=RED)
ax.set_xlabel('Theoretical Quantiles'); ax.set_ylabel('Sample Quantiles')
ax.set_title('⑦ Q-Q Plot\n(normality check)')
_, p_sw = stats.shapiro(scores[:50])
ax.text(0.05, 0.92, f'Shapiro p={p_sw:.3f}', transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round', fc='#EAF5EE', ec=GREEN))
add_code(ax, 'stats.probplot(data, plot=ax)\nsns.residplot(x, y)')

# ── 8. Rugplot ─────────────────────────────────────────────────────────────
ax = fig1.add_subplot(gs1[2, 0])
kde2 = stats.gaussian_kde(scores)
xx   = np.linspace(scores.min()-5, scores.max()+5, 300)
ax.plot(xx, kde2(xx), color=PUR, lw=2.5)
ax.fill_between(xx, kde2(xx), alpha=0.15, color=PUR)
sns.rugplot(x=scores, ax=ax, color=PUR, alpha=0.4, height=0.06)
ax.set_xlabel('Score'); ax.set_ylabel('Density')
ax.set_title('⑧ KDE + Rugplot\n(shows individual data points)')
add_code(ax, 'sns.kdeplot(data)\nsns.rugplot(data, height=0.06)')

# ── 9. Bar chart (frequency) ───────────────────────────────────────────────
ax = fig1.add_subplot(gs1[2, 1])
vals, cnts = np.unique(category, return_counts=True)
bars = ax.bar(vals, cnts, color=PALETTE[:4], alpha=0.80, edgecolor='white', width=0.6)
for bar, cnt in zip(bars, cnts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
            f'{cnt}\n({cnt/n*100:.0f}%)', ha='center', fontsize=8, fontweight='bold')
ax.set_xlabel('Category'); ax.set_ylabel('Count')
ax.set_title('⑨ Bar Chart\n(categorical frequency)')
add_code(ax, 'ax.bar(categories, counts)\nsns.countplot(x="col", data=df)')

# ── 10. Pie / Donut ────────────────────────────────────────────────────────
ax = fig1.add_subplot(gs1[2, 2])
wedge_props = dict(width=0.45, edgecolor='white', linewidth=2)
ax.pie(cnts, labels=vals, colors=PALETTE[:4], autopct='%1.1f%%',
       wedgeprops=wedge_props, startangle=90, pctdistance=0.75)
circle = plt.Circle((0,0), 0.55, color=BG)
ax.add_patch(circle)
ax.set_title('⑩ Donut Chart\n(categorical proportion)')
add_code(ax, 'ax.pie(counts, wedgeprops=dict(width=0.45))')

# ── 11. Strip + Swarm ─────────────────────────────────────────────────────
ax = fig1.add_subplot(gs1[2, 3])
# Manual strip plot
jitter = np.random.uniform(-0.2, 0.2, n)
scatter_colors = [PALETTE[['A','B','C','D'].index(c)] for c in category]
ax.scatter(jitter, scores, c=scatter_colors, s=8, alpha=0.45, edgecolor='none')
ax.axhline(scores.mean(), color=RED, lw=2, ls='--', label='Mean')
ax.set_xticks([0]); ax.set_xticklabels(['All Students'])
ax.set_ylabel('Score')
ax.set_title('⑪ Strip Plot\n(individual data points)')
add_code(ax, 'sns.stripplot(y=data, jitter=0.2)\nsns.swarmplot(y=data)')
ax.legend(fontsize=7)

fig1.savefig('/mnt/user-data/outputs/PLOTS_1_Univariate.png', bbox_inches='tight', dpi=130)
print('✓ Figure 1 — Univariate Plots saved')
plt.close(fig1)


# =============================================================================
# FIGURE 2 — BIVARIATE PLOTS (12 plots)
# =============================================================================
fig2 = plt.figure(figsize=(22, 20))
fig2.suptitle('BIVARIATE PLOTS — Relationships Between Two Variables',
              fontsize=17, fontweight='bold', y=0.985, color=DARK)
gs2 = gridspec.GridSpec(3, 4, figure=fig2, hspace=0.60, wspace=0.38,
                        top=0.955, bottom=0.05, left=0.04, right=0.97)

# ── 1. Scatter plot ────────────────────────────────────────────────────────
ax = fig2.add_subplot(gs2[0, 0])
ax.scatter(hours, scores, color=BLUE, s=18, alpha=0.55, edgecolor='none')
m, b, r, p, _ = stats.linregress(hours, scores)
xl = np.linspace(hours.min(), hours.max(), 100)
ax.plot(xl, m*xl+b, color=RED, lw=2.2)
ax.set_xlabel('Hours'); ax.set_ylabel('Score')
ax.set_title(f'① Scatter + Regression\nr={r:.3f}  R²={r**2:.3f}')
add_code(ax, 'ax.scatter(x, y)\nsns.regplot(x="x", y="y", data=df)')

# ── 2. Scatter + confidence band ───────────────────────────────────────────
ax = fig2.add_subplot(gs2[0, 1])
sns.regplot(x=hours, y=scores, ax=ax, scatter_kws={'s':12,'alpha':0.45,'color':BLUE},
            line_kws={'color':RED,'lw':2}, ci=95, color=BLUE)
ax.set_xlabel('Hours'); ax.set_ylabel('Score')
ax.set_title('② sns.regplot\n(95% confidence band)')
add_code(ax, 'sns.regplot(x, y, ci=95)\n# shades 95% CI automatically')

# ── 3. Joint plot (scatter + marginals) ────────────────────────────────────
ax = fig2.add_subplot(gs2[0, 2])
h2 = hours[:100]; s2 = scores[:100]  # subset for clarity
ax.scatter(h2, s2, color=TEAL, s=25, alpha=0.6, edgecolor='none')
# Marginal histograms (manual)
ax_top  = ax.inset_axes([0, 1.04, 1, 0.22])
ax_rght = ax.inset_axes([1.04, 0, 0.22, 1])
ax_top.hist(h2, bins=18, color=TEAL, alpha=0.6, edgecolor='none')
ax_rght.hist(s2, bins=18, color=TEAL, alpha=0.6, edgecolor='none',
             orientation='horizontal')
ax_top.axis('off'); ax_rght.axis('off')
ax.set_xlabel('Hours'); ax.set_ylabel('Score')
ax.set_title('③ Joint Plot (marginals)\n(bivariate + univariate)')
add_code(ax, 'sns.jointplot(x="h", y="s", data=df,\n  kind="scatter")')

# ── 4. Hex bin ─────────────────────────────────────────────────────────────
ax = fig2.add_subplot(gs2[0, 3])
hb = ax.hexbin(hours, scores, gridsize=18, cmap='YlOrRd', mincnt=1)
plt.colorbar(hb, ax=ax, shrink=0.8, label='Count')
ax.set_xlabel('Hours'); ax.set_ylabel('Score')
ax.set_title('④ Hexbin Plot\n(dense scatter alternative)')
add_code(ax, 'ax.hexbin(x, y, gridsize=18, cmap="YlOrRd")')

# ── 5. Box by group ────────────────────────────────────────────────────────
ax = fig2.add_subplot(gs2[1, 0])
grp_data = [scores[category==c] for c in ['A','B','C','D']]
bp = ax.boxplot(grp_data, patch_artist=True, widths=0.5,
                medianprops=dict(color='white', lw=2.5))
for patch, color in zip(bp['boxes'], PALETTE[:4]):
    patch.set_facecolor(color); patch.set_alpha(0.60)
ax.set_xticklabels(['A','B','C','D'])
ax.set_xlabel('Category'); ax.set_ylabel('Score')
ax.set_title('⑤ Box Plot by Group\n(numeric vs categorical)')
add_code(ax, 'sns.boxplot(x="cat", y="num", data=df)')

# ── 6. Violin by group ─────────────────────────────────────────────────────
ax = fig2.add_subplot(gs2[1, 1])
sns.violinplot(x=category, y=scores, ax=ax,
               palette=PALETTE[:4], alpha=0.65, inner='box')
ax.set_xlabel('Category'); ax.set_ylabel('Score')
ax.set_title('⑥ Violin Plot by Group\n(distribution shape per group)')
add_code(ax, 'sns.violinplot(x="cat", y="num",\n  data=df, inner="box")')

# ── 7. Bar + error bars ────────────────────────────────────────────────────
ax = fig2.add_subplot(gs2[1, 2])
means  = [scores[category==c].mean() for c in ['A','B','C','D']]
sems   = [scores[category==c].std()/np.sqrt((category==c).sum()) for c in ['A','B','C','D']]
bars = ax.bar(['A','B','C','D'], means, yerr=sems, color=PALETTE[:4],
              alpha=0.75, edgecolor='white', capsize=6, error_kw={'lw':2})
ax.set_xlabel('Category'); ax.set_ylabel('Mean Score ± SEM')
ax.set_title('⑦ Bar + Error Bars\n(mean ± standard error)')
add_code(ax, 'ax.bar(cats, means, yerr=sems, capsize=6)\nsns.barplot(x, y, data=df)')

# ── 8. Point plot ──────────────────────────────────────────────────────────
ax = fig2.add_subplot(gs2[1, 3])
sns.pointplot(x=category, y=scores, ax=ax, color=BLUE,
              capsize=0.15, err_kws={'linewidth':1.5}, linestyles='--')
ax.set_xlabel('Category'); ax.set_ylabel('Mean Score')
ax.set_title('⑧ Point Plot\n(means + CI, shows trend)')
add_code(ax, 'sns.pointplot(x="cat", y="num",\n  data=df, capsize=0.1)')

# ── 9. Correlation heatmap ─────────────────────────────────────────────────
ax = fig2.add_subplot(gs2[2, 0])
corr = iris_df[['SepalL','SepalW','PetalL','PetalW']].corr()
sns.heatmap(corr, ax=ax, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.6, square=True,
            cbar_kws={'shrink':0.85}, annot_kws={'size':8.5})
ax.set_title('⑨ Correlation Heatmap\n(all numeric pairs)')
add_code(ax, 'sns.heatmap(df.corr(), annot=True,\n  cmap="RdBu_r", center=0)')

# ── 10. Line plot ─────────────────────────────────────────────────────────
ax = fig2.add_subplot(gs2[2, 1])
months = np.arange(1, 13)
loss   = 1.5 * np.exp(-0.3*months) + np.random.normal(0, 0.04, 12)
val_l  = 1.6 * np.exp(-0.25*months) + np.random.normal(0, 0.05, 12)
ax.plot(months, loss,  color=BLUE, lw=2.2, marker='o', ms=5, label='Train loss')
ax.plot(months, val_l, color=RED,  lw=2.2, marker='s', ms=5, ls='--', label='Val loss')
ax.fill_between(months, loss, val_l, alpha=0.10, color=AMBER)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('⑩ Line Plot\n(trend over time / epochs)')
ax.legend(fontsize=8)
add_code(ax, 'ax.plot(x, y, marker="o")\nsns.lineplot(x, y, hue="group")')

# ── 11. 2D KDE ────────────────────────────────────────────────────────────
ax = fig2.add_subplot(gs2[2, 2])
sns.kdeplot(x=hours, y=scores, ax=ax, fill=True, cmap='Blues',
            levels=10, thresh=0.05)
ax.scatter(hours, scores, s=6, alpha=0.20, color=DARK, edgecolor='none')
ax.set_xlabel('Hours'); ax.set_ylabel('Score')
ax.set_title('⑪ 2D KDE Contour\n(bivariate density)')
add_code(ax, 'sns.kdeplot(x=x, y=y, fill=True,\n  cmap="Blues", levels=10)')

# ── 12. Strip + Box overlay ───────────────────────────────────────────────
ax = fig2.add_subplot(gs2[2, 3])
cats = ['A','B','C','D']
positions = range(len(cats))
for i, (cat, pos) in enumerate(zip(cats, positions)):
    data_c = scores[category==cat]
    jitter  = np.random.uniform(-0.18, 0.18, len(data_c))
    ax.scatter(np.ones(len(data_c))*pos + jitter, data_c,
               color=PALETTE[i], s=10, alpha=0.40, edgecolor='none')
ax.boxplot([scores[category==c] for c in cats], positions=list(positions),
           patch_artist=True, widths=0.25,
           medianprops=dict(color='white', lw=2),
           boxprops=dict(facecolor='none', edgecolor=DARK))
ax.set_xticks(list(positions)); ax.set_xticklabels(cats)
ax.set_xlabel('Category'); ax.set_ylabel('Score')
ax.set_title('⑫ Strip + Box Overlay\n(raw data + summary)')
add_code(ax, 'sns.stripplot(x, y) + sns.boxplot(x, y)\n# or sns.violinplot(inner="box")')

fig2.savefig('/mnt/user-data/outputs/PLOTS_2_Bivariate.png', bbox_inches='tight', dpi=130)
print('✓ Figure 2 — Bivariate Plots saved')
plt.close(fig2)


# =============================================================================
# FIGURE 3 — MULTIVARIATE PLOTS (12 plots)
# =============================================================================
fig3 = plt.figure(figsize=(22, 20))
fig3.suptitle('MULTIVARIATE PLOTS — Three or More Variables',
              fontsize=17, fontweight='bold', y=0.985, color=DARK)
gs3 = gridspec.GridSpec(3, 4, figure=fig3, hspace=0.60, wspace=0.38,
                        top=0.955, bottom=0.05, left=0.04, right=0.97)

# ── 1. Pair plot (manual 3×3) ─────────────────────────────────────────────
ax = fig3.add_subplot(gs3[0, 0])
for i, sp in enumerate(iris.target_names):
    mask = iris_df['species'] == sp
    ax.scatter(iris_df.loc[mask,'SepalL'], iris_df.loc[mask,'PetalL'],
               color=PALETTE[i], s=20, alpha=0.65, edgecolor='none', label=sp)
ax.set_xlabel('Sepal Length'); ax.set_ylabel('Petal Length')
ax.set_title('① Pair Plot (subset)\n(all feature pairs by species)')
ax.legend(fontsize=7)
add_code(ax, 'sns.pairplot(df, hue="species",\n  diag_kind="kde")')

# ── 2. Full heatmap / correlation matrix ──────────────────────────────────
ax = fig3.add_subplot(gs3[0, 1])
full_corr = iris_df[['SepalL','SepalW','PetalL','PetalW']].corr()
mask = np.triu(np.ones_like(full_corr, dtype=bool), k=1)
sns.heatmap(full_corr, ax=ax, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.8, square=True,
            cbar_kws={'shrink':0.8}, annot_kws={'size':9})
ax.set_title('② Correlation Matrix\n(multivariate heatmap)')
add_code(ax, 'sns.heatmap(df.corr(), annot=True,\n  cmap="RdBu_r", square=True)')

# ── 3. Bubble chart (4 vars) ──────────────────────────────────────────────
ax = fig3.add_subplot(gs3[0, 2])
sample_idx = np.random.choice(n, 120, replace=False)
sc = ax.scatter(hours[sample_idx], scores[sample_idx],
                s=income[sample_idx]/500,
                c=scores[sample_idx], cmap='viridis',
                alpha=0.60, edgecolor='none')
plt.colorbar(sc, ax=ax, shrink=0.85, label='Score (color)')
ax.set_xlabel('Hours (x)'); ax.set_ylabel('Score (y)')
ax.set_title('③ Bubble Chart\n(x, y, size=income, color=score)')
add_code(ax, 'ax.scatter(x, y, s=size_var,\n  c=color_var, cmap="viridis")')

# ── 4. Facet grid ─────────────────────────────────────────────────────────
ax = fig3.add_subplot(gs3[0, 3])
colors_sp = [BLUE, GREEN, RED]
for i, sp in enumerate(iris.target_names):
    data = iris_df.loc[iris_df['species']==sp, 'SepalL'].values
    kde_x = np.linspace(4, 8.5, 200)
    kde_y = stats.gaussian_kde(data)(kde_x)
    ax.plot(kde_x, kde_y, color=colors_sp[i], lw=2.2, label=sp)
    ax.fill_between(kde_x, kde_y, alpha=0.15, color=colors_sp[i])
ax.set_xlabel('Sepal Length'); ax.set_ylabel('Density')
ax.set_title('④ Facet / Multi-KDE\n(one variable, multiple groups)')
ax.legend(fontsize=7.5)
add_code(ax, 'g = sns.FacetGrid(df, col="species")\ng.map(sns.histplot, "SepalL")')

# ── 5. PCA scatter ────────────────────────────────────────────────────────
ax = fig3.add_subplot(gs3[1, 0])
X_sc  = StandardScaler().fit_transform(iris.data)
pca2  = PCA(n_components=2)
X_pca = pca2.fit_transform(X_sc)
for i, sp in enumerate(iris.target_names):
    mask = iris.target == i
    ax.scatter(X_pca[mask,0], X_pca[mask,1], color=PALETTE[i],
               s=28, alpha=0.72, edgecolor='none', label=sp)
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.0f}%)')
ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.0f}%)')
ax.set_title('⑤ PCA Scatter\n(dimensionality reduction visual)')
ax.legend(fontsize=7.5)
add_code(ax, 'pca = PCA(n_components=2).fit_transform(X)\nax.scatter(pca[:,0], pca[:,1], c=y)')

# ── 6. Cluster map ────────────────────────────────────────────────────────
ax = fig3.add_subplot(gs3[1, 1])
km3 = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_sc)
for k in range(3):
    mask = km3.labels_ == k
    ax.scatter(X_pca[mask,0], X_pca[mask,1], color=PALETTE[k],
               s=22, alpha=0.65, edgecolor='none', label=f'Cluster {k+1}')
cents = pca2.transform(km3.cluster_centers_)
ax.scatter(cents[:,0], cents[:,1], s=200, marker='*', color=DARK, zorder=6, label='Centroids')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('⑥ Cluster Scatter\n(K-Means result in PCA space)')
ax.legend(fontsize=7)
add_code(ax, 'km = KMeans(k=3).fit(X)\nax.scatter(X[:,0],X[:,1], c=km.labels_)')

# ── 7. 3D scatter ─────────────────────────────────────────────────────────
ax7 = fig3.add_subplot(gs3[1, 2], projection='3d')
for i, sp in enumerate(iris.target_names):
    mask = iris_df['species'] == sp
    ax7.scatter(iris_df.loc[mask,'SepalL'],
                iris_df.loc[mask,'PetalL'],
                iris_df.loc[mask,'PetalW'],
                color=PALETTE[i], s=22, alpha=0.65, label=sp)
ax7.set_xlabel('SepalL', fontsize=7); ax7.set_ylabel('PetalL', fontsize=7)
ax7.set_zlabel('PetalW', fontsize=7)
ax7.set_title('⑦ 3D Scatter\n(3 numeric vars)', pad=8)
ax7.legend(fontsize=7, loc='upper left')
ax7.text2D(0.5, -0.06, '3D scatter: fig.add_subplot(projection=3d)', transform=ax7.transAxes, ha='center', fontsize=7.5, color=BLUE)

# ── 8. Radar / Spider chart ────────────────────────────────────────────────
ax = fig3.add_subplot(gs3[1, 3], polar=True)
categories_r = ['SepalL','SepalW','PetalL','PetalW']
N = len(categories_r)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]
for i, sp in enumerate(iris.target_names):
    means_r = iris_df.loc[iris_df['species']==sp, categories_r].mean().values
    means_r = (means_r - iris_df[categories_r].min()) / (iris_df[categories_r].max() - iris_df[categories_r].min())
    vals = means_r.tolist() + means_r[:1].tolist()
    ax.plot(angles, vals, color=PALETTE[i], lw=2, label=sp)
    ax.fill(angles, vals, color=PALETTE[i], alpha=0.12)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories_r, fontsize=8)
ax.set_title('⑧ Radar Chart\n(multi-feature profiles)', pad=18)
ax.legend(fontsize=7, loc='lower right')
add_code(ax, 'ax = fig.add_subplot(polar=True)\nax.plot(angles, values)')

# ── 9. Parallel coordinates ───────────────────────────────────────────────
ax = fig3.add_subplot(gs3[2, 0])
cols = ['SepalL','SepalW','PetalL','PetalW']
df_norm = iris_df.copy()
for c in cols:
    df_norm[c] = (iris_df[c] - iris_df[c].min()) / (iris_df[c].max() - iris_df[c].min())
for i, sp in enumerate(iris.target_names):
    subset = df_norm[df_norm['species']==sp]
    for _, row in subset.iterrows():
        ax.plot(range(len(cols)), [row[c] for c in cols],
                color=PALETTE[i], alpha=0.12, lw=1)
# Mean lines
for i, sp in enumerate(iris.target_names):
    subset = df_norm[df_norm['species']==sp]
    ax.plot(range(len(cols)), [subset[c].mean() for c in cols],
            color=PALETTE[i], lw=3, label=sp)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(['SepalL','SepalW','PetalL','PetalW'])
ax.set_ylabel('Normalized Value')
ax.set_title('⑨ Parallel Coordinates\n(all features per sample)')
ax.legend(fontsize=7)
add_code(ax, 'from pandas.plotting import parallel_coordinates\nparallel_coordinates(df, "class")')

# ── 10. Grouped bar chart ─────────────────────────────────────────────────
ax = fig3.add_subplot(gs3[2, 1])
species_list = iris.target_names
feature_list = ['SepalL','SepalW','PetalL','PetalW']
x = np.arange(len(feature_list)); width = 0.27
for i, sp in enumerate(species_list):
    means_gb = iris_df.loc[iris_df['species']==sp, feature_list].mean()
    ax.bar(x + i*width, means_gb, width, label=sp,
           color=PALETTE[i], alpha=0.80, edgecolor='white')
ax.set_xticks(x + width); ax.set_xticklabels(feature_list, fontsize=8)
ax.set_ylabel('Mean (cm)')
ax.set_title('⑩ Grouped Bar Chart\n(multi-category comparison)')
ax.legend(fontsize=7.5)
add_code(ax, 'ax.bar(x+i*width, means, width)\n# or sns.barplot(x, y, hue="group")')

# ── 11. Stacked area ──────────────────────────────────────────────────────
ax = fig3.add_subplot(gs3[2, 2])
t = np.arange(0, 12)
a = np.clip(np.cumsum(np.random.normal(3, 1, 12)), 5, 50)
b_area = np.clip(np.cumsum(np.random.normal(2, 1, 12)), 3, 40)
c_area = np.clip(np.cumsum(np.random.normal(1.5, 1, 12)), 2, 30)
ax.stackplot(t, a, b_area, c_area, labels=['Product A','Product B','Product C'],
             colors=[BLUE, GREEN, AMBER], alpha=0.70)
ax.set_xlabel('Month'); ax.set_ylabel('Cumulative Sales')
ax.set_title('⑪ Stacked Area Chart\n(part-to-whole over time)')
ax.legend(fontsize=7.5, loc='upper left')
add_code(ax, 'ax.stackplot(t, y1, y2, y3,\n  labels=names, alpha=0.7)')

# ── 12. Heatmap (time × category) ────────────────────────────────────────
ax = fig3.add_subplot(gs3[2, 3])
hm_data = np.random.randint(10, 100, (5, 7))
hm_df = pd.DataFrame(hm_data,
                      index=['Mon','Tue','Wed','Thu','Fri'],
                      columns=['00h','04h','08h','12h','16h','20h','23h'])
sns.heatmap(hm_df, ax=ax, annot=True, fmt='d', cmap='YlOrRd',
            linewidths=0.5, cbar_kws={'shrink':0.8}, annot_kws={'size':8})
ax.set_title('⑫ Heatmap (time × category)\n(2D frequency / intensity)')
add_code(ax, 'sns.heatmap(pivot_df, annot=True,\n  fmt="d", cmap="YlOrRd")')

fig3.savefig('/mnt/user-data/outputs/PLOTS_3_Multivariate.png', bbox_inches='tight', dpi=130)
print('✓ Figure 3 — Multivariate Plots saved')
plt.close(fig3)


# =============================================================================
# FIGURE 4 — ML / DL / AI PRODUCTION PLOTS (Part A)
# =============================================================================
fig4 = plt.figure(figsize=(22, 20))
fig4.suptitle('ML / DL / AI PRODUCTION PLOTS — Part A: Model Evaluation',
              fontsize=17, fontweight='bold', y=0.985, color=DARK)
gs4 = gridspec.GridSpec(3, 4, figure=fig4, hspace=0.60, wspace=0.38,
                        top=0.955, bottom=0.05, left=0.04, right=0.97)

# Build classifiers
X_tr = X_clf[:400]; y_tr = y_clf[:400]
X_te = X_clf[400:]; y_te = y_clf[400:]
rf  = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
lr  = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr, y_tr)

# ── 1. Confusion Matrix ────────────────────────────────────────────────────
ax = fig4.add_subplot(gs4[0, 0])
cm = confusion_matrix(y_te, rf.predict(X_te))
sns.heatmap(cm, ax=ax, annot=True, fmt='d', cmap='Blues',
            linewidths=0.5, cbar_kws={'shrink':0.8}, annot_kws={'size':12})
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('① Confusion Matrix\n(classification errors per class)')
add_code(ax, 'cm = confusion_matrix(y_true, y_pred)\nsns.heatmap(cm, annot=True, fmt="d")')

# ── 2. Normalized Confusion Matrix ────────────────────────────────────────
ax = fig4.add_subplot(gs4[0, 1])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, ax=ax, annot=True, fmt='.2f', cmap='Greens',
            linewidths=0.5, vmin=0, vmax=1, cbar_kws={'shrink':0.8},
            annot_kws={'size':11})
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('② Normalized Confusion Matrix\n(per-class recall on diagonal)')
add_code(ax, 'cm_norm = cm / cm.sum(axis=1, keepdims=True)\nsns.heatmap(cm_norm, fmt=".2f")')

# ── 3. ROC Curve ──────────────────────────────────────────────────────────
ax = fig4.add_subplot(gs4[0, 2])
y_te_bin = label_binarize(y_te, classes=[0,1,2])
y_prob   = rf.predict_proba(X_te)
for i, color in enumerate([BLUE, GREEN, RED]):
    fpr, tpr, _ = roc_curve(y_te_bin[:,i], y_prob[:,i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'Class {i} (AUC={roc_auc:.3f})')
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random')
ax.fill_between([0,1],[0,1],[1,1], alpha=0.04, color=LITE)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('③ ROC Curve\n(AUC per class — higher=better)')
ax.legend(fontsize=7.5)
add_code(ax, 'fpr, tpr, _ = roc_curve(y_bin, y_prob)\nax.plot(fpr, tpr, label=f"AUC={auc:.3f}")')

# ── 4. Precision-Recall Curve ─────────────────────────────────────────────
ax = fig4.add_subplot(gs4[0, 3])
for i, color in enumerate([BLUE, GREEN, RED]):
    prec, rec, _ = precision_recall_curve(y_te_bin[:,i], y_prob[:,i])
    ap = average_precision_score(y_te_bin[:,i], y_prob[:,i])
    ax.plot(rec, prec, color=color, lw=2, label=f'Class {i} (AP={ap:.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('④ Precision-Recall Curve\n(better than ROC for imbalanced)')
ax.legend(fontsize=7.5)
add_code(ax, 'prec, rec, _ = precision_recall_curve(y, prob)\nax.plot(rec, prec)')

# ── 5. Training / Validation Loss ─────────────────────────────────────────
ax = fig4.add_subplot(gs4[1, 0])
epochs = np.arange(1, 51)
t_loss = 2.5*np.exp(-0.08*epochs) + 0.15 + np.random.normal(0, 0.02, 50)
v_loss = 2.7*np.exp(-0.065*epochs) + 0.25 + np.random.normal(0, 0.025, 50)
t_acc  = 1 - 0.7*np.exp(-0.07*epochs) + np.random.normal(0, 0.01, 50)
v_acc  = 1 - 0.75*np.exp(-0.06*epochs) + np.random.normal(0, 0.012, 50)
ax.plot(epochs, t_loss, color=BLUE, lw=2, label='Train loss')
ax.plot(epochs, v_loss, color=RED, lw=2, ls='--', label='Val loss')
ax.axvline(35, color=AMBER, lw=1.5, ls=':', label='Early stop')
ax.fill_between(epochs, t_loss, v_loss, alpha=0.10, color=AMBER,
                label='Generalization gap')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('⑤ Training / Val Loss Curve\n(overfitting detection)')
ax.legend(fontsize=7.5)
add_code(ax, 'ax.plot(epochs, train_loss, label="Train")\nax.plot(epochs, val_loss, ls="--", label="Val")')

# ── 6. Learning Curve ─────────────────────────────────────────────────────
ax = fig4.add_subplot(gs4[1, 1])
train_sizes, train_scores, val_scores = learning_curve(
    rf, X_clf, y_clf, train_sizes=np.linspace(0.1,1.0,8), cv=5, n_jobs=-1)
train_mean = train_scores.mean(axis=1); train_std = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1);   val_std   = val_scores.std(axis=1)
ax.plot(train_sizes, train_mean, color=BLUE, lw=2, marker='o', ms=5, label='Train score')
ax.plot(train_sizes, val_mean,   color=RED,  lw=2, marker='s', ms=5, ls='--', label='CV score')
ax.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.12, color=BLUE)
ax.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.12, color=RED)
ax.set_xlabel('Training Set Size'); ax.set_ylabel('Accuracy')
ax.set_title('⑥ Learning Curve\n(bias-variance diagnosis)')
ax.legend(fontsize=7.5)
add_code(ax, 'from sklearn.model_selection import learning_curve\nlearning_curve(model, X, y, cv=5)')

# ── 7. Feature Importance ─────────────────────────────────────────────────
ax = fig4.add_subplot(gs4[1, 2])
importances = rf.feature_importances_
feat_names  = [f'Feature {i+1}' for i in range(10)]
idx = np.argsort(importances)[::-1]
colors_imp = [BLUE if i<3 else LITE for i in range(10)]
ax.barh([feat_names[i] for i in idx[::-1]],
         importances[idx[::-1]],
         color=[colors_imp[r] for r in range(10)],
         alpha=0.80, edgecolor='white')
ax.axvline(importances.mean(), color=RED, lw=1.5, ls='--', label='Mean importance')
ax.set_xlabel('Importance Score'); ax.set_title('⑦ Feature Importance\n(Random Forest Gini)')
ax.legend(fontsize=7.5)
add_code(ax, 'imps = rf.feature_importances_\nax.barh(feat_names, imps[sorted_idx])')

# ── 8. SHAP-style waterfall (manual) ──────────────────────────────────────
ax = fig4.add_subplot(gs4[1, 3])
shap_feats = ['Feature 3','Feature 1','Feature 6','Feature 2','Feature 8',
              'Feature 5','Feature 7','Feature 4']
shap_vals  = [0.18, 0.14, 0.09, -0.07, 0.05, -0.04, 0.03, -0.02]
colors_shap = [GREEN if v > 0 else RED for v in shap_vals]
ax.barh(shap_feats, shap_vals, color=colors_shap, alpha=0.80, edgecolor='white')
ax.axvline(0, color=DARK, lw=1)
for i, v in enumerate(shap_vals):
    ax.text(v + (0.005 if v>0 else -0.005), i,
            f'{v:+.3f}', va='center', fontsize=8, fontweight='bold',
            ha='left' if v>0 else 'right')
ax.set_xlabel('SHAP Value (impact on output)')
ax.set_title('⑧ SHAP Feature Impact\n(model explainability — production)')
add_code(ax, 'import shap\nexplainer = shap.TreeExplainer(model)\nshap.plots.waterfall(shap_values[0])')

# ── 9. Cross-validation box plot ──────────────────────────────────────────
ax = fig4.add_subplot(gs4[2, 0])
models = {'LR': lr, 'RF': rf}
model_names = list(models.keys())
cv_results = [cross_val_score(m, X_clf, y_clf, cv=10) for m in models.values()]
bp = ax.boxplot(cv_results, patch_artist=True, widths=0.5,
                medianprops=dict(color='white', lw=2.5))
for patch, color in zip(bp['boxes'], [BLUE, GREEN]):
    patch.set_facecolor(color); patch.set_alpha(0.65)
for i, scores_cv in enumerate(cv_results):
    ax.scatter(np.ones(len(scores_cv))*(i+1) + np.random.uniform(-0.1,0.1,len(scores_cv)),
               scores_cv, color=PALETTE[i], s=15, alpha=0.5, edgecolor='none')
ax.set_xticks([1,2]); ax.set_xticklabels(model_names)
ax.set_ylabel('CV Accuracy')
ax.set_title('⑨ Cross-Validation Box Plot\n(model comparison, k=10)')
add_code(ax, 'scores = cross_val_score(model, X, y, cv=10)\nax.boxplot(scores)')

# ── 10. Calibration curve ─────────────────────────────────────────────────
ax = fig4.add_subplot(gs4[2, 1])
from sklearn.calibration import calibration_curve
y_bin_single = (y_clf == 0).astype(int)
prob_rf = rf.predict_proba(X_clf)[:,0]
prob_lr = lr.predict_proba(X_clf)[:,0]
frac_pos_rf, mean_pred_rf = calibration_curve(y_bin_single, prob_rf, n_bins=10)
frac_pos_lr, mean_pred_lr = calibration_curve(y_bin_single, prob_lr, n_bins=10)
ax.plot([0,1],[0,1], 'k--', lw=1.5, label='Perfect calibration')
ax.plot(mean_pred_rf, frac_pos_rf, color=BLUE, lw=2, marker='o', ms=6, label='Random Forest')
ax.plot(mean_pred_lr, frac_pos_lr, color=RED, lw=2, marker='s', ms=6, label='Logistic Reg')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('⑩ Calibration Curve\n(probability reliability)')
ax.legend(fontsize=7.5)
add_code(ax, 'from sklearn.calibration import calibration_curve\nfrac, mean = calibration_curve(y, prob)')

# ── 11. Residual plot (regression) ────────────────────────────────────────
ax = fig4.add_subplot(gs4[2, 2])
reg_model = LinearRegression().fit(X_reg, y_reg)
y_pred_r  = reg_model.predict(X_reg)
resids    = y_reg - y_pred_r
ax.scatter(y_pred_r, resids, color=BLUE, s=12, alpha=0.50, edgecolor='none')
ax.axhline(0, color=RED, lw=2, ls='--')
# Local trend
from scipy.ndimage import uniform_filter1d
sort_idx = np.argsort(y_pred_r)
smooth_resid = uniform_filter1d(resids[sort_idx], size=30)
ax.plot(y_pred_r[sort_idx], smooth_resid, color=AMBER, lw=2, label='Trend (should be flat)')
ax.set_xlabel('Fitted Values'); ax.set_ylabel('Residuals')
ax.set_title('⑪ Residual Plot\n(regression diagnostic)')
ax.legend(fontsize=7.5)
add_code(ax, 'resids = y_true - y_pred\nax.scatter(y_pred, resids)\nax.axhline(0, ls="--")')

# ── 12. Actual vs Predicted ────────────────────────────────────────────────
ax = fig4.add_subplot(gs4[2, 3])
ax.scatter(y_reg, y_pred_r, color=BLUE, s=12, alpha=0.50, edgecolor='none')
mn_v = min(y_reg.min(), y_pred_r.min()); mx_v = max(y_reg.max(), y_pred_r.max())
ax.plot([mn_v,mx_v],[mn_v,mx_v], color=RED, lw=2, ls='--', label='y=ŷ (perfect)')
r2_val = reg_model.score(X_reg, y_reg)
ax.text(0.05, 0.92, f'R² = {r2_val:.4f}', transform=ax.transAxes, fontsize=9,
        fontweight='bold', bbox=dict(boxstyle='round', fc='#EAF0F8', ec=BLUE))
ax.set_xlabel('Actual y'); ax.set_ylabel('Predicted ŷ')
ax.set_title('⑫ Actual vs Predicted\n(regression quality check)')
ax.legend(fontsize=7.5)
add_code(ax, 'ax.scatter(y_true, y_pred)\nax.plot([mn,mx],[mn,mx], ls="--")')

fig4.savefig('/mnt/user-data/outputs/PLOTS_4_ML_Evaluation.png', bbox_inches='tight', dpi=130)
print('✓ Figure 4 — ML Evaluation Plots saved')
plt.close(fig4)


# =============================================================================
# FIGURE 5 — ML / DL / AI PRODUCTION PLOTS (Part B)
# =============================================================================
fig5 = plt.figure(figsize=(22, 20))
fig5.suptitle('ML / DL / AI PRODUCTION PLOTS — Part B: Deep Learning, NLP, CV, Time Series',
              fontsize=17, fontweight='bold', y=0.985, color=DARK)
gs5 = gridspec.GridSpec(3, 4, figure=fig5, hspace=0.60, wspace=0.38,
                        top=0.955, bottom=0.05, left=0.04, right=0.97)

# ── 1. Loss + Accuracy dual axis ──────────────────────────────────────────
ax = fig5.add_subplot(gs5[0, 0])
epochs = np.arange(1, 61)
dl_loss = 2.8*np.exp(-0.09*epochs) + 0.12 + np.random.normal(0, 0.02, 60)
dl_acc  = 1 - 0.75*np.exp(-0.08*epochs) + np.random.normal(0, 0.01, 60)
ax2 = ax.twinx()
ax.plot(epochs, dl_loss, color=BLUE, lw=2, label='Loss')
ax2.plot(epochs, dl_acc, color=GREEN, lw=2, ls='--', label='Accuracy')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss', color=BLUE)
ax2.set_ylabel('Accuracy', color=GREEN)
ax.set_title('① Loss + Accuracy (dual-axis)\n(DL training dashboard)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=7.5, loc='center right')
add_code(ax, 'ax2 = ax.twinx()\nax.plot(epochs, loss)\nax2.plot(epochs, acc, color="g")')

# ── 2. LR Finder plot ─────────────────────────────────────────────────────
ax = fig5.add_subplot(gs5[0, 1])
lrs_log = np.logspace(-6, 0, 100)
loss_finder = 3.0 * np.exp(-2.5*np.log10(lrs_log+1e-6)) + 0.3
loss_finder += np.random.normal(0, 0.05, 100)
loss_finder[60:] += np.linspace(0, 3, 40)
ax.semilogx(lrs_log, loss_finder, color=BLUE, lw=2)
ax.axvline(lrs_log[np.argmin(loss_finder)-10], color=RED, lw=2, ls='--',
           label=f'Optimal LR≈{lrs_log[np.argmin(loss_finder)-10]:.4f}')
ax.set_xlabel('Learning Rate (log scale)'); ax.set_ylabel('Loss')
ax.set_title('② LR Finder Plot\n(optimal learning rate search)')
ax.legend(fontsize=7.5)
add_code(ax, 'ax.semilogx(lr_range, losses)\nax.axvline(optimal_lr, ls="--")')

# ── 3. Attention Heatmap ──────────────────────────────────────────────────
ax = fig5.add_subplot(gs5[0, 2])
tokens = ['The', 'model', 'learned', 'attention', 'weights', '[SEP]']
attn   = np.abs(np.random.randn(6, 6))
attn   = attn / attn.sum(axis=1, keepdims=True)
np.fill_diagonal(attn, attn.diagonal()*1.5)
sns.heatmap(attn, ax=ax, annot=True, fmt='.2f', cmap='Purples',
            xticklabels=tokens, yticklabels=tokens,
            linewidths=0.5, cbar_kws={'shrink':0.8}, annot_kws={'size':8})
ax.set_title('③ Attention Heatmap\n(NLP Transformer attention weights)')
ax.tick_params(axis='x', rotation=30)
add_code(ax, 'sns.heatmap(attention_matrix, annot=True,\n  cmap="Purples", xticklabels=tokens)')

# ── 4. Word / Token Embeddings (t-SNE style) ──────────────────────────────
ax = fig5.add_subplot(gs5[0, 3])
from sklearn.manifold import TSNE
embed_data = np.random.randn(80, 50)
groups     = np.repeat([0,1,2,3], 20)
group_names = ['Sentiment+','Sentiment−','Entity','Action']
tsne_result = TSNE(n_components=2, random_state=42, perplexity=15).fit_transform(embed_data)
for i, gname in enumerate(group_names):
    mask = groups == i
    ax.scatter(tsne_result[mask,0], tsne_result[mask,1],
               color=PALETTE[i], s=30, alpha=0.75, edgecolor='none', label=gname)
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
ax.set_title('④ t-SNE Embedding Plot\n(NLP/CV feature space visualization)')
ax.legend(fontsize=7.5)
add_code(ax, 'from sklearn.manifold import TSNE\ntsne = TSNE(2).fit_transform(embeddings)')

# ── 5. Class Activation Map (CAM) simulation ─────────────────────────────
ax = fig5.add_subplot(gs5[1, 0])
cam = np.zeros((8, 8))
cx, cy = 4, 3
for i in range(8):
    for j in range(8):
        cam[i,j] = np.exp(-((i-cx)**2 + (j-cy)**2) / 4)
cam += np.random.uniform(0, 0.15, (8,8))
im = ax.imshow(cam, cmap='jet', interpolation='bilinear', aspect='auto')
ax.contour(cam, levels=5, colors='white', linewidths=0.7, alpha=0.6)
plt.colorbar(im, ax=ax, shrink=0.85)
ax.set_title('⑤ Class Activation Map (CAM)\n(CV: what region the model focuses on)')
add_code(ax, 'ax.imshow(cam_heatmap, cmap="jet")\nax.contour(cam, colors="white")')

# ── 6. Image grid (augmentation preview) ──────────────────────────────────
ax = fig5.add_subplot(gs5[1, 1])
ax.axis('off')
inner = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs5[1,1], hspace=0.05, wspace=0.05)
transforms = ['Original','Flip','Crop','Rotate','Blur','Noise','Color Jit','Cutout']
for i, title in enumerate(transforms):
    ax_in = fig5.add_subplot(inner[i//4, i%4])
    fake_img = np.random.uniform(0.2, 0.9, (12, 12, 3))
    if i == 1: fake_img = fake_img[:, ::-1]
    elif i == 2: fake_img = fake_img[2:10, 2:10]; fake_img = np.pad(fake_img, ((2,2),(2,2),(0,0)), mode='edge')
    elif i == 3: fake_img = np.rot90(fake_img)
    elif i == 4: from scipy.ndimage import gaussian_filter; fake_img = gaussian_filter(fake_img, sigma=1.2)
    elif i == 5: fake_img += np.random.normal(0, 0.1, fake_img.shape)
    elif i == 7: fake_img[4:8, 4:8] = 0
    ax_in.imshow(np.clip(fake_img, 0, 1)); ax_in.axis('off')
    ax_in.set_title(title, fontsize=6.5, pad=1)
ax.set_title('⑥ Augmentation Grid\n(CV data augmentation preview)')
add_code(ax, 'fig, axes = plt.subplots(2, 4)\n# show each transform on axes[i]')

# ── 7. Gradient / Weight Distribution ─────────────────────────────────────
ax = fig5.add_subplot(gs5[1, 2])
layers = ['Conv1', 'Conv2', 'Dense1', 'Dense2', 'Output']
for i, layer in enumerate(layers):
    grad_data = np.random.normal(0, 0.1/(i+1), 500)
    kde_g = stats.gaussian_kde(grad_data)
    x_g   = np.linspace(-0.5, 0.5, 200)
    ax.plot(x_g, kde_g(x_g) + i*3, color=PALETTE[i], lw=2, label=layer)
    ax.fill_between(x_g, kde_g(x_g)+i*3, i*3, alpha=0.20, color=PALETTE[i])
ax.set_xlabel('Gradient Value'); ax.set_yticks([])
ax.set_title('⑦ Gradient Distribution\n(vanishing/exploding grad check)')
ax.legend(fontsize=7.5)
ax.axvline(0, color=DARK, lw=1, ls='--')
add_code(ax, '# After each epoch:\nfor layer in model.layers:\n  ax.hist(layer.grad.flatten())')

# ── 8. Activation Distribution ────────────────────────────────────────────
ax = fig5.add_subplot(gs5[1, 3])
epochs_act = [1, 5, 10, 20, 50]
for i, ep in enumerate(epochs_act):
    act_data = np.random.normal(0, 0.3 + 0.05*ep, 500)
    act_data = np.clip(act_data, -2, 2)  # ReLU-like
    act_data[act_data < 0] = 0
    kde_a = stats.gaussian_kde(act_data[act_data > 0] + 1e-5)
    x_a   = np.linspace(0, 2.5, 200)
    ax.plot(x_a, kde_a(x_a), color=PALETTE[i], lw=2, label=f'Epoch {ep}')
ax.set_xlabel('Activation Value'); ax.set_ylabel('Density')
ax.set_title('⑧ Activation Distribution\n(layer health over training)')
ax.legend(fontsize=7.5)
add_code(ax, 'acts = model.get_layer("relu").output\nsns.kdeplot(acts.numpy().flatten())')

# ── 9. Time Series forecast ───────────────────────────────────────────────
ax = fig5.add_subplot(gs5[2, 0])
t_ts = np.arange(100)
actual = (2*np.sin(t_ts*0.2) + 0.05*t_ts
          + np.random.normal(0, 0.4, 100))
ax.plot(t_ts[:80], actual[:80], color=BLUE, lw=2, label='Historical')
ax.plot(t_ts[79:], actual[79:], color=LITE, lw=1.5, ls='--', label='Actual (held out)')
forecast = actual[79] + np.linspace(0, 1.5, 21) + np.random.normal(0, 0.3, 21)
ax.plot(t_ts[79:], forecast, color=RED, lw=2, label='Forecast')
ax.fill_between(t_ts[79:], forecast-0.8, forecast+0.8, alpha=0.18, color=RED, label='95% PI')
ax.axvline(79, color=DARK, lw=1.5, ls='--')
ax.set_xlabel('Time Step'); ax.set_ylabel('Value')
ax.set_title('⑨ Time Series Forecast\n(LSTM/ARIMA/Prophet output)')
ax.legend(fontsize=7, ncol=2)
add_code(ax, 'ax.plot(t, forecast, label="Forecast")\nax.fill_between(t, lb, ub, alpha=0.2)')

# ── 10. Anomaly Detection ─────────────────────────────────────────────────
ax = fig5.add_subplot(gs5[2, 1])
t_ad = np.arange(200)
signal = np.sin(t_ad*0.15) + np.random.normal(0, 0.2, 200)
anomaly_idx = [45, 90, 130, 175]
signal[anomaly_idx] += np.random.uniform(2, 3.5, 4)
threshold = signal.mean() + 2.5*signal.std()
is_anomaly = signal > threshold
ax.plot(t_ad, signal, color=BLUE, lw=1.5, alpha=0.8, label='Signal')
ax.axhline(threshold, color=AMBER, lw=2, ls='--', label=f'Threshold (μ+2.5σ)')
ax.scatter(t_ad[is_anomaly], signal[is_anomaly], color=RED, s=80, zorder=5,
           marker='^', label='Anomaly', edgecolor='white')
ax.set_xlabel('Time'); ax.set_ylabel('Value')
ax.set_title('⑩ Anomaly Detection Plot\n(monitoring, alerting, AIOps)')
ax.legend(fontsize=7.5)
add_code(ax, 'ax.scatter(t[is_anomaly], x[is_anomaly],\n  color="red", marker="^", zorder=5)')

# ── 11. Hyperparameter tuning heatmap ────────────────────────────────────
ax = fig5.add_subplot(gs5[2, 2])
lr_vals   = [0.001, 0.005, 0.01, 0.05, 0.1]
depth_vals = [2, 3, 4, 5, 6]
acc_grid  = np.array([[0.72, 0.78, 0.80, 0.77, 0.71],
                       [0.75, 0.82, 0.86, 0.83, 0.76],
                       [0.76, 0.84, 0.91, 0.87, 0.79],
                       [0.74, 0.81, 0.88, 0.84, 0.77],
                       [0.70, 0.76, 0.82, 0.79, 0.73]])
im2 = sns.heatmap(acc_grid, ax=ax, annot=True, fmt='.2f', cmap='YlGn',
                   xticklabels=[str(l) for l in lr_vals],
                   yticklabels=[str(d) for d in depth_vals],
                   linewidths=0.5, cbar_kws={'shrink':0.8}, annot_kws={'size':8.5})
ax.set_xlabel('Learning Rate'); ax.set_ylabel('Max Depth')
ax.set_title('⑪ Hyperparameter Grid Search\n(GridSearch CV accuracy heatmap)')
add_code(ax, 'sns.heatmap(results.pivot("lr","depth","score"),\n  annot=True, cmap="YlGn")')

# ── 12. Model Comparison (radar) ──────────────────────────────────────────
ax = fig5.add_subplot(gs5[2, 3], polar=True)
metrics = ['Accuracy','Precision','Recall','F1','AUC','Speed']
N_m = len(metrics)
angles_m = np.linspace(0, 2*np.pi, N_m, endpoint=False).tolist() + [0]
models_m = {
    'Random Forest': [0.91, 0.89, 0.88, 0.89, 0.95, 0.60],
    'Logistic Reg':  [0.83, 0.81, 0.82, 0.81, 0.88, 0.95],
    'XGBoost':       [0.93, 0.92, 0.91, 0.92, 0.97, 0.70],
}
for i, (model_name, vals) in enumerate(models_m.items()):
    vals_closed = vals + vals[:1]
    ax.plot(angles_m, vals_closed, color=PALETTE[i], lw=2.2, label=model_name)
    ax.fill(angles_m, vals_closed, color=PALETTE[i], alpha=0.10)
ax.set_xticks(angles_m[:-1])
ax.set_xticklabels(metrics, fontsize=8)
ax.set_ylim(0, 1)
ax.set_title('⑫ Model Comparison Radar\n(multi-metric model selection)', pad=18)
ax.legend(fontsize=7.5, loc='lower right')
add_code(ax, 'ax = fig.add_subplot(polar=True)\nax.plot(angles, values)\nax.fill(angles, values)')

fig5.savefig('/mnt/user-data/outputs/PLOTS_5_DL_AI_Production.png', bbox_inches='tight', dpi=130)
print('✓ Figure 5 — DL/AI Production Plots saved')
plt.close(fig5)


# =============================================================================
# FIGURE 6 — MASTER REFERENCE TABLE
# =============================================================================
fig6 = plt.figure(figsize=(22, 20))
fig6.patch.set_facecolor(BG)
ax_ref = fig6.add_axes([0, 0, 1, 1])
ax_ref.axis('off')
fig6.suptitle('COMPLETE REFERENCE: When to Use Which Plot',
              fontsize=17, fontweight='bold', y=0.985, color=DARK)

headers = ['Category', 'Plot Name', 'Best For', 'Data Type', 'Key Function', 'Industry Use']
col_widths = [0.13, 0.15, 0.20, 0.12, 0.22, 0.16]
col_starts = [0.01]
for w in col_widths[:-1]:
    col_starts.append(col_starts[-1]+w)

# Header row
header_bg = FancyBboxPatch((0.005, 0.945), 0.99, 0.028, boxstyle="round,pad=0.003",
                            lw=0, facecolor=DARK, transform=ax_ref.transAxes)
ax_ref.add_patch(header_bg)
for j, (hdr, cx) in enumerate(zip(headers, col_starts)):
    ax_ref.text(cx+0.005, 0.958, hdr, transform=ax_ref.transAxes,
                fontsize=8.5, fontweight='bold', color='white', va='center')

rows = [
    # (category, name, best_for, dtype, function, industry, row_color)
    ('UNIVARIATE', 'Histogram',           'Distribution shape',         'Numeric',   'plt.hist / sns.histplot',            'EDA, feature analysis',          BLUE+'18'),
    ('UNIVARIATE', 'KDE Plot',            'Smooth density estimate',     'Numeric',   'sns.kdeplot(fill=True)',             'Distribution comparison',        BLUE+'18'),
    ('UNIVARIATE', 'Box Plot',            '5-number summary + outliers', 'Numeric',   'ax.boxplot / sns.boxplot',           'Feature outlier reports',        BLUE+'18'),
    ('UNIVARIATE', 'Violin Plot',         'Distribution shape + box',    'Numeric',   'sns.violinplot',                     'EDA, group comparison',          BLUE+'18'),
    ('UNIVARIATE', 'ECDF',                'Cumulative distribution',     'Numeric',   'sns.ecdfplot',                       'Statistical reporting',          BLUE+'18'),
    ('UNIVARIATE', 'Q-Q Plot',            'Normality check',             'Numeric',   'stats.probplot(data, plot=ax)',      'Model assumption testing',       BLUE+'18'),
    ('UNIVARIATE', 'Bar Chart',           'Category frequency',          'Categorical','sns.countplot / ax.bar',            'Dashboards, reports',            BLUE+'18'),
    ('UNIVARIATE', 'Rug Plot',            'Raw data density',            'Numeric',   'sns.rugplot',                        'Overlaid with KDE',              BLUE+'18'),
    ('BIVARIATE',  'Scatter Plot',        'Numeric relationship',        'Num×Num',   'ax.scatter / sns.scatterplot',      'Feature correlation',            GREEN+'18'),
    ('BIVARIATE',  'Reg Plot',            'Regression + CI band',        'Num×Num',   'sns.regplot(ci=95)',                 'Linear relationship',            GREEN+'18'),
    ('BIVARIATE',  'Joint Plot',          'Bivariate + marginals',       'Num×Num',   'sns.jointplot(kind="scatter")',      'EDA, pair exploration',          GREEN+'18'),
    ('BIVARIATE',  'Hex Bin',             'Dense scatter (large n)',      'Num×Num',   'ax.hexbin(gridsize=20)',             'Big data EDA',                   GREEN+'18'),
    ('BIVARIATE',  'Box by Group',        'Numeric vs categorical',       'Mix',       'sns.boxplot(x="cat", y="num")',     'A/B test, group compare',        GREEN+'18'),
    ('BIVARIATE',  'Violin by Group',     'Shape per category',          'Mix',       'sns.violinplot(x, y, inner="box")', 'ML feature importance',          GREEN+'18'),
    ('BIVARIATE',  'Corr Heatmap',        'All pairwise correlations',   'Num×Num',   'sns.heatmap(df.corr(), annot=True)','Feature selection',              GREEN+'18'),
    ('BIVARIATE',  '2D KDE Contour',      'Bivariate density',           'Num×Num',   'sns.kdeplot(x, y, fill=True)',      'Probability density maps',       GREEN+'18'),
    ('MULTIVARIATE','Pair Plot',          'All feature pairs at once',   'Numeric',   'sns.pairplot(df, hue="class")',     'EDA on structured datasets',     AMBER+'18'),
    ('MULTIVARIATE','Bubble Chart',       '4 variables (x,y,size,color)','Numeric',   'ax.scatter(s=size, c=color)',       'Business intelligence',          AMBER+'18'),
    ('MULTIVARIATE','PCA Scatter',        'High-dim data in 2D',         'Numeric',   'PCA(2).fit_transform(X)',           'Dimensionality reduction',       AMBER+'18'),
    ('MULTIVARIATE','Parallel Coords',    'Compare profiles across feats','Numeric',   'pandas.plotting.parallel_coordinates','Multi-feature comparison',    AMBER+'18'),
    ('MULTIVARIATE','Radar Chart',        'Multi-metric profiles',        'Numeric',   'ax = fig.add_subplot(polar=True)', 'Model comparison',               AMBER+'18'),
    ('MULTIVARIATE','Cluster Scatter',    'Cluster membership in 2D',    'Numeric',   'ax.scatter(c=km.labels_)',          'Customer segmentation',          AMBER+'18'),
    ('MULTIVARIATE','3D Scatter',         '3 continuous variables',      'Numeric',   'ax = fig.add_subplot(projection="3d")', 'Physics, engineering',       AMBER+'18'),
    ('MULTIVARIATE','Stacked Area',       'Part-to-whole over time',     'Time series','ax.stackplot(t, y1, y2)',          'Revenue, traffic dashboards',    AMBER+'18'),
    ('ML EVAL',    'Confusion Matrix',    'Classification errors',       'Labels',    'sns.heatmap(confusion_matrix(y,ŷ))','Every classifier in prod',     RED+'18'),
    ('ML EVAL',    'ROC Curve',           'Threshold trade-off',         'Prob+Labels','roc_curve + ax.plot(fpr, tpr)',    'Binary/multi-class models',      RED+'18'),
    ('ML EVAL',    'PR Curve',            'Imbalanced class eval',       'Prob+Labels','precision_recall_curve(y, prob)',  'Fraud, medical diagnosis',       RED+'18'),
    ('ML EVAL',    'Feature Importance',  'Tree model feature ranks',    'Model',     'ax.barh(feats, rf.importances_)',  'XGBoost, RF, LightGBM',          RED+'18'),
    ('ML EVAL',    'Learning Curve',      'Bias-variance diagnosis',     'Model',     'sklearn.model_selection.learning_curve','Train size vs accuracy',    RED+'18'),
    ('ML EVAL',    'SHAP Waterfall',      'Per-prediction explainability','Model',    'shap.plots.waterfall(shap_vals[i])', 'Regulatory AI, MLOps',         RED+'18'),
    ('DL / AI',    'Train/Val Loss',      'Overfitting detection',       'Training',  'ax.plot(epochs, loss)',             'All DL training pipelines',      PUR+'18'),
    ('DL / AI',    'Attention Heatmap',   'Transformer weights',         'NLP/Vision','sns.heatmap(attn_matrix)',          'BERT, GPT, ViT interpretability',PUR+'18'),
    ('DL / AI',    't-SNE / UMAP',        'Embedding space visual',      'Embeddings','TSNE(2).fit_transform(embeddings)', 'NLP, CV feature space',          PUR+'18'),
    ('DL / AI',    'CAM / Grad-CAM',      'Image focus region',          'CV',        'ax.imshow(cam, cmap="jet")',        'Object detection models',        PUR+'18'),
    ('DL / AI',    'Gradient Distribution','Vanishing gradient check',   'DL',        'ax.hist(layer.grad.flatten())',    'DNN training monitoring',        PUR+'18'),
    ('DL / AI',    'Anomaly Detection',   'Out-of-distribution points',  'Time series','ax.scatter(t[anoms], x[anoms])',  'AIOps, fraud, monitoring',       PUR+'18'),
    ('DL / AI',    'HP Tuning Heatmap',   'Best hyperparameter combo',   'Grid/Random','sns.heatmap(results_pivot)',       'MLflow, W&B experiment tracking',PUR+'18'),
    ('DL / AI',    'Forecast + PI',       'Predicted + confidence band', 'Time series','ax.fill_between(t, lb, ub)',       'LSTM, Prophet, NeuralForecast',  PUR+'18'),
]

row_height = 0.022
start_y    = 0.934
for i, row in enumerate(rows):
    y = start_y - (i+1)*row_height
    row_bg_color = row[6]
    bg = FancyBboxPatch((0.005, y-0.001), 0.99, row_height-0.001,
                         boxstyle="square,pad=0", lw=0,
                         facecolor=row_bg_color, transform=ax_ref.transAxes)
    ax_ref.add_patch(bg)
    for j, (cell, cx) in enumerate(zip(row[:6], col_starts)):
        fw = 'bold' if j == 0 else 'normal'
        col_text = [BLUE, DARK, DARK, MID, '#1A7A4A', MID][j]
        ax_ref.text(cx+0.005, y + row_height*0.42, cell,
                    transform=ax_ref.transAxes, fontsize=7.5, va='center',
                    color=col_text, fontweight=fw,
                    fontfamily='monospace' if j == 4 else 'sans-serif')

fig6.savefig('/mnt/user-data/outputs/PLOTS_6_Reference_Table.png', bbox_inches='tight', dpi=130)
print('✓ Figure 6 — Master Reference Table saved')
plt.close(fig6)

print('\n' + '='*60)
print('ALL 6 FIGURES SAVED')
print('='*60)
print("""
  PLOTS_1_Univariate.png      — 11 univariate plots
  PLOTS_2_Bivariate.png       — 12 bivariate plots
  PLOTS_3_Multivariate.png    — 12 multivariate plots
  PLOTS_4_ML_Evaluation.png   — 12 ML model evaluation plots
  PLOTS_5_DL_AI_Production.png— 12 DL/NLP/CV/Time-series plots
  PLOTS_6_Reference_Table.png — Master when-to-use reference
""")
