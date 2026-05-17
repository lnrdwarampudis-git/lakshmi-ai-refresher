"""
=============================================================================
COMPLETE THEORY + MATH + WORKED EXAMPLES
Univariate · Bivariate · Multivariate Statistical Analysis
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Style ────────────────────────────────────────────────────────────────────
BG      = '#FAFAF8'
CARD    = '#F3F1EB'
BLUE    = '#1A5FA8'
RED     = '#C0392B'
GREEN   = '#1A7A4A'
AMBER   = '#B5770D'
PURPLE  = '#6A3D9A'
TEAL    = '#0B7A75'
DARK    = '#1A1A1A'
MID     = '#555555'
LIGHT   = '#999999'
BORDER  = '#D8D5CC'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#E5E3DC', 'grid.linewidth': 0.5,
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'figure.dpi': 130,
})

# ── Sample data (used throughout) ────────────────────────────────────────────
scores    = np.array([55, 62, 68, 70, 71, 73, 74, 75, 76, 77,
                      78, 79, 80, 81, 82, 83, 85, 87, 90, 95])
hours     = np.array([2,  3,  4,  4,  5,  5,  6,  6,  7,  7,
                      7,  8,  8,  9,  9,  10, 10, 11, 12, 12])
n         = len(scores)

# ─────────────────────────────────────────────────────────────────────────────
def section_box(ax, x, y, w, h, title, color, facecolor=None):
    """Draw a titled section box."""
    fc = facecolor or BG
    rect = FancyBboxPatch((x,y), w, h,
                          boxstyle="round,pad=0.01",
                          linewidth=1.5, edgecolor=color,
                          facecolor=fc, transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.008, title,
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9.5, fontweight='bold', color=color)

def theory_text(ax, x, y, lines, color=DARK, size=9, spacing=0.038):
    for i, line in enumerate(lines):
        ax.text(x, y - i*spacing, line,
                transform=ax.transAxes, ha='left', va='top',
                fontsize=size, color=color, linespacing=1.4)

# =============================================================================
# PAGE 1 — UNIVARIATE: THEORY + MATH
# =============================================================================
fig1 = plt.figure(figsize=(20, 26))
fig1.patch.set_facecolor(BG)
fig1.suptitle('UNIVARIATE ANALYSIS — Complete Theory & Mathematics',
              fontsize=18, fontweight='bold', y=0.985, color=DARK)

gs = gridspec.GridSpec(5, 2, figure=fig1, hspace=0.12, wspace=0.08,
                       top=0.96, bottom=0.02, left=0.04, right=0.96)

# ─── SECTION 1: Definition ────────────────────────────────────────────────────
ax0 = fig1.add_subplot(gs[0, :])
ax0.axis('off')

# Banner
banner = FancyBboxPatch((0, 0.65), 1.0, 0.33,
                         boxstyle="round,pad=0.01", lw=0,
                         facecolor='#EAF0F8', transform=ax0.transAxes)
ax0.add_patch(banner)
ax0.text(0.5, 0.94, '📊  WHAT IS UNIVARIATE ANALYSIS?',
         transform=ax0.transAxes, ha='center', fontsize=13,
         fontweight='bold', color=BLUE)
ax0.text(0.5, 0.86,
         'Univariate analysis examines ONE variable at a time. The goal is to describe the distribution,\n'
         'central tendency, spread, and shape of a single feature — before any relationships are explored.',
         transform=ax0.transAxes, ha='center', fontsize=10.5, color=DARK, linespacing=1.6)
ax0.text(0.5, 0.72,
         'Key question:  "What does this variable look like? Where is it centred? How spread out is it? Is it symmetric?"',
         transform=ax0.transAxes, ha='center', fontsize=10, color=BLUE, style='italic')

# ─── Three columns: Central Tendency | Spread | Shape ─────────────────────────
ax0.text(0.02,  0.58, '① CENTRAL TENDENCY', transform=ax0.transAxes,
         fontsize=11, fontweight='bold', color=BLUE)
ax0.text(0.35,  0.58, '② SPREAD / DISPERSION', transform=ax0.transAxes,
         fontsize=11, fontweight='bold', color=GREEN)
ax0.text(0.68,  0.58, '③ SHAPE DESCRIPTORS', transform=ax0.transAxes,
         fontsize=11, fontweight='bold', color=PURPLE)

# Central tendency formulas
ct_lines = [
    'MEAN (Arithmetic Average)',
    '',
    '  x̄  =  Σxᵢ / n',
    '',
    '  → Sum all values, divide by count.',
    '  → Sensitive to outliers.',
    '',
    'MEDIAN (Middle value)',
    '',
    '  • Sort data ascending',
    '  • n odd:   M = x[(n+1)/2]',
    '  • n even:  M = (x[n/2] + x[n/2+1]) / 2',
    '  → Robust to outliers.',
    '',
    'MODE',
    '  • Most frequently occurring value',
    '  • Can be multiple (bimodal, multimodal)',
    '  → Used for categorical data',
]
for i, line in enumerate(ct_lines):
    ax0.text(0.02, 0.52 - i*0.028, line,
             transform=ax0.transAxes, fontsize=8.8, color=DARK,
             fontfamily='monospace' if '=' in line or '→' in line else 'sans-serif')

# Spread formulas
sp_lines = [
    'VARIANCE (s²)',
    '',
    '  s² = Σ(xᵢ − x̄)² / (n−1)',
    '',
    '  → Average squared deviation from mean.',
    '  → (n−1): Bessel correction for sample.',
    '',
    'STANDARD DEVIATION (s)',
    '',
    '  s = √[ Σ(xᵢ − x̄)² / (n−1) ]',
    '',
    '  → Same units as the data.',
    '  → 68-95-99.7 rule (normal dist)',
    '',
    'INTERQUARTILE RANGE (IQR)',
    '',
    '  IQR = Q3 − Q1',
    '  Q1 = 25th percentile',
    '  Q3 = 75th percentile',
    '  → Robust: ignores extreme values',
    '  → Outlier: x < Q1−1.5·IQR',
    '         or  x > Q3+1.5·IQR',
]
for i, line in enumerate(sp_lines):
    ax0.text(0.35, 0.52 - i*0.028, line,
             transform=ax0.transAxes, fontsize=8.8, color=DARK,
             fontfamily='monospace' if '=' in line or '→' in line else 'sans-serif')

# Shape formulas
sh_lines = [
    'SKEWNESS (γ₁)',
    '',
    '  γ₁ = [Σ(xᵢ−x̄)³/n] / s³',
    '',
    '  γ₁ = 0  → symmetric',
    '  γ₁ > 0  → right-skewed (long right tail)',
    '  γ₁ < 0  → left-skewed  (long left tail)',
    '  |γ₁| > 1 → substantially skewed',
    '',
    'KURTOSIS (γ₂)',
    '',
    '  γ₂ = [Σ(xᵢ−x̄)⁴/n] / s⁴ − 3',
    '',
    '  γ₂ = 0  → normal (mesokurtic)',
    '  γ₂ > 0  → heavy tails (leptokurtic)',
    '  γ₂ < 0  → light tails (platykurtic)',
    '',
    'COEFFICIENT OF VARIATION (CV)',
    '',
    '  CV = (s / x̄) × 100%',
    '  → Relative spread (unit-free)',
]
for i, line in enumerate(sh_lines):
    ax0.text(0.68, 0.52 - i*0.028, line,
             transform=ax0.transAxes, fontsize=8.8, color=DARK,
             fontfamily='monospace' if '=' in line or '→' in line or '|' in line else 'sans-serif')

# ─── WORKED EXAMPLE ──────────────────────────────────────────────────────────
ax1 = fig1.add_subplot(gs[1, :])
ax1.axis('off')

# Section header
hdr = FancyBboxPatch((0, 0.88), 1.0, 0.10,
                      boxstyle="round,pad=0.01", lw=0,
                      facecolor='#EAF5EE', transform=ax1.transAxes)
ax1.add_patch(hdr)
ax1.text(0.5, 0.96, '✏️  WORKED EXAMPLE — Student Exam Scores',
         transform=ax1.transAxes, ha='center', fontsize=12,
         fontweight='bold', color=GREEN)

# Show raw data
ax1.text(0.01, 0.84, 'Raw data (n=20 students):', transform=ax1.transAxes,
         fontsize=9.5, fontweight='bold', color=DARK)
ax1.text(0.01, 0.78,
         str(scores.tolist()),
         transform=ax1.transAxes, fontsize=9, color=BLUE, fontfamily='monospace')

# Step-by-step calculation
mean_val   = np.mean(scores)
median_val = np.median(scores)
mode_val   = stats.mode(scores, keepdims=True).mode[0]
var_val    = np.var(scores, ddof=1)
std_val    = np.std(scores, ddof=1)
q1, q3     = np.percentile(scores, [25, 75])
iqr_val    = q3 - q1
skew_val   = stats.skew(scores)
kurt_val   = stats.kurtosis(scores)
cv_val     = (std_val / mean_val) * 100

steps = [
    ('STEP 1 — MEAN', BLUE,
     f'x̄ = ({" + ".join(str(x) for x in scores[:5])} + ... + {scores[-1]}) / {n}  =  {sum(scores)} / {n}  =  {mean_val:.2f}'),
    ('STEP 2 — MEDIAN', GREEN,
     f'Sorted: [{", ".join(str(x) for x in sorted(scores)[:5])}, ..., {sorted(scores)[-1]}]   →   Average of 10th & 11th values  =  ({sorted(scores)[9]} + {sorted(scores)[10]}) / 2  =  {median_val:.1f}'),
    ('STEP 3 — VARIANCE', AMBER,
     f's² = [(55−{mean_val:.1f})² + (62−{mean_val:.1f})² + ... ] / (20−1)  =  {sum((x-mean_val)**2 for x in scores):.1f} / 19  =  {var_val:.2f}'),
    ('STEP 4 — STD DEV', PURPLE,
     f's = √{var_val:.2f}  =  {std_val:.2f}  → Scores typically deviate ±{std_val:.1f} points from the mean'),
    ('STEP 5 — IQR & QUARTILES', TEAL,
     f'Q1 = {q1:.1f}  |  Q3 = {q3:.1f}  |  IQR = {q3:.1f} − {q1:.1f} = {iqr_val:.1f}   Outlier fences: [{q1-1.5*iqr_val:.1f}, {q3+1.5*iqr_val:.1f}]'),
    ('STEP 6 — SKEWNESS & KURTOSIS', RED,
     f'Skewness = {skew_val:.3f}  ({"right-skewed" if skew_val>0 else "left-skewed"})   |   Kurtosis = {kurt_val:.3f}   |   CV = {cv_val:.1f}%'),
]

for i, (label, color, text) in enumerate(steps):
    y = 0.70 - i * 0.115
    step_bg = FancyBboxPatch((0, y - 0.005), 1.0, 0.10,
                              boxstyle="round,pad=0.005", lw=0.8,
                              edgecolor=color, facecolor=BG,
                              transform=ax1.transAxes)
    ax1.add_patch(step_bg)
    ax1.text(0.01, y + 0.075, label, transform=ax1.transAxes,
             fontsize=8.5, fontweight='bold', color=color)
    ax1.text(0.16, y + 0.075, text, transform=ax1.transAxes,
             fontsize=8.5, color=DARK, fontfamily='monospace')

# Summary box
summary_y = 0.70 - len(steps)*0.115 - 0.01
sum_bg = FancyBboxPatch((0, summary_y - 0.01), 1.0, 0.075,
                         boxstyle="round,pad=0.005", lw=1.5,
                         edgecolor=DARK, facecolor='#F5F0FF',
                         transform=ax1.transAxes)
ax1.add_patch(sum_bg)
summary = (f'SUMMARY:  n={n}  |  Mean={mean_val:.2f}  |  Median={median_val:.1f}  |  '
           f'Mode={mode_val}  |  SD={std_val:.2f}  |  Var={var_val:.2f}  |  '
           f'IQR={iqr_val:.1f}  |  Skew={skew_val:.3f}  |  Kurt={kurt_val:.3f}  |  CV={cv_val:.1f}%')
ax1.text(0.5, summary_y + 0.045, summary,
         transform=ax1.transAxes, ha='center', fontsize=9,
         fontweight='bold', color=DARK, fontfamily='monospace')

# ─── VISUALIZATIONS ──────────────────────────────────────────────────────────
ax_hist = fig1.add_subplot(gs[2, 0])
ax_box  = fig1.add_subplot(gs[2, 1])
ax_qq   = fig1.add_subplot(gs[3, 0])
ax_skew = fig1.add_subplot(gs[3, 1])
ax_code = fig1.add_subplot(gs[4, :])

# Histogram
ax_hist.hist(scores, bins=10, color=BLUE, alpha=0.65, edgecolor='white', density=True)
kde_x = np.linspace(scores.min()-5, scores.max()+5, 300)
kde = stats.gaussian_kde(scores, bw_method=0.5)
ax_hist.plot(kde_x, kde(kde_x), color=RED, lw=2.5, label='KDE')
ax_hist.axvline(mean_val,   color=AMBER,  lw=2, ls='--', label=f'Mean={mean_val:.1f}')
ax_hist.axvline(median_val, color=GREEN,  lw=2, ls='-.',  label=f'Median={median_val:.1f}')
ax_hist.axvline(mode_val,   color=PURPLE, lw=2, ls=':',  label=f'Mode={mode_val}')
ax_hist.set_title('Histogram + KDE\n(Central Tendency Markers)')
ax_hist.set_xlabel('Exam Score'); ax_hist.set_ylabel('Density')
ax_hist.legend(fontsize=8)
# Shade 1-SD region
ax_hist.axvspan(mean_val-std_val, mean_val+std_val, alpha=0.08, color=BLUE)
ax_hist.text(mean_val, kde(np.array([mean_val]))[0]*0.5,
             f'±1SD\n({mean_val-std_val:.0f}–{mean_val+std_val:.0f})',
             ha='center', fontsize=7.5, color=BLUE)

# Box plot (detailed)
bp = ax_box.boxplot(scores, patch_artist=True, vert=True, widths=0.5,
                    medianprops=dict(color='white', lw=3),
                    boxprops=dict(facecolor=BLUE, alpha=0.5),
                    whiskerprops=dict(color=BLUE, lw=1.8, ls='--'),
                    capprops=dict(color=BLUE, lw=2.5),
                    flierprops=dict(marker='o', ms=7, color=RED, alpha=0.7))
annotations = [
    (scores.min(), f'Min = {scores.min()}'),
    (q1,           f'Q1 = {q1:.0f}'),
    (median_val,   f'Median = {median_val:.0f}'),
    (q3,           f'Q3 = {q3:.0f}'),
    (scores.max(), f'Max = {scores.max()}'),
]
for val, label in annotations:
    ax_box.annotate(f'  {label}', xy=(1, val), xytext=(1.35, val),
                    fontsize=8.5, va='center', color=DARK,
                    arrowprops=dict(arrowstyle='->', color=DARK, lw=0.8))
ax_box.set_title('Box Plot — 5-Number Summary\n& IQR Fence for Outliers')
ax_box.set_xticklabels(['Scores'])
ax_box.set_ylabel('Score')
ax_box.text(0.5, 0.04, f'IQR = {iqr_val:.0f}  |  Lower fence = {q1-1.5*iqr_val:.0f}  |  Upper fence = {q3+1.5*iqr_val:.0f}',
            transform=ax_box.transAxes, ha='center', fontsize=8,
            bbox=dict(boxstyle='round', fc='#EAF5EE', ec=GREEN))

# Q-Q Plot
(osm, osr), (slope, intercept, r) = stats.probplot(scores, dist='norm')
ax_qq.scatter(osm, osr, color=BLUE, s=50, zorder=5, label='Observed quantiles')
line_x = np.array([osm.min(), osm.max()])
ax_qq.plot(line_x, slope*line_x+intercept, color=RED, lw=2, label='Normal reference')
ax_qq.fill_between(line_x,
                   (slope*line_x+intercept) - 0.8*std_val*0.3,
                   (slope*line_x+intercept) + 0.8*std_val*0.3,
                   alpha=0.1, color=RED)
ax_qq.set_title('Q-Q Plot — Normality Check\n(Points on line → normal)')
ax_qq.set_xlabel('Theoretical Quantiles'); ax_qq.set_ylabel('Sample Quantiles')
ax_qq.legend(fontsize=8.5)
_, p_sw = stats.shapiro(scores)
ax_qq.text(0.03, 0.95,
           f'Shapiro-Wilk W test:\np = {p_sw:.4f}\n{"✓ Normal (p>0.05)" if p_sw>0.05 else "✗ Non-normal (p≤0.05)"}',
           transform=ax_qq.transAxes, fontsize=8.5, va='top',
           bbox=dict(boxstyle='round', fc='#EAF5EE' if p_sw>0.05 else '#FFEBEE',
                     ec=GREEN if p_sw>0.05 else RED))

# Skewness visual
x_range = np.linspace(-4, 4, 400)
ax_skew.plot(x_range, stats.norm.pdf(x_range), color=BLUE, lw=2.5,
             label='Symmetric  (γ₁=0)')
right = stats.skewnorm.pdf(x_range, a=6)
ax_skew.plot(x_range, right/right.max()*0.4, color=RED, lw=2, ls='--',
             label='Right skew  (γ₁>0)')
left  = stats.skewnorm.pdf(x_range, a=-6)
ax_skew.plot(x_range, left/left.max()*0.4,  color=GREEN, lw=2, ls='-.',
             label='Left skew  (γ₁<0)')
ax_skew.plot(x_range, stats.t.pdf(x_range, df=2)*1.2, color=PURPLE, lw=2, ls=':',
             label='High kurtosis  (γ₂>0)')
# Arrows showing tail directions
ax_skew.annotate('', xy=(3.2,0.04), xytext=(1.5,0.04),
                 arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
ax_skew.text(2.2, 0.055, 'right tail', fontsize=8, color=RED)
ax_skew.annotate('', xy=(-3.2,0.04), xytext=(-1.5,0.04),
                 arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))
ax_skew.text(-3.4, 0.055, 'left tail', fontsize=8, color=GREEN, ha='right')
ax_skew.set_title(f'Skewness & Kurtosis Shapes\n(Our data: γ₁={skew_val:.2f}, γ₂={kurt_val:.2f})')
ax_skew.set_xlabel('Standardized Value'); ax_skew.set_ylabel('Density')
ax_skew.legend(fontsize=8.5)

# Code reference
ax_code.axis('off')
code_bg = FancyBboxPatch((0, 0), 1.0, 1.0,
                          boxstyle="round,pad=0.01", lw=1.2,
                          edgecolor='#AAAAAA', facecolor='#F5F5F0',
                          transform=ax_code.transAxes)
ax_code.add_patch(code_bg)
ax_code.text(0.5, 0.96, '💻  MATPLOTLIB + SCIPY + SEABORN CODE', transform=ax_code.transAxes,
             ha='center', fontsize=11, fontweight='bold', color=DARK)

code_blocks = [
    (0.01, 0.88, BLUE, '# ── Central Tendency & Spread ───────────────────────────────────────────────'),
    (0.01, 0.82, DARK, 'import numpy as np; from scipy import stats'),
    (0.01, 0.76, DARK, 'mean   = np.mean(scores)               # arithmetic average'),
    (0.01, 0.70, DARK, 'median = np.median(scores)             # middle value (robust)'),
    (0.01, 0.64, DARK, 'mode   = stats.mode(scores).mode[0]    # most frequent value'),
    (0.01, 0.58, DARK, 'var    = np.var(scores, ddof=1)        # sample variance  s²'),
    (0.01, 0.52, DARK, 'std    = np.std(scores, ddof=1)        # standard deviation  s'),
    (0.01, 0.46, DARK, 'q1,q3  = np.percentile(scores,[25,75]) # quartiles'),
    (0.01, 0.40, DARK, 'iqr    = q3 - q1                       # interquartile range'),
    (0.01, 0.34, DARK, 'skew   = stats.skew(scores)            # skewness γ₁'),
    (0.01, 0.28, DARK, 'kurt   = stats.kurtosis(scores)        # excess kurtosis γ₂'),
    (0.01, 0.22, BLUE, '# ── Visualizations ──────────────────────────────────────────────────────────'),
    (0.01, 0.16, DARK, 'plt.hist(scores, bins=15, density=True)         # histogram'),
    (0.01, 0.10, DARK, 'sns.histplot(scores, kde=True)                  # histogram + KDE'),
    (0.01, 0.04, DARK, 'stats.probplot(scores, plot=plt)                # Q-Q plot'),
    (0.51, 0.88, BLUE, '# ── Box Plot ────────────────────────────────────────────────────────────────'),
    (0.51, 0.82, DARK, 'fig, ax = plt.subplots()'),
    (0.51, 0.76, DARK, 'ax.boxplot(scores, patch_artist=True)           # standard box plot'),
    (0.51, 0.70, DARK, 'sns.boxplot(y=scores)                           # seaborn style'),
    (0.51, 0.64, DARK, 'sns.violinplot(y=scores)                        # violin (shows KDE)'),
    (0.51, 0.58, BLUE, '# ── Normality Tests ─────────────────────────────────────────────────────────'),
    (0.51, 0.52, DARK, 'stat, p = stats.shapiro(scores)                 # Shapiro-Wilk test'),
    (0.51, 0.46, DARK, 'stat, p = stats.normaltest(scores)              # D\'Agostino-Pearson'),
    (0.51, 0.40, BLUE, '# ── Full summary ────────────────────────────────────────────────────────────'),
    (0.51, 0.34, DARK, 'pd.Series(scores).describe()                    # count, mean, std, min, Q1, Q2, Q3, max'),
    (0.51, 0.28, DARK, 'pd.Series(scores).skew()                        # skewness'),
    (0.51, 0.22, DARK, 'pd.Series(scores).kurtosis()                    # excess kurtosis'),
]
for cx, cy, color, text in code_blocks:
    ax_code.text(cx + 0.01, cy, text, transform=ax_code.transAxes,
                 fontsize=8.2, color=color, fontfamily='monospace', va='top')

fig1.savefig('/mnt/user-data/outputs/THEORY_1_Univariate.png', bbox_inches='tight', dpi=130)
print("✓ Page 1 saved — Univariate Theory & Math")
plt.close(fig1)


# =============================================================================
# PAGE 2 — BIVARIATE: THEORY + MATH
# =============================================================================
fig2 = plt.figure(figsize=(20, 28))
fig2.patch.set_facecolor(BG)
fig2.suptitle('BIVARIATE ANALYSIS — Complete Theory & Mathematics',
              fontsize=18, fontweight='bold', y=0.985, color=DARK)

gs2 = gridspec.GridSpec(6, 2, figure=fig2, hspace=0.14, wspace=0.08,
                        top=0.96, bottom=0.02, left=0.04, right=0.96)

# ─── Definition ──────────────────────────────────────────────────────────────
ax_def = fig2.add_subplot(gs2[0, :])
ax_def.axis('off')

banner2 = FancyBboxPatch((0, 0.82), 1.0, 0.16,
                          boxstyle="round,pad=0.01", lw=0,
                          facecolor='#EAF0F8', transform=ax_def.transAxes)
ax_def.add_patch(banner2)
ax_def.text(0.5, 0.96, '📊  WHAT IS BIVARIATE ANALYSIS?',
            transform=ax_def.transAxes, ha='center', fontsize=13,
            fontweight='bold', color=BLUE)
ax_def.text(0.5, 0.88,
            'Bivariate analysis examines the RELATIONSHIP between TWO variables.\n'
            'It answers: "Do these two things move together? How strongly? Can one predict the other?"',
            transform=ax_def.transAxes, ha='center', fontsize=10.5, color=DARK, linespacing=1.6)

# 4 technique theory blocks
techniques = [
    ('PEARSON CORRELATION  (r)', BLUE, [
        'Measures the LINEAR relationship between two numeric variables.',
        '',
        'Formula:',
        '',
        '         Σ (xᵢ − x̄)(yᵢ − ȳ)',
        '  r  =  ─────────────────────────────────',
        '        √[Σ(xᵢ−x̄)²] · √[Σ(yᵢ−ȳ)²]',
        '',
        'Equivalent:   r = Cov(X,Y) / (sₓ · sᵧ)',
        '',
        'Range: −1 ≤ r ≤ +1',
        '  r = +1 → perfect positive linear',
        '  r = −1 → perfect negative linear',
        '  r =  0 → no linear relationship',
        '',
        'Interpretation guide:',
        '  |r| < 0.3  → weak',
        '  |r| 0.3–0.7 → moderate',
        '  |r| > 0.7  → strong',
        '',
        'Significance test:',
        '  H₀: ρ = 0  (no correlation in population)',
        '  t = r√(n−2) / √(1−r²)',
        '  t ~ t-distribution with df = n−2',
        '',
        'Assumption: both variables continuous,',
        '  linear relationship, no extreme outliers.',
        '',
        'SPEARMAN ρ (rank-based version):',
        '  Applies Pearson to RANKS of data.',
        '  Robust to outliers + monotonic relations.',
        '  ρ = 1 − 6Σdᵢ² / [n(n²−1)]',
    ]),
    ('SIMPLE LINEAR REGRESSION', GREEN, [
        'Fits a straight line to predict Y from X.',
        '',
        'Model:   ŷᵢ = β₀ + β₁xᵢ + εᵢ',
        '',
        '  β₀ = intercept (Y when X=0)',
        '  β₁ = slope     (ΔY per unit ΔX)',
        '  εᵢ = residual  (ŷᵢ − yᵢ)',
        '',
        'Least Squares Estimation (OLS):',
        'Minimize:  SSE = Σ(yᵢ − ŷᵢ)²',
        '',
        '          Σ(xᵢ−x̄)(yᵢ−ȳ)',
        '  β₁ =  ───────────────────',
        '          Σ(xᵢ−x̄)²',
        '',
        '  β₀ =  ȳ − β₁x̄',
        '',
        'Goodness of fit:',
        '  SST = Σ(yᵢ−ȳ)²      (total variance)',
        '  SSR = Σ(ŷᵢ−ȳ)²      (explained)',
        '  SSE = Σ(yᵢ−ŷᵢ)²     (unexplained)',
        '',
        '  R² = SSR/SST  =  1 − SSE/SST',
        '  R² ∈ [0,1]  →  1 = perfect fit',
        '',
        'OLS Assumptions (LINE):',
        '  L - Linearity of relationship',
        '  I - Independence of errors',
        '  N - Normality of residuals',
        '  E - Equal variance (homoscedasticity)',
    ]),
    ('T-TEST & ANOVA', AMBER, [
        'Compare means across groups.',
        '',
        'INDEPENDENT SAMPLES T-TEST:',
        '  H₀: μ₁ = μ₂  (two group means equal)',
        '',
        '       x̄₁ − x̄₂',
        '  t = ─────────────────────────────',
        '      sₚ √(1/n₁ + 1/n₂)',
        '',
        '  Pooled SD:  sₚ = √[(s₁²(n₁-1) + s₂²(n₂-1)) / (n₁+n₂-2)]',
        '  df = n₁ + n₂ − 2',
        '  Reject H₀ if p < α (usually 0.05)',
        '',
        'Effect size (Cohen\'s d):',
        '  d = (x̄₁ − x̄₂) / sₚ',
        '  d = 0.2 small, 0.5 medium, 0.8 large',
        '',
        'ONE-WAY ANOVA (k ≥ 3 groups):',
        '  H₀: μ₁ = μ₂ = ... = μₖ',
        '',
        '  SSB = Σnⱼ(x̄ⱼ − x̄)²   (between groups)',
        '  SSW = ΣΣ(xᵢⱼ − x̄ⱼ)²  (within groups)',
        '',
        '  F = [SSB/(k−1)] / [SSW/(N−k)]',
        '    = MS_between / MS_within',
        '',
        '  F ~ F-distribution (df₁=k−1, df₂=N−k)',
        '',
        'Post-hoc (after significant F):',
        '  Tukey HSD, Bonferroni correction',
    ]),
    ('CHI-SQUARE TEST  (χ²)', PURPLE, [
        'Tests independence of two CATEGORICAL variables.',
        '',
        'Setup: Create contingency table (r × c)',
        '',
        '  Oᵢⱼ = observed frequency in cell (i,j)',
        '         rowᵢ_total × colⱼ_total',
        '  Eᵢⱼ = ────────────────────────────',
        '                grand_total',
        '',
        'Test statistic:',
        '',
        '          (Oᵢⱼ − Eᵢⱼ)²',
        '  χ² = Σ ─────────────────',
        '               Eᵢⱼ',
        '',
        '  df = (rows − 1)(cols − 1)',
        '  Reject H₀ if χ² > χ²_critical or p < α',
        '',
        'Requirement: Eᵢⱼ ≥ 5 in each cell.',
        '  (Use Fisher\'s exact test if violated)',
        '',
        'Effect size — Cramér\'s V:',
        '  V = √[χ² / (n · min(r−1, c−1))]',
        '  V ∈ [0, 1]  →  0 = no association',
        '',
        'LOGISTIC REGRESSION (binary outcome):',
        '  log[p/(1−p)] = β₀ + β₁x',
        '  p = 1 / [1 + e^(−(β₀+β₁x))]',
        '  Interpret: exp(β₁) = odds ratio',
    ]),
]

col_tops = [0.73, 0.73, 0.73, 0.73]
col_xs   = [0.01, 0.26, 0.51, 0.76]
for (title, color, lines), cx in zip(techniques, col_xs):
    ax_def.text(cx, 0.75, title, transform=ax_def.transAxes,
                fontsize=9.5, fontweight='bold', color=color)
    for i, line in enumerate(lines):
        ax_def.text(cx, 0.73 - i*0.022, line,
                    transform=ax_def.transAxes, fontsize=7.8, color=DARK,
                    fontfamily='monospace' if any(c in line for c in ['=','→','─','Σ','+','−','·','√','H₀']) else 'sans-serif',
                    va='top')

# ─── WORKED EXAMPLE (bivariate) ───────────────────────────────────────────────
ax_ex = fig2.add_subplot(gs2[1, :])
ax_ex.axis('off')

hdr2 = FancyBboxPatch((0, 0.88), 1.0, 0.10, boxstyle="round,pad=0.01",
                       lw=0, facecolor='#EAF5EE', transform=ax_ex.transAxes)
ax_ex.add_patch(hdr2)
ax_ex.text(0.5, 0.96, '✏️  WORKED EXAMPLE — Study Hours vs Exam Score (n=20)',
           transform=ax_ex.transAxes, ha='center', fontsize=12,
           fontweight='bold', color=GREEN)

# Show data table
ax_ex.text(0.01, 0.84, 'X = Study Hours:', transform=ax_ex.transAxes,
           fontsize=9, fontweight='bold', color=BLUE)
ax_ex.text(0.16, 0.84, str(hours.tolist()), transform=ax_ex.transAxes,
           fontsize=8.5, fontfamily='monospace', color=BLUE)
ax_ex.text(0.01, 0.78, 'Y = Exam Scores:', transform=ax_ex.transAxes,
           fontsize=9, fontweight='bold', color=GREEN)
ax_ex.text(0.16, 0.78, str(scores.tolist()), transform=ax_ex.transAxes,
           fontsize=8.5, fontfamily='monospace', color=GREEN)

# Calculations
x_bar = np.mean(hours); y_bar = np.mean(scores)
sxy   = np.sum((hours-x_bar)*(scores-y_bar))
sxx   = np.sum((hours-x_bar)**2)
syy   = np.sum((scores-y_bar)**2)
beta1 = sxy / sxx
beta0 = y_bar - beta1 * x_bar
r_val = sxy / np.sqrt(sxx * syy)
r2    = r_val**2
n_pts = len(hours)
t_stat_r = r_val * np.sqrt(n_pts-2) / np.sqrt(1-r_val**2)
p_r   = 2 * stats.t.sf(abs(t_stat_r), df=n_pts-2)
sst   = syy
y_pred_all = beta0 + beta1 * hours
sse   = np.sum((scores - y_pred_all)**2)
ssr   = sst - sse
se_beta1 = np.sqrt(sse/(n_pts-2)/sxx)
t_b1  = beta1 / se_beta1
p_b1  = 2 * stats.t.sf(abs(t_b1), df=n_pts-2)

calc_steps = [
    ('STEP 1 — Compute means', BLUE,
     f'x̄ = {x_bar:.2f} hours   ȳ = {y_bar:.2f} points'),
    ('STEP 2 — Cross-products Σ(x−x̄)(y−ȳ)', GREEN,
     f'Sxy = {sxy:.2f}   Sxx = {sxx:.2f}   Syy = {syy:.2f}'),
    ('STEP 3 — Pearson r', BLUE,
     f'r = Sxy / √(Sxx·Syy) = {sxy:.2f} / √({sxx:.2f}·{syy:.2f}) = {r_val:.4f}'),
    ('STEP 4 — Significance of r', AMBER,
     f't = r√(n−2)/√(1−r²) = {t_stat_r:.3f}  |  df={n_pts-2}  |  p = {p_r:.6f}  → {"Significant ✓" if p_r<0.05 else "Not sig."}'),
    ('STEP 5 — Regression slope β₁', GREEN,
     f'β₁ = Sxy/Sxx = {sxy:.2f}/{sxx:.2f} = {beta1:.4f}  (each extra hour → +{beta1:.2f} points)'),
    ('STEP 6 — Intercept β₀', GREEN,
     f'β₀ = ȳ − β₁x̄ = {y_bar:.2f} − {beta1:.4f}×{x_bar:.2f} = {beta0:.4f}'),
    ('STEP 7 — Model equation', PURPLE,
     f'ŷ = {beta0:.2f} + {beta1:.2f}·x   →   Predict 8hrs: ŷ = {beta0:.2f}+{beta1:.2f}×8 = {beta0+beta1*8:.1f}'),
    ('STEP 8 — R² & Partition', RED,
     f'SST={sst:.1f}  SSR={ssr:.1f}  SSE={sse:.1f}  →  R²={r2:.4f}  ({r2*100:.1f}% of variance explained)'),
]

for i, (label, color, text) in enumerate(calc_steps):
    y_pos = 0.71 - i * 0.082
    bg = FancyBboxPatch((0, y_pos-0.005), 1.0, 0.072,
                         boxstyle="round,pad=0.005", lw=0.8,
                         edgecolor=color, facecolor=BG,
                         transform=ax_ex.transAxes)
    ax_ex.add_patch(bg)
    ax_ex.text(0.01, y_pos+0.055, label, transform=ax_ex.transAxes,
               fontsize=8.5, fontweight='bold', color=color)
    ax_ex.text(0.20, y_pos+0.055, text, transform=ax_ex.transAxes,
               fontsize=8.5, color=DARK, fontfamily='monospace')

# ─── PLOTS ────────────────────────────────────────────────────────────────────
ax_scat  = fig2.add_subplot(gs2[2, 0])
ax_resid = fig2.add_subplot(gs2[2, 1])
ax_corr  = fig2.add_subplot(gs2[3, 0])
ax_ttst  = fig2.add_subplot(gs2[3, 1])
ax_chi   = fig2.add_subplot(gs2[4, 0])
ax_logr  = fig2.add_subplot(gs2[4, 1])
ax_code2 = fig2.add_subplot(gs2[5, :])

# Scatter + regression
ax_scat.scatter(hours, scores, color=BLUE, s=80, alpha=0.75, edgecolor='white', zorder=5)
x_line = np.linspace(hours.min()-0.5, hours.max()+0.5, 100)
y_line = beta0 + beta1 * x_line
ax_scat.plot(x_line, y_line, color=RED, lw=2.5, zorder=4)
# Residual lines
for xi, yi in zip(hours, scores):
    ax_scat.plot([xi, xi], [yi, beta0+beta1*xi], color=AMBER, lw=0.8, alpha=0.7)
ax_scat.set_title(f'Scatter + OLS Regression\nŷ = {beta0:.1f} + {beta1:.1f}x   r={r_val:.3f}   R²={r2:.3f}')
ax_scat.set_xlabel('Study Hours'); ax_scat.set_ylabel('Exam Score')
ax_scat.text(2, 92, f'Each line = residual εᵢ', fontsize=8, color=AMBER)

# Residual plot
residuals = scores - (beta0 + beta1*hours)
ax_resid.scatter(y_pred_all, residuals, color=BLUE, s=60, alpha=0.75, edgecolor='white')
ax_resid.axhline(0, color=RED, lw=2, ls='--')
ax_resid.set_title('Residual Plot (Model Diagnostic)\nWant: random scatter around 0')
ax_resid.set_xlabel('Fitted Values ŷ'); ax_resid.set_ylabel('Residuals e = y − ŷ')
ax_resid.text(0.5, 0.92, 'Funnel shape = heteroscedasticity ✗\nRandom cloud = good fit ✓',
              transform=ax_resid.transAxes, ha='center', fontsize=8, va='top')

# Correlation heatmap (student features)
fake_df = pd.DataFrame({'Score':scores,'Hours':hours,
                         'Sleep':np.random.normal(7,1,n).round(1),
                         'Attend':np.clip(scores/100+np.random.uniform(0,0.2,n),0.5,1)})
corr = fake_df.corr()
import seaborn as sns
sns.heatmap(corr, ax=ax_corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5, square=True,
            cbar_kws={'shrink':0.8}, annot_kws={'size':10})
ax_corr.set_title('Pearson Correlation Matrix\n(All numeric feature pairs)')
ax_corr.tick_params(axis='x', rotation=20)

# T-test
g1 = scores[hours <= 6]; g2 = scores[hours > 6]
t_val, p_ttest = stats.ttest_ind(g1, g2)
ax_ttst.boxplot([g1, g2], patch_artist=True, widths=0.4,
                medianprops=dict(color='white', lw=2.5),
                boxprops=dict(alpha=0.6),)
for patch, color in zip(ax_ttst.patches[:2], [BLUE, GREEN]):
    patch.set_facecolor(color)
for xi, val in zip([1]*len(g1) + [2]*len(g2), list(g1)+list(g2)):
    ax_ttst.scatter(xi + np.random.uniform(-0.1,0.1), val,
                    color=BLUE if xi==1 else GREEN, s=12, alpha=0.4, edgecolor='none')
y_max = max(scores.max(), 102)
ax_ttst.plot([1,1,2,2],[y_max-1,y_max+1,y_max+1,y_max-1], color=DARK, lw=1)
sig = '***' if p_ttest<0.001 else ('**' if p_ttest<0.01 else ('*' if p_ttest<0.05 else 'ns'))
ax_ttst.text(1.5, y_max+1.5, f'{sig}  t={t_val:.2f}, p={p_ttest:.4f}',
             ha='center', fontsize=9, fontweight='bold')
ax_ttst.set_xticklabels([f'Low study\n≤6 hrs\n(n={len(g1)})',
                          f'High study\n>6 hrs\n(n={len(g2)})'])
ax_ttst.set_title('Independent T-Test\nMean score: low vs high study hours')
ax_ttst.set_ylabel('Score')
ax_ttst.text(0.5, 0.03,
             f'd = {abs(g2.mean()-g1.mean())/np.sqrt(((len(g1)-1)*g1.std()**2+(len(g2)-1)*g2.std()**2)/(len(g1)+len(g2)-2)):.2f}  (Cohen\'s d)',
             transform=ax_ttst.transAxes, ha='center', fontsize=8.5,
             bbox=dict(boxstyle='round', fc='#EAF0F8', ec=BLUE))

# Chi-square
np.random.seed(0)
cat1 = np.random.choice(['Pass','Fail'], n, p=[0.7, 0.3])
cat2 = np.random.choice(['HighStudy','LowStudy'], n, p=[0.55, 0.45])
ct = pd.crosstab(cat1, cat2)
chi2_val, p_chi, dof, expected = stats.chi2_contingency(ct)
sns.heatmap(ct, ax=ax_chi, annot=True, fmt='d', cmap='Blues',
            linewidths=1, cbar_kws={'shrink':0.7})
ax_chi.set_title(f'Chi-Square Contingency Table\nχ²={chi2_val:.2f}  df={dof}  p={p_chi:.4f}')
ax_chi.set_xlabel('Study Level'); ax_chi.set_ylabel('Outcome')
cramer_v = np.sqrt(chi2_val / (n * min(ct.shape[0]-1, ct.shape[1]-1)))
ax_chi.text(0.5, -0.18, f"Cramér's V = {cramer_v:.3f}  (effect size)",
            transform=ax_chi.transAxes, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', fc='#EAF0F8', ec=BLUE))

# Logistic regression
np.random.seed(42)
log_x = np.linspace(0, 12, 200)
log_b0, log_b1 = -4.5, 0.7
log_p = 1 / (1 + np.exp(-(log_b0 + log_b1*log_x)))
ax_logr.plot(log_x, log_p, color=PURPLE, lw=2.5)
binary_y = (scores > 75).astype(int)
ax_logr.scatter(hours, binary_y + np.random.uniform(-0.04,0.04,n),
                color=BLUE, s=40, alpha=0.65, edgecolor='white', zorder=5)
ax_logr.axhline(0.5, color=AMBER, lw=1.5, ls='--', label='Decision boundary (p=0.5)')
ax_logr.set_title('Logistic Regression\nP(Pass) from Study Hours')
ax_logr.set_xlabel('Study Hours'); ax_logr.set_ylabel('P(Score > 75)')
ax_logr.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_logr.legend(fontsize=8)
ax_logr.text(0.02, 0.55,
             f'log[p/(1−p)] = β₀ + β₁x\nOR = exp(β₁) = exp({log_b1}) = {np.exp(log_b1):.2f}',
             transform=ax_logr.transAxes, fontsize=8.5,
             bbox=dict(boxstyle='round', fc='#F5F0FF', ec=PURPLE))

# Code block
ax_code2.axis('off')
cbg = FancyBboxPatch((0,0), 1.0, 1.0, boxstyle="round,pad=0.01", lw=1.2,
                      edgecolor='#AAAAAA', facecolor='#F5F5F0',
                      transform=ax_code2.transAxes)
ax_code2.add_patch(cbg)
ax_code2.text(0.5, 0.97, '💻  BIVARIATE CODE REFERENCE',
              transform=ax_code2.transAxes, ha='center', fontsize=11,
              fontweight='bold', color=DARK)

code2 = [
    (0.01, 0.90, BLUE, '# ── Correlation ──────────────────────────────'),
    (0.01, 0.83, DARK, 'r, p = stats.pearsonr(x, y)'),
    (0.01, 0.76, DARK, 'rho, p = stats.spearmanr(x, y)'),
    (0.01, 0.69, DARK, 'df.corr()                   # full matrix'),
    (0.01, 0.62, BLUE, '# ── Regression ───────────────────────────────'),
    (0.01, 0.55, DARK, 'm,b,r,p,se = stats.linregress(x, y)'),
    (0.01, 0.48, DARK, 'from sklearn.linear_model import LinearRegression'),
    (0.01, 0.41, DARK, 'reg = LinearRegression().fit(X,y)'),
    (0.01, 0.34, DARK, 'print(reg.coef_, reg.intercept_, reg.score(X,y))'),
    (0.01, 0.27, BLUE, '# ── T-test ────────────────────────────────────'),
    (0.01, 0.20, DARK, 't, p = stats.ttest_ind(group1, group2)'),
    (0.01, 0.13, DARK, 't, p = stats.ttest_rel(before, after) # paired'),
    (0.01, 0.06, DARK, 'f, p = stats.f_oneway(g1, g2, g3)   # ANOVA'),
    (0.51, 0.90, BLUE, '# ── Chi-Square ───────────────────────────────'),
    (0.51, 0.83, DARK, 'ct = pd.crosstab(df.col1, df.col2)'),
    (0.51, 0.76, DARK, 'chi2, p, dof, exp = stats.chi2_contingency(ct)'),
    (0.51, 0.69, DARK, 'sns.heatmap(ct, annot=True, fmt="d")'),
    (0.51, 0.62, BLUE, '# ── Logistic Regression ──────────────────────'),
    (0.51, 0.55, DARK, 'from sklearn.linear_model import LogisticRegression'),
    (0.51, 0.48, DARK, 'clf = LogisticRegression().fit(X, y_binary)'),
    (0.51, 0.41, DARK, 'proba = clf.predict_proba(X)[:,1]'),
    (0.51, 0.34, BLUE, '# ── Visualizations ───────────────────────────'),
    (0.51, 0.27, DARK, 'sns.regplot(x="X", y="Y", data=df)'),
    (0.51, 0.20, DARK, 'sns.residplot(x="X", y="Y", data=df)'),
    (0.51, 0.13, DARK, 'sns.boxplot(x="cat", y="num", data=df)'),
    (0.51, 0.06, DARK, 'sns.scatterplot(x,y,hue="group", data=df)'),
]
for cx, cy, color, text in code2:
    ax_code2.text(cx+0.01, cy, text, transform=ax_code2.transAxes,
                  fontsize=8.2, color=color, fontfamily='monospace', va='top')

fig2.savefig('/mnt/user-data/outputs/THEORY_2_Bivariate.png', bbox_inches='tight', dpi=130)
print("✓ Page 2 saved — Bivariate Theory & Math")
plt.close(fig2)


# =============================================================================
# PAGE 3 — MULTIVARIATE: THEORY + MATH
# =============================================================================
fig3 = plt.figure(figsize=(20, 28))
fig3.patch.set_facecolor(BG)
fig3.suptitle('MULTIVARIATE ANALYSIS — Complete Theory & Mathematics',
              fontsize=18, fontweight='bold', y=0.985, color=DARK)

gs3 = gridspec.GridSpec(6, 2, figure=fig3, hspace=0.14, wspace=0.08,
                        top=0.96, bottom=0.02, left=0.04, right=0.96)

ax_mv_def = fig3.add_subplot(gs3[0, :])
ax_mv_def.axis('off')

bnr3 = FancyBboxPatch((0, 0.82), 1.0, 0.16, boxstyle="round,pad=0.01", lw=0,
                       facecolor='#EAF0F8', transform=ax_mv_def.transAxes)
ax_mv_def.add_patch(bnr3)
ax_mv_def.text(0.5, 0.96, '📊  WHAT IS MULTIVARIATE ANALYSIS?',
               transform=ax_mv_def.transAxes, ha='center', fontsize=13,
               fontweight='bold', color=BLUE)
ax_mv_def.text(0.5, 0.88,
               'Multivariate analysis handles THREE OR MORE variables simultaneously.\n'
               'Goal: discover hidden structure, build predictive models, or control for confounders.',
               transform=ax_mv_def.transAxes, ha='center', fontsize=10.5, color=DARK, linespacing=1.6)

mv_techniques = [
    ('MULTIPLE LINEAR REGRESSION', BLUE, [
        'Predict continuous Y from p predictors.',
        '',
        'Model (matrix form):',
        '  y = Xβ + ε',
        '',
        '  y: n×1 outcome vector',
        '  X: n×(p+1) design matrix',
        '  β: (p+1)×1 coefficient vector',
        '  ε: n×1 error vector ~ N(0,σ²I)',
        '',
        'OLS solution (Normal Equations):',
        '  β̂ = (XᵀX)⁻¹ Xᵀy',
        '',
        '  β̂ₖ = change in Ŷ per unit increase',
        '       in Xₖ, HOLDING all others fixed.',
        '',
        'Goodness of fit:',
        '  R² = 1 − SSE/SST',
        '  Adj-R² = 1 − (1−R²)(n−1)/(n−p−1)',
        '  Penalizes extra predictors.',
        '',
        'Overall F-test:',
        '  F = (R²/p) / ((1−R²)/(n−p−1))',
        '  Tests H₀: all β₁...βₚ = 0',
        '',
        'Multicollinearity check:',
        '  VIF = 1/(1−R²ⱼ)',
        '  VIF > 10 → problematic',
        '',
        'Regularization (prevents overfitting):',
        '  Ridge: minimize SSE + λΣβₖ²',
        '  Lasso: minimize SSE + λΣ|βₖ|',
    ]),
    ('PCA — PRINCIPAL COMPONENT ANALYSIS', GREEN, [
        'Reduce dimensionality while maximizing',
        'preserved variance.',
        '',
        'Steps:',
        '1. Standardize: Z = (X − μ) / σ',
        '',
        '2. Compute covariance matrix:',
        '   Σ = (1/n) ZᵀZ',
        '',
        '3. Eigendecomposition:',
        '   Σvᵢ = λᵢvᵢ',
        '   λ = eigenvalues (variance explained)',
        '   v = eigenvectors (principal components)',
        '',
        '4. Sort by λ₁ ≥ λ₂ ≥ ... ≥ λₚ',
        '',
        '5. Project data:',
        '   Z_new = Z · V_k',
        '   (V_k = top k eigenvectors)',
        '',
        'Explained variance ratio:',
        '  EVRᵢ = λᵢ / Σλⱼ × 100%',
        '',
        'Scree plot: plot EVR vs component.',
        'Choose k at the "elbow".',
        '',
        'Biplot: overlay loading vectors',
        '(eigenvectors) on score scatter.',
        '',
        'When to use: p > 10 features,',
        'high multicollinearity, visualization.',
    ]),
    ('K-MEANS CLUSTERING', AMBER, [
        'Partition n observations into k clusters',
        'by minimizing within-cluster variance.',
        '',
        'Objective function:',
        '',
        '  J = Σₖ Σ_{x∈Cₖ} ‖x − μₖ‖²',
        '',
        '  μₖ = centroid of cluster k',
        '  ‖·‖ = Euclidean distance',
        '',
        'Algorithm (Lloyd\'s):',
        '1. Initialize k centroids randomly',
        '2. Assign each point to nearest μₖ',
        '3. Update μₖ = mean of assigned pts',
        '4. Repeat 2-3 until convergence',
        '',
        'Choosing k:',
        '  Elbow: plot J vs k, find bend.',
        '  Silhouette: s = (b−a)/max(a,b)',
        '    a = avg within-cluster dist',
        '    b = avg nearest-cluster dist',
        '    s ≈ 1 → well-clustered',
        '',
        'Hierarchical clustering:',
        '  Bottom-up (agglomerative):',
        '  1. Each pt = its own cluster',
        '  2. Merge two closest clusters',
        '  3. Repeat until one cluster',
        '  Visualize as dendrogram.',
        '',
        'Linkage: single, complete, average,',
        'Ward (minimizes within-SS).',
    ]),
    ('FACTOR ANALYSIS & LDA', PURPLE, [
        'FACTOR ANALYSIS:',
        'Model observed X from latent factors F.',
        '',
        '  X = ΛF + ε',
        '',
        '  Λ = factor loadings matrix (p×m)',
        '  F = latent factors (m < p)',
        '  ε = unique (error) factors',
        '',
        'Factor loading: λᵢⱼ = correlation',
        '  of variable i with factor j.',
        '',
        'Rotation (Varimax): maximize variance',
        '  of squared loadings per factor.',
        '  → Each variable loads high on one',
        '    factor only (simple structure).',
        '',
        '─────────────────────────────────',
        '',
        'LDA (LINEAR DISCRIMINANT ANALYSIS):',
        'Find projection maximizing class',
        'separability.',
        '',
        '  Maximize: J(w) = wᵀSBw / wᵀSWw',
        '',
        '  SB = between-class scatter',
        '  SW = within-class scatter',
        '',
        '  Solution: eigenvectors of SW⁻¹SB',
        '',
        'Assumes: multivariate normality,',
        '  equal covariance matrices.',
        '',
        'QDA: allows unequal covariances.',
    ]),
]

for (title, color, lines), cx in zip(mv_techniques, [0.01, 0.26, 0.51, 0.76]):
    ax_mv_def.text(cx, 0.75, title, transform=ax_mv_def.transAxes,
                   fontsize=9.5, fontweight='bold', color=color)
    for i, line in enumerate(lines):
        ax_mv_def.text(cx, 0.73 - i*0.022, line,
                       transform=ax_mv_def.transAxes, fontsize=7.8, color=DARK,
                       fontfamily='monospace' if any(c in line for c in ['=','→','λ','β','Σ','‖','·','√','ε','ₖ']) else 'sans-serif',
                       va='top')

# ─── WORKED EXAMPLE (multivariate) ───────────────────────────────────────────
ax_mv_ex = fig3.add_subplot(gs3[1, :])
ax_mv_ex.axis('off')

hdr3 = FancyBboxPatch((0, 0.88), 1.0, 0.10, boxstyle="round,pad=0.01", lw=0,
                       facecolor='#EAF5EE', transform=ax_mv_ex.transAxes)
ax_mv_ex.add_patch(hdr3)
ax_mv_ex.text(0.5, 0.96,
              '✏️  WORKED EXAMPLE — Multiple Regression: Predict Score from Hours + Sleep + Attendance',
              transform=ax_mv_ex.transAxes, ha='center', fontsize=12,
              fontweight='bold', color=GREEN)

# Build multivariate dataset
np.random.seed(42)
X_mv = np.column_stack([
    hours,
    np.clip(np.random.normal(7, 1, n), 4, 10),   # sleep
    np.clip(scores/100 + np.random.uniform(0,0.15,n), 0.5, 1.0)  # attendance
])
scaler3 = StandardScaler()
X_mv_sc = scaler3.fit_transform(X_mv)
X_with_int = np.hstack([np.ones((n,1)), X_mv_sc])
beta_ols = np.linalg.lstsq(X_with_int, scores, rcond=None)[0]
y_hat_mv = X_with_int @ beta_ols
ss_res = np.sum((scores - y_hat_mv)**2)
ss_tot = np.sum((scores - scores.mean())**2)
r2_mv  = 1 - ss_res/ss_tot
adj_r2 = 1 - (1-r2_mv)*(n-1)/(n-3-1)

mv_steps = [
    ('DATA', BLUE,
     f'X₁=Study Hrs, X₂=Sleep Hrs, X₃=Attendance  |  Y=Exam Score  |  n={n}  (all standardized)'),
    ('MODEL', GREEN,
     'ŷ = β₀ + β₁·StudyHrs + β₂·SleepHrs + β₃·Attendance'),
    ('β̂ = (XᵀX)⁻¹Xᵀy', AMBER,
     f'β̂ = [{", ".join(f"{b:.3f}" for b in beta_ols)}]  (intercept, study, sleep, attendance)'),
    ('INTERPRETATION', PURPLE,
     f'β₁={beta_ols[1]:.3f}: +1 SD study hours → +{beta_ols[1]:.2f} points (holding sleep & attendance constant)'),
    ('GOODNESS OF FIT', RED,
     f'SST={ss_tot:.1f}  SSE={ss_res:.1f}  →  R²={r2_mv:.4f}  Adj-R²={adj_r2:.4f}  ({r2_mv*100:.1f}% of variance explained by 3 predictors)'),
    ('PREDICTION', TEAL,
     f'New student: 9hrs study, 8hrs sleep, 90% attendance → standardize → ŷ = {beta_ols[0]+beta_ols[1]*0.8+beta_ols[2]*0.5+beta_ols[3]*0.8:.1f} points'),
]

for i, (label, color, text) in enumerate(mv_steps):
    y_pos = 0.77 - i * 0.095
    bg = FancyBboxPatch((0, y_pos-0.005), 1.0, 0.083,
                         boxstyle="round,pad=0.005", lw=0.8,
                         edgecolor=color, facecolor=BG,
                         transform=ax_mv_ex.transAxes)
    ax_mv_ex.add_patch(bg)
    ax_mv_ex.text(0.01, y_pos+0.063, label, transform=ax_mv_ex.transAxes,
                  fontsize=8.5, fontweight='bold', color=color)
    ax_mv_ex.text(0.17, y_pos+0.063, text, transform=ax_mv_ex.transAxes,
                  fontsize=8.5, color=DARK, fontfamily='monospace')

# ─── PLOTS ────────────────────────────────────────────────────────────────────
ax_mlr   = fig3.add_subplot(gs3[2, 0])
ax_coef  = fig3.add_subplot(gs3[2, 1])
ax_pca_s = fig3.add_subplot(gs3[3, 0])
ax_pca_b = fig3.add_subplot(gs3[3, 1])
ax_clust = fig3.add_subplot(gs3[4, 0])
ax_elbow = fig3.add_subplot(gs3[4, 1])
ax_code3 = fig3.add_subplot(gs3[5, :])

# Multiple regression actual vs predicted
ax_mlr.scatter(scores, y_hat_mv, color=BLUE, s=70, alpha=0.75, edgecolor='white', zorder=5)
mn, mx = min(scores.min(), y_hat_mv.min()), max(scores.max(), y_hat_mv.max())
ax_mlr.plot([mn,mx],[mn,mx], color=RED, lw=2, ls='--', label='Perfect fit')
ax_mlr.set_title(f'Multiple Regression: Actual vs Predicted\nR²={r2_mv:.3f}   Adj-R²={adj_r2:.3f}')
ax_mlr.set_xlabel('Actual Score'); ax_mlr.set_ylabel('Predicted Score')
ax_mlr.legend(fontsize=9)

# Coefficient plot
feat_names = ['Intercept','Study Hrs','Sleep Hrs','Attendance']
colors_c   = [LIGHT if i==0 else (GREEN if c>0 else RED) for i,c in enumerate(beta_ols)]
ax_coef.barh(feat_names, beta_ols, color=colors_c, alpha=0.8, edgecolor='white')
ax_coef.axvline(0, color=DARK, lw=1)
ax_coef.set_title('Regression Coefficients\n(Standardized Predictors)')
ax_coef.set_xlabel('Coefficient (β)')
for i, v in enumerate(beta_ols):
    ax_coef.text(v + (0.05 if v>=0 else -0.05), i, f'{v:.2f}',
                 va='center', fontsize=9, fontweight='bold')

# PCA on iris
from sklearn.datasets import load_iris
iris = load_iris()
iris_X = iris.data
iris_X_sc = StandardScaler().fit_transform(iris_X)
pca_full = PCA().fit(iris_X_sc)
evr = pca_full.explained_variance_ratio_ * 100
ax_pca_s.bar(range(1,5), evr, color=BLUE, alpha=0.75, edgecolor='white', label='Individual')
ax_pca_s2 = ax_pca_s.twinx()
ax_pca_s2.plot(range(1,5), np.cumsum(evr), 'o-', color=RED, lw=2, ms=7, label='Cumulative')
ax_pca_s2.axhline(95, color=LIGHT, lw=1, ls='--')
ax_pca_s2.set_ylabel('Cumulative %', fontsize=9); ax_pca_s2.set_ylim(0, 115)
ax_pca_s.set_title('PCA Scree Plot (Iris)\nEigenvalues → Explained Variance')
ax_pca_s.set_xlabel('PC'); ax_pca_s.set_ylabel('Explained Variance %')
ax_pca_s.set_xticks(range(1,5))
for i, (e, cum) in enumerate(zip(evr, np.cumsum(evr))):
    ax_pca_s.text(i+1, e+0.5, f'{e:.0f}%', ha='center', fontsize=8.5,
                  fontweight='bold', color=BLUE)

# PCA biplot
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(iris_X_sc)
colors_sp = [BLUE, GREEN, RED]
for i, sp in enumerate(iris.target_names):
    mask = iris.target == i
    ax_pca_b.scatter(X_pca[mask,0], X_pca[mask,1], color=colors_sp[i],
                     s=35, alpha=0.7, edgecolor='white', lw=0.4, label=sp)
scale = 3.0
for j, feat in enumerate(iris.feature_names):
    ax_pca_b.annotate('', xy=(pca2.components_[0,j]*scale, pca2.components_[1,j]*scale),
                      xytext=(0,0),
                      arrowprops=dict(arrowstyle='->', color='#333', lw=1.8))
    ax_pca_b.text(pca2.components_[0,j]*scale*1.15, pca2.components_[1,j]*scale*1.15,
                  feat.replace(' (cm)',''), fontsize=7.5, color='#333', ha='center')
ax_pca_b.set_title(f'PCA Biplot (Iris)\nPC1={pca2.explained_variance_ratio_[0]*100:.0f}%  PC2={pca2.explained_variance_ratio_[1]*100:.0f}%')
ax_pca_b.set_xlabel('PC1'); ax_pca_b.set_ylabel('PC2')
ax_pca_b.legend(fontsize=8.5)
ax_pca_b.text(0.02, 0.06, 'Arrows = loading vectors\n(direction a feature "points" in PC space)',
              transform=ax_pca_b.transAxes, fontsize=8, va='bottom',
              bbox=dict(boxstyle='round', fc='#F5F5F0', ec=LIGHT))

# K-Means
km = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = km.fit_predict(iris_X_sc)
for k in range(3):
    mask = labels == k
    ax_clust.scatter(X_pca[mask,0], X_pca[mask,1], color=colors_sp[k],
                     s=35, alpha=0.7, edgecolor='white', lw=0.4, label=f'Cluster {k+1}')
cents = pca2.transform(km.cluster_centers_)
ax_clust.scatter(cents[:,0], cents[:,1], s=200, marker='*', color='black',
                 zorder=6, label='Centroids')
ax_clust.set_title('K-Means Clustering (k=3)\nProjected onto PCA space')
ax_clust.set_xlabel('PC1'); ax_clust.set_ylabel('PC2')
ax_clust.legend(fontsize=8.5)

# Elbow
inertias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(iris_X_sc).inertia_
            for k in range(1, 9)]
ax_elbow.plot(range(1,9), inertias, 'o-', color=BLUE, lw=2, ms=7)
ax_elbow.axvline(3, color=RED, lw=2, ls='--', label='Elbow at k=3')
ax_elbow.set_title('K-Means Elbow Method\n(Choose k where inertia "elbows")')
ax_elbow.set_xlabel('K'); ax_elbow.set_ylabel('Inertia  J = Σ‖x−μₖ‖²')
ax_elbow.legend(fontsize=9)
ax_elbow.fill_between([2.6,3.4],[min(inertias)*0.95,min(inertias)*0.95],
                       [max(inertias)*1.02,max(inertias)*1.02], alpha=0.1, color=RED)

# Code block
ax_code3.axis('off')
cbg3 = FancyBboxPatch((0,0), 1.0, 1.0, boxstyle="round,pad=0.01", lw=1.2,
                       edgecolor='#AAAAAA', facecolor='#F5F5F0',
                       transform=ax_code3.transAxes)
ax_code3.add_patch(cbg3)
ax_code3.text(0.5, 0.97, '💻  MULTIVARIATE CODE REFERENCE',
              transform=ax_code3.transAxes, ha='center', fontsize=11,
              fontweight='bold', color=DARK)

code3 = [
    (0.01, 0.90, BLUE, '# ── Multiple Regression ────────────────────────────────────────'),
    (0.01, 0.83, DARK, 'from sklearn.linear_model import LinearRegression'),
    (0.01, 0.76, DARK, 'reg = LinearRegression().fit(X_scaled, y)'),
    (0.01, 0.69, DARK, 'print(reg.coef_, reg.intercept_, reg.score(X,y))'),
    (0.01, 0.62, DARK, '# OLS directly: beta = np.linalg.lstsq(X_with_intercept, y)'),
    (0.01, 0.55, BLUE, '# ── PCA ─────────────────────────────────────────────────────────'),
    (0.01, 0.48, DARK, 'from sklearn.decomposition import PCA'),
    (0.01, 0.41, DARK, 'from sklearn.preprocessing import StandardScaler'),
    (0.01, 0.34, DARK, 'X_sc = StandardScaler().fit_transform(X)'),
    (0.01, 0.27, DARK, 'pca = PCA(n_components=2)'),
    (0.01, 0.20, DARK, 'X_pca = pca.fit_transform(X_sc)'),
    (0.01, 0.13, DARK, 'print(pca.explained_variance_ratio_)  # variance per PC'),
    (0.01, 0.06, DARK, 'print(pca.components_)                # eigenvectors (loadings)'),
    (0.51, 0.90, BLUE, '# ── K-Means Clustering ──────────────────────────────────────────'),
    (0.51, 0.83, DARK, 'from sklearn.cluster import KMeans'),
    (0.51, 0.76, DARK, 'km = KMeans(n_clusters=3, n_init=10, random_state=42)'),
    (0.51, 0.69, DARK, 'labels = km.fit_predict(X_scaled)'),
    (0.51, 0.62, DARK, 'print(km.inertia_, km.cluster_centers_)'),
    (0.51, 0.55, BLUE, '# ── LDA ─────────────────────────────────────────────────────────'),
    (0.51, 0.48, DARK, 'from sklearn.discriminant_analysis import LinearDiscriminantAnalysis'),
    (0.51, 0.41, DARK, 'lda = LinearDiscriminantAnalysis().fit(X, y_labels)'),
    (0.51, 0.34, DARK, 'X_lda = lda.transform(X)                # project'),
    (0.51, 0.27, BLUE, '# ── Seaborn multivariate visuals ────────────────────────────────'),
    (0.51, 0.20, DARK, 'sns.pairplot(df, hue="species", diag_kind="kde")'),
    (0.51, 0.13, DARK, 'sns.heatmap(df.corr(), annot=True, cmap="coolwarm")'),
    (0.51, 0.06, DARK, 'sns.clustermap(df, cmap="viridis", standard_scale=1)'),
]
for cx, cy, color, text in code3:
    ax_code3.text(cx+0.01, cy, text, transform=ax_code3.transAxes,
                  fontsize=8.2, color=color, fontfamily='monospace', va='top')

fig3.savefig('/mnt/user-data/outputs/THEORY_3_Multivariate.png', bbox_inches='tight', dpi=130)
print("✓ Page 3 saved — Multivariate Theory & Math")
plt.close(fig3)

print("\n" + "="*60)
print("ALL 3 THEORY PAGES SAVED")
print("="*60)
