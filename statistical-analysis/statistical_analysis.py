"""
=============================================================================
COMPLETE STATISTICAL ANALYSIS GUIDE
Theory + Math + Sample Datasets + Matplotlib/Seaborn Visualizations
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_iris, make_classification
import warnings
warnings.filterwarnings('ignore')

# ── Global Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#FAFAF8',
    'axes.facecolor':   '#FAFAF8',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.color':       '#E5E3DC',
    'grid.linewidth':   0.6,
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.titlesize':   12,
    'axes.titleweight': 'bold',
    'axes.labelsize':   10,
    'figure.dpi':       130,
})
COLORS = ['#3266AD','#E85D24','#1A9E75','#8E44AD','#E6AC00','#C0392B','#2980B9','#27AE60']
sns.set_theme(style="whitegrid", palette=COLORS)

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE DATASETS
# ─────────────────────────────────────────────────────────────────────────────

# Dataset 1: Student performance (univariate)
n_students = 200
exam_scores = np.concatenate([
    np.random.normal(72, 12, 160),
    np.random.normal(45, 8, 40)   # struggling students - creates slight left skew
])
exam_scores = np.clip(exam_scores, 20, 100)

student_df = pd.DataFrame({
    'score':        exam_scores,
    'study_hours':  np.clip(exam_scores/10 + np.random.normal(0,1,n_students), 1, 12),
    'gender':       np.random.choice(['Male','Female'], n_students, p=[0.48,0.52]),
    'school_type':  np.random.choice(['Public','Private','Charter'], n_students, p=[0.5,0.3,0.2]),
    'sleep_hours':  np.clip(np.random.normal(7, 1.2, n_students), 4, 10),
    'attendance':   np.clip(exam_scores/100 * 0.6 + np.random.uniform(0.3,0.5,n_students), 0.5, 1.0),
})

# Dataset 2: House prices (bivariate / multivariate)
n_houses = 300
sqft       = np.random.normal(1800, 600, n_houses).clip(500, 4000)
bedrooms   = np.random.choice([1,2,3,4,5], n_houses, p=[0.05,0.2,0.4,0.25,0.1])
age        = np.random.exponential(15, n_houses).clip(0, 80)
price      = (80 * sqft + 15000 * bedrooms - 1200 * age
              + np.random.normal(0, 30000, n_houses) + 50000).clip(80000)

house_df = pd.DataFrame({
    'price':    price,
    'sqft':     sqft,
    'bedrooms': bedrooms,
    'age':      age,
    'location': np.random.choice(['Urban','Suburban','Rural'], n_houses, p=[0.35,0.45,0.2]),
})

# Dataset 3: Iris (multivariate)
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("✓ Datasets created")
print(f"  student_df: {student_df.shape}  |  house_df: {house_df.shape}  |  iris_df: {iris_df.shape}")


# =============================================================================
# FIGURE 1 — UNIVARIATE ANALYSIS
# =============================================================================
fig1 = plt.figure(figsize=(18, 14))
fig1.suptitle('UNIVARIATE ANALYSIS\nOne Variable at a Time — Theory, Math & Visualization',
              fontsize=16, fontweight='bold', y=0.98, color='#1A1A1A')
gs = gridspec.GridSpec(3, 4, figure=fig1, hspace=0.55, wspace=0.42)

scores = student_df['score'].values

# ── 1A: Histogram + KDE ──────────────────────────────────────────────────────
ax1a = fig1.add_subplot(gs[0, :2])
ax1a.hist(scores, bins=25, color=COLORS[0], alpha=0.65, edgecolor='white',
          linewidth=0.5, density=True, label='Histogram (density)')
kde_x = np.linspace(scores.min()-5, scores.max()+5, 300)
kde = stats.gaussian_kde(scores, bw_method=0.35)
ax1a.plot(kde_x, kde(kde_x), color=COLORS[1], lw=2.2, label='KDE (kernel density)')
ax1a.axvline(np.mean(scores),   color='#E6AC00', lw=1.8, ls='--', label=f'Mean={np.mean(scores):.1f}')
ax1a.axvline(np.median(scores), color='#1A9E75', lw=1.8, ls='-.',  label=f'Median={np.median(scores):.1f}')
ax1a.axvline(stats.mode(scores, keepdims=True).mode[0], color='#8E44AD', lw=1.8, ls=':',
             label=f'Mode≈{stats.mode(scores,keepdims=True).mode[0]:.0f}')
ax1a.set_title('1A · Histogram + KDE\n(Frequency Distribution)')
ax1a.set_xlabel('Exam Score'); ax1a.set_ylabel('Density')
ax1a.legend(fontsize=8, framealpha=0.9)
# Math annotation
ax1a.text(0.98, 0.95,
    r'$\bar{x}=\frac{\sum x_i}{n}$  $\tilde{x}=$ middle value',
    transform=ax1a.transAxes, ha='right', va='top', fontsize=8.5,
    bbox=dict(boxstyle='round,pad=0.4', fc='#EFF3FB', ec=COLORS[0], alpha=0.9))

# ── 1B: Box Plot ──────────────────────────────────────────────────────────────
ax1b = fig1.add_subplot(gs[0, 2])
bp = ax1b.boxplot(scores, patch_artist=True, widths=0.5,
                  medianprops=dict(color='#E85D24', lw=2.5),
                  boxprops=dict(facecolor='#DDEEFF', color=COLORS[0]),
                  whiskerprops=dict(color=COLORS[0], lw=1.5),
                  capprops=dict(color=COLORS[0], lw=2),
                  flierprops=dict(marker='o', ms=4, color='#C0392B', alpha=0.6))
q1, q3 = np.percentile(scores, [25, 75])
ax1b.annotate(f'Q3={q3:.0f}', xy=(1.25, q3), fontsize=8, color='#444')
ax1b.annotate(f'Q1={q1:.0f}', xy=(1.25, q1), fontsize=8, color='#444')
ax1b.annotate(f'Median={np.median(scores):.0f}', xy=(1.25, np.median(scores)), fontsize=8.5, color='#E85D24', fontweight='bold')
ax1b.set_title('1B · Box Plot\n(5-Number Summary)')
ax1b.set_xticklabels(['Scores']); ax1b.set_ylabel('Score')
ax1b.text(0.5, 0.07, f'IQR = Q3−Q1 = {q3-q1:.0f}', transform=ax1b.transAxes,
          ha='center', fontsize=8.5, color='#1A1A1A',
          bbox=dict(boxstyle='round', fc='#FFF8E6', ec='#E6AC00'))

# ── 1C: Violin Plot ───────────────────────────────────────────────────────────
ax1c = fig1.add_subplot(gs[0, 3])
parts = ax1c.violinplot([scores], positions=[1], showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor(COLORS[2]); pc.set_alpha(0.6)
parts['cmeans'].set_color('#E85D24'); parts['cmedians'].set_color(COLORS[0])
ax1c.set_title('1C · Violin Plot\n(Distribution Shape)')
ax1c.set_xticks([1]); ax1c.set_xticklabels(['Scores']); ax1c.set_ylabel('Score')

# ── 1D: Measures of spread (bar chart summary) ───────────────────────────────
ax1d = fig1.add_subplot(gs[1, :2])
stats_names  = ['Mean', 'Median', 'Std Dev', 'Variance/10', 'IQR', 'Range/10']
stats_values = [np.mean(scores), np.median(scores), np.std(scores),
                np.var(scores)/10, q3-q1, (scores.max()-scores.min())/10]
bar_colors = [COLORS[0], COLORS[2], COLORS[1], COLORS[3], COLORS[4], COLORS[5]]
bars = ax1d.bar(stats_names, stats_values, color=bar_colors, alpha=0.85, edgecolor='white', width=0.65)
for bar, val in zip(bars, stats_values):
    ax1d.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val:.1f}',
              ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax1d.set_title('1D · Measures of Central Tendency & Spread')
ax1d.set_ylabel('Value'); ax1d.tick_params(axis='x', rotation=15)

# Math box
math_text = (r'Variance: $s^2=\frac{\sum(x_i-\bar{x})^2}{n-1}$'
             '\n'
             r'Std Dev: $s=\sqrt{s^2}$'
             '\n'
             r'IQR $= Q_3 - Q_1$')
ax1d.text(0.98, 0.97, math_text, transform=ax1d.transAxes,
          ha='right', va='top', fontsize=8.5, linespacing=1.7,
          bbox=dict(boxstyle='round,pad=0.5', fc='#EFF3FB', ec=COLORS[0], alpha=0.9))

# ── 1E: Skewness & Kurtosis visual ────────────────────────────────────────────
ax1e = fig1.add_subplot(gs[1, 2:])
x_range = np.linspace(-4, 4, 400)
# Symmetric
ax1e.plot(x_range, stats.norm.pdf(x_range), color=COLORS[0], lw=2, label='Symmetric (skew≈0)')
# Right skewed
right_skew = stats.skewnorm.pdf(x_range, a=5)
ax1e.plot(x_range, right_skew/right_skew.max()*0.45, color=COLORS[1], lw=2, ls='--', label='Right skew (+)')
# Left skewed
left_skew = stats.skewnorm.pdf(x_range, a=-5)
ax1e.plot(x_range, left_skew/left_skew.max()*0.45, color=COLORS[2], lw=2, ls='-.', label='Left skew (−)')
# Fat tails
ax1e.plot(x_range, stats.t.pdf(x_range, df=2)*1.3, color=COLORS[3], lw=2, ls=':', label='High kurtosis (fat tails)')
ax1e.set_title('1E · Skewness & Kurtosis\n(Shape Descriptors)')
ax1e.set_xlabel('Value'); ax1e.set_ylabel('Density'); ax1e.legend(fontsize=8)
skew_val = stats.skew(scores); kurt_val = stats.kurtosis(scores)
ax1e.text(0.02, 0.95, f'Our data:\nSkewness={skew_val:.2f}\nKurtosis={kurt_val:.2f}',
          transform=ax1e.transAxes, fontsize=8.5, va='top',
          bbox=dict(boxstyle='round', fc='#FFF8E6', ec='#E6AC00'))

# ── 1F: Empirical CDF ─────────────────────────────────────────────────────────
ax1f = fig1.add_subplot(gs[2, :2])
sorted_scores = np.sort(scores)
ecdf = np.arange(1, len(sorted_scores)+1) / len(sorted_scores)
ax1f.step(sorted_scores, ecdf, color=COLORS[0], lw=2, label='ECDF')
ax1f.fill_between(sorted_scores, ecdf, alpha=0.12, color=COLORS[0])
for p, label in [(0.25,'Q1'),(0.5,'Q2'),(0.75,'Q3')]:
    pval = np.percentile(scores, p*100)
    ax1f.axhline(p, color='#999', lw=0.8, ls='--')
    ax1f.axvline(pval, color='#999', lw=0.8, ls='--')
    ax1f.text(pval+0.5, p+0.02, label, fontsize=8, color='#555')
ax1f.set_title('1F · Empirical CDF\n(Cumulative Distribution)')
ax1f.set_xlabel('Score'); ax1f.set_ylabel('Cumulative Probability')
ax1f.text(0.98, 0.05, r'$F(x) = P(X \leq x) = \frac{\text{# obs} \leq x}{n}$',
          transform=ax1f.transAxes, ha='right', fontsize=8.5,
          bbox=dict(boxstyle='round,pad=0.4', fc='#EFF3FB', ec=COLORS[0]))

# ── 1G: Q-Q Plot ──────────────────────────────────────────────────────────────
ax1g = fig1.add_subplot(gs[2, 2:])
(osm, osr), (slope, intercept, r) = stats.probplot(scores, dist='norm')
ax1g.scatter(osm, osr, color=COLORS[0], s=18, alpha=0.6, label='Data quantiles')
line_x = np.array([osm.min(), osm.max()])
ax1g.plot(line_x, slope*line_x + intercept, color=COLORS[1], lw=2, label='Normal reference line')
ax1g.set_title('1G · Q-Q Plot\n(Normality Check)')
ax1g.set_xlabel('Theoretical Quantiles (Normal)'); ax1g.set_ylabel('Sample Quantiles')
ax1g.legend(fontsize=8)
_, p_shapiro = stats.shapiro(scores[:50])  # Shapiro works best on <50
ax1g.text(0.02, 0.95, f'Shapiro-Wilk p={p_shapiro:.4f}\n{"Normal ✓" if p_shapiro>0.05 else "Not normal ✗"}',
          transform=ax1g.transAxes, fontsize=9, va='top',
          bbox=dict(boxstyle='round', fc='#E8F5E9' if p_shapiro>0.05 else '#FFEBEE', 
                    ec='#1A9E75' if p_shapiro>0.05 else '#C0392B'))

fig1.savefig('/mnt/user-data/outputs/1_univariate_analysis.png', bbox_inches='tight', dpi=140)
print("✓ Figure 1 saved — Univariate Analysis")
plt.close(fig1)


# =============================================================================
# FIGURE 2 — BIVARIATE ANALYSIS
# =============================================================================
fig2 = plt.figure(figsize=(18, 16))
fig2.suptitle('BIVARIATE ANALYSIS\nRelationships Between Two Variables — Theory, Math & Visualization',
              fontsize=16, fontweight='bold', y=0.98, color='#1A1A1A')
gs2 = gridspec.GridSpec(3, 3, figure=fig2, hspace=0.55, wspace=0.4)

# ── 2A: Scatter + Regression ─────────────────────────────────────────────────
ax2a = fig2.add_subplot(gs2[0, :2])
x_hrs = student_df['study_hours'].values
y_sc  = student_df['score'].values
ax2a.scatter(x_hrs, y_sc, color=COLORS[0], alpha=0.5, s=30, edgecolor='white', lw=0.5)
# Fit regression
m, b, r_val, p_val, se = stats.linregress(x_hrs, y_sc)
x_line = np.linspace(x_hrs.min(), x_hrs.max(), 100)
ax2a.plot(x_line, m*x_line+b, color=COLORS[1], lw=2.2,
          label=f'ŷ = {b:.1f} + {m:.1f}x  |  R²={r_val**2:.3f}  p<0.001')
# Confidence band
n = len(x_hrs); x_bar = x_hrs.mean()
se_line = se * np.sqrt(1/n + (x_line-x_bar)**2/np.sum((x_hrs-x_bar)**2))
t_crit = stats.t.ppf(0.975, df=n-2)
ax2a.fill_between(x_line, m*x_line+b - t_crit*se_line,
                           m*x_line+b + t_crit*se_line,
                  color=COLORS[1], alpha=0.12, label='95% CI')
ax2a.set_title('2A · Scatter Plot + Simple Linear Regression')
ax2a.set_xlabel('Study Hours / Day'); ax2a.set_ylabel('Exam Score')
ax2a.legend(fontsize=8.5)
ax2a.text(0.98, 0.07,
    r'$\hat{y} = \beta_0 + \beta_1 x$  where  $\beta_1 = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2}$',
    transform=ax2a.transAxes, ha='right', fontsize=8.5,
    bbox=dict(boxstyle='round,pad=0.4', fc='#EFF3FB', ec=COLORS[0]))

# ── 2B: Correlation heatmap ───────────────────────────────────────────────────
ax2b = fig2.add_subplot(gs2[0, 2])
num_cols = ['score','study_hours','sleep_hours','attendance']
corr_mat = student_df[num_cols].corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, ax=ax2b, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5, linecolor='#F0EDE6',
            cbar_kws={'shrink':0.8}, square=True, annot_kws={'size':9})
ax2b.set_title('2B · Pearson Correlation Matrix')
ax2b.text(0.5, -0.15, r'$r = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2\sum(y_i-\bar{y})^2}}$',
          transform=ax2b.transAxes, ha='center', fontsize=8.5)

# ── 2C: Box plots by group (T-test) ───────────────────────────────────────────
ax2c = fig2.add_subplot(gs2[1, 0])
groups_data = [student_df[student_df['gender']==g]['score'].values for g in ['Male','Female']]
bp2 = ax2c.boxplot(groups_data, patch_artist=True, widths=0.5,
                   medianprops=dict(color='white', lw=2))
for patch, color in zip(bp2['boxes'], [COLORS[0], COLORS[2]]):
    patch.set_facecolor(color); patch.set_alpha(0.75)
t_stat, p_t = stats.ttest_ind(*groups_data)
ax2c.set_xticklabels(['Male','Female'])
ax2c.set_title(f'2C · T-Test\n(Numeric vs Categorical)')
ax2c.set_ylabel('Score')
sig_label = '*** p<0.001' if p_t<0.001 else ('** p<0.01' if p_t<0.01 else ('* p<0.05' if p_t<0.05 else f'ns (p={p_t:.3f})'))
y_max = max(g.max() for g in groups_data)
ax2c.plot([1,1,2,2], [y_max+2, y_max+4, y_max+4, y_max+2], color='#333', lw=1)
ax2c.text(1.5, y_max+4.5, sig_label, ha='center', fontsize=9, fontweight='bold')
ax2c.text(0.5, 0.03, r'$t = \frac{\bar{x}_1 - \bar{x}_2}{s_p\sqrt{1/n_1+1/n_2}}$',
          transform=ax2c.transAxes, ha='center', fontsize=8.5,
          bbox=dict(boxstyle='round', fc='#EFF3FB', ec=COLORS[0]))

# ── 2D: ANOVA (3 groups) ──────────────────────────────────────────────────────
ax2d = fig2.add_subplot(gs2[1, 1])
school_groups = [student_df[student_df['school_type']==s]['score'].values
                 for s in ['Public','Private','Charter']]
parts2 = ax2d.violinplot(school_groups, positions=[1,2,3], showmeans=True, showmedians=True)
for i, pc in enumerate(parts2['bodies']):
    pc.set_facecolor(COLORS[i]); pc.set_alpha(0.65)
parts2['cmeans'].set_color('#E85D24')
f_stat, p_anova = stats.f_oneway(*school_groups)
ax2d.set_xticks([1,2,3]); ax2d.set_xticklabels(['Public','Private','Charter'])
ax2d.set_title(f'2D · One-Way ANOVA\nF={f_stat:.2f}, p={p_anova:.4f}')
ax2d.set_ylabel('Score')
ax2d.text(0.5, 0.03,
    r'$F = \frac{MS_{between}}{MS_{within}} = \frac{SS_B/(k-1)}{SS_W/(N-k)}$',
    transform=ax2d.transAxes, ha='center', fontsize=8,
    bbox=dict(boxstyle='round', fc='#EFF3FB', ec=COLORS[0]))

# ── 2E: Chi-Square test ───────────────────────────────────────────────────────
ax2e = fig2.add_subplot(gs2[1, 2])
contingency = pd.crosstab(student_df['gender'], student_df['school_type'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
sns.heatmap(contingency, ax=ax2e, annot=True, fmt='d', cmap='Blues',
            linewidths=0.5, cbar_kws={'shrink':0.8})
ax2e.set_title(f'2E · Chi-Square Test\nχ²={chi2:.2f}, p={p_chi:.3f}, df={dof}')
ax2e.set_xlabel('School Type'); ax2e.set_ylabel('Gender')
ax2e.text(0.5, -0.22,
    r'$\chi^2 = \sum\frac{(O_{ij}-E_{ij})^2}{E_{ij}}$',
    transform=ax2e.transAxes, ha='center', fontsize=8.5)

# ── 2F: Pair plot style — joint distribution ─────────────────────────────────
ax2f = fig2.add_subplot(gs2[2, :2])
sc = ax2f.scatter(house_df['sqft'], house_df['price']/1000,
                  c=house_df['bedrooms'], cmap='viridis',
                  s=35, alpha=0.6, edgecolor='none')
cbar = plt.colorbar(sc, ax=ax2f)
cbar.set_label('Bedrooms', fontsize=8)
# Add Pearson r
r_house, p_house = stats.pearsonr(house_df['sqft'], house_df['price'])
ax2f.set_title(f'2F · Scatter (with 3rd variable as color)\nPearson r={r_house:.3f}, p<0.001')
ax2f.set_xlabel('House Area (sqft)'); ax2f.set_ylabel('Price ($000s)')
ax2f.text(0.02, 0.95, f'Spearman ρ={stats.spearmanr(house_df["sqft"],house_df["price"]).correlation:.3f}',
          transform=ax2f.transAxes, fontsize=8.5, va='top',
          bbox=dict(boxstyle='round', fc='#EFF3FB', ec=COLORS[0]))

# ── 2G: Residual plot ─────────────────────────────────────────────────────────
ax2g = fig2.add_subplot(gs2[2, 2])
reg = LinearRegression().fit(house_df[['sqft']], house_df['price'])
fitted = reg.predict(house_df[['sqft']])
residuals = house_df['price'] - fitted
ax2g.scatter(fitted/1000, residuals/1000, color=COLORS[0], alpha=0.45, s=20)
ax2g.axhline(0, color=COLORS[1], lw=2, ls='--')
ax2g.set_title('2G · Residual Plot\n(Model Diagnostic)')
ax2g.set_xlabel('Fitted Values ($000s)'); ax2g.set_ylabel('Residuals ($000s)')
ax2g.text(0.5, 0.97, 'Good: random scatter around 0\nBad: funnel/curve pattern',
          transform=ax2g.transAxes, ha='center', va='top', fontsize=8,
          color='#555')

fig2.savefig('/mnt/user-data/outputs/2_bivariate_analysis.png', bbox_inches='tight', dpi=140)
print("✓ Figure 2 saved — Bivariate Analysis")
plt.close(fig2)


# =============================================================================
# FIGURE 3 — MULTIVARIATE ANALYSIS
# =============================================================================
fig3 = plt.figure(figsize=(18, 18))
fig3.suptitle('MULTIVARIATE ANALYSIS\nThree or More Variables — Theory, Math & Visualization',
              fontsize=16, fontweight='bold', y=0.98, color='#1A1A1A')
gs3 = gridspec.GridSpec(3, 3, figure=fig3, hspace=0.55, wspace=0.4)

# ── 3A: Multiple Regression — actual vs predicted ────────────────────────────
ax3a = fig3.add_subplot(gs3[0, :2])
features = ['sqft','bedrooms','age']
X = house_df[features].values
y = house_df['price'].values
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
mlr = LinearRegression().fit(X_sc, y)
y_pred = mlr.predict(X_sc)
r2 = mlr.score(X_sc, y)
ax3a.scatter(y/1000, y_pred/1000, color=COLORS[0], alpha=0.45, s=22, edgecolor='none')
perfect = np.array([y.min(), y.max()])/1000
ax3a.plot(perfect, perfect, color=COLORS[1], lw=2, ls='--', label='Perfect fit')
ax3a.set_title(f'3A · Multiple Linear Regression\nActual vs Predicted  R²={r2:.3f}')
ax3a.set_xlabel('Actual Price ($000s)'); ax3a.set_ylabel('Predicted Price ($000s)')
ax3a.legend(fontsize=8.5)
coeff_text = '\n'.join([f'{f}: β={c:.0f}' for f,c in zip(features, mlr.coef_)])
ax3a.text(0.02, 0.97, f'Coefficients (standardized):\n{coeff_text}',
          transform=ax3a.transAxes, fontsize=8, va='top',
          bbox=dict(boxstyle='round', fc='#EFF3FB', ec=COLORS[0]))
ax3a.text(0.98, 0.07,
    r'$\hat{y}=\beta_0+\beta_1x_1+\beta_2x_2+...+\beta_kx_k$',
    transform=ax3a.transAxes, ha='right', fontsize=8.5,
    bbox=dict(boxstyle='round,pad=0.4', fc='#EFF3FB', ec=COLORS[0]))

# ── 3B: Coefficient plot ──────────────────────────────────────────────────────
ax3b = fig3.add_subplot(gs3[0, 2])
coef_df = pd.DataFrame({'feature':features,'coef':mlr.coef_})
colors_coef = [COLORS[2] if c>0 else COLORS[1] for c in coef_df['coef']]
ax3b.barh(coef_df['feature'], coef_df['coef'], color=colors_coef, alpha=0.8, edgecolor='white')
ax3b.axvline(0, color='#333', lw=1)
ax3b.set_title('3B · Feature Coefficients\n(Standardized)')
ax3b.set_xlabel('Coefficient Value')
for i, (_, row) in enumerate(coef_df.iterrows()):
    ax3b.text(row['coef'] + (500 if row['coef']>0 else -500), i,
              f"{row['coef']:+.0f}", va='center', fontsize=9, fontweight='bold')

# ── 3C: PCA — Scree + Biplot ─────────────────────────────────────────────────
ax3c = fig3.add_subplot(gs3[1, 0])
iris_X = iris_df[iris.feature_names].values
pca = PCA(n_components=4)
pca.fit(StandardScaler().fit_transform(iris_X))
expl = pca.explained_variance_ratio_ * 100
cumul = np.cumsum(expl)
ax3c.bar(range(1,5), expl, color=COLORS[0], alpha=0.75, edgecolor='white', label='Individual')
ax3c2 = ax3c.twinx()
ax3c2.plot(range(1,5), cumul, 'o-', color=COLORS[1], lw=2, ms=6, label='Cumulative')
ax3c2.axhline(95, color='#aaa', lw=0.8, ls='--')
ax3c2.set_ylabel('Cumulative %', fontsize=9)
ax3c2.set_ylim(0,110)
ax3c.set_title('3C · PCA Scree Plot\n(Explained Variance per Component)')
ax3c.set_xlabel('Principal Component'); ax3c.set_ylabel('Explained Variance %')
ax3c.set_xticks(range(1,5))
for i, (e, c) in enumerate(zip(expl, cumul)):
    ax3c.text(i+1, e+0.5, f'{e:.0f}%', ha='center', fontsize=8.5, fontweight='bold', color=COLORS[0])

# ── 3D: PCA Biplot ────────────────────────────────────────────────────────────
ax3d = fig3.add_subplot(gs3[1, 1])
pca2 = PCA(n_components=2)
iris_scaled = StandardScaler().fit_transform(iris_X)
X_pca = pca2.fit_transform(iris_scaled)
species_colors = [COLORS[0], COLORS[2], COLORS[1]]
for i, sp in enumerate(iris.target_names):
    mask = iris.target == i
    ax3d.scatter(X_pca[mask,0], X_pca[mask,1],
                 color=species_colors[i], s=35, alpha=0.7,
                 edgecolor='white', lw=0.5, label=sp)
# Loading vectors
scale = 3.5
for j, feat in enumerate(iris.feature_names):
    ax3d.annotate('', xy=(pca2.components_[0,j]*scale, pca2.components_[1,j]*scale),
                  xytext=(0,0),
                  arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax3d.text(pca2.components_[0,j]*scale*1.12, pca2.components_[1,j]*scale*1.12,
              feat.replace(' (cm)','').replace('sepal ','sep.\n').replace('petal ','pet.\n'),
              fontsize=7.5, ha='center', color='#333', fontweight='bold')
ax3d.set_title(f'3D · PCA Biplot\nPC1={pca2.explained_variance_ratio_[0]*100:.0f}%  PC2={pca2.explained_variance_ratio_[1]*100:.0f}%')
ax3d.set_xlabel(f'PC1'); ax3d.set_ylabel('PC2')
ax3d.legend(fontsize=8, loc='upper right')
ax3d.text(0.02, 0.05, r'$\mathbf{Z}=\mathbf{X}\mathbf{W}$  (project onto eigenvectors)',
          transform=ax3d.transAxes, fontsize=7.5,
          bbox=dict(boxstyle='round', fc='#EFF3FB', ec=COLORS[0]))

# ── 3E: K-Means Clustering ───────────────────────────────────────────────────
ax3e = fig3.add_subplot(gs3[1, 2])
# Elbow method
inertias = []
K_range = range(1, 9)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(iris_scaled)
    inertias.append(km.inertia_)
ax3e.plot(K_range, inertias, 'o-', color=COLORS[0], lw=2, ms=7)
ax3e.axvline(3, color=COLORS[1], lw=2, ls='--', label='Elbow at k=3')
ax3e.set_title('3E · K-Means Elbow Method\n(Choosing Optimal K)')
ax3e.set_xlabel('Number of Clusters (K)'); ax3e.set_ylabel('Inertia (Within-Cluster SS)')
ax3e.legend(fontsize=8.5)
ax3e.fill_between([2.7, 3.3], [min(inertias)*0.95, min(inertias)*0.95],
                  [max(inertias)*1.02, max(inertias)*1.02],
                  alpha=0.1, color=COLORS[1])
ax3e.text(0.98, 0.97,
    r'$J = \sum_{k=1}^K\sum_{x \in C_k} \|x - \mu_k\|^2$',
    transform=ax3e.transAxes, ha='right', va='top', fontsize=8.5,
    bbox=dict(boxstyle='round,pad=0.4', fc='#EFF3FB', ec=COLORS[0]))

# ── 3F: Clustered scatter ─────────────────────────────────────────────────────
ax3f = fig3.add_subplot(gs3[2, 0])
km_final = KMeans(n_clusters=3, n_init=10, random_state=42)
cluster_labels = km_final.fit_predict(iris_scaled)
for k in range(3):
    mask = cluster_labels == k
    ax3f.scatter(X_pca[mask,0], X_pca[mask,1],
                 color=COLORS[k], s=35, alpha=0.7, edgecolor='white', lw=0.5,
                 label=f'Cluster {k+1}')
# Centroids
centroids_pca = pca2.transform(km_final.cluster_centers_)
ax3f.scatter(centroids_pca[:,0], centroids_pca[:,1],
             s=150, marker='*', color='black', zorder=5, label='Centroids')
ax3f.set_title('3F · K-Means Clusters (k=3)\nProjected onto PCA Space')
ax3f.set_xlabel('PC1'); ax3f.set_ylabel('PC2')
ax3f.legend(fontsize=8)

# ── 3G: Full pair plot (seaborn) ─────────────────────────────────────────────
ax3g = fig3.add_subplot(gs3[2, 1:])
# Correlation matrix as proxy for pairplot concept
corr_iris = iris_df[iris.feature_names].corr()
mask_tri = np.triu(np.ones_like(corr_iris, dtype=bool), k=1)
sns.heatmap(corr_iris, ax=ax3g, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, linewidths=0.8, linecolor='#F0EDE6',
            square=True, cbar_kws={'shrink':0.8}, annot_kws={'size':9})
ax3g.set_title('3G · Multivariate Correlation Heatmap\n(Iris — All Feature Pairs)')
ax3g.set_xticklabels(ax3g.get_xticklabels(), rotation=25, ha='right', fontsize=8)
ax3g.set_yticklabels(ax3g.get_yticklabels(), rotation=0, fontsize=8)

fig3.savefig('/mnt/user-data/outputs/3_multivariate_analysis.png', bbox_inches='tight', dpi=140)
print("✓ Figure 3 saved — Multivariate Analysis")
plt.close(fig3)


# =============================================================================
# FIGURE 4 — SEABORN GALLERY (Advanced Plots)
# =============================================================================
fig4 = plt.figure(figsize=(18, 14))
fig4.suptitle('SEABORN ADVANCED VISUALIZATION GALLERY\nExpressive Statistical Charts for All Analysis Types',
              fontsize=16, fontweight='bold', y=0.98, color='#1A1A1A')
gs4 = gridspec.GridSpec(2, 3, figure=fig4, hspace=0.5, wspace=0.4)

# ── 4A: seaborn pairplot replacement (manual for single figure) ──────────────
ax4a = fig4.add_subplot(gs4[0, 0])
for i, sp in enumerate(iris.target_names):
    mask = iris_df['species'] == sp
    ax4a.scatter(iris_df.loc[mask,'sepal length (cm)'],
                 iris_df.loc[mask,'petal length (cm)'],
                 color=COLORS[i], s=35, alpha=0.7, edgecolor='white', lw=0.3, label=sp)
ax4a.set_xlabel('Sepal Length (cm)'); ax4a.set_ylabel('Petal Length (cm)')
ax4a.set_title('4A · sns.scatterplot\nSpecies differentiated by hue')
ax4a.legend(fontsize=8)

# ── 4B: seaborn KDE plot ──────────────────────────────────────────────────────
ax4b = fig4.add_subplot(gs4[0, 1])
for i, sp in enumerate(iris.target_names):
    data_sp = iris_df.loc[iris_df['species']==sp, 'petal length (cm)'].values
    kde_x = np.linspace(0, 8, 300)
    kde_y = stats.gaussian_kde(data_sp)(kde_x)
    ax4b.plot(kde_x, kde_y, color=COLORS[i], lw=2.2, label=sp)
    ax4b.fill_between(kde_x, kde_y, alpha=0.18, color=COLORS[i])
ax4b.set_xlabel('Petal Length (cm)'); ax4b.set_ylabel('Density')
ax4b.set_title('4B · sns.kdeplot (filled)\nOverlapping distributions')
ax4b.legend(fontsize=8)

# ── 4C: seaborn catplot style — strip + box ───────────────────────────────────
ax4c = fig4.add_subplot(gs4[0, 2])
bp3 = ax4c.boxplot(
    [student_df[student_df['school_type']==s]['score'].values for s in ['Public','Private','Charter']],
    patch_artist=True, widths=0.4,
    medianprops=dict(color='white', lw=2.5),
)
for patch, color in zip(bp3['boxes'], COLORS[:3]):
    patch.set_facecolor(color); patch.set_alpha(0.4)
for i, (s, color) in enumerate(zip(['Public','Private','Charter'], COLORS)):
    data = student_df[student_df['school_type']==s]['score'].values
    jitter = np.random.uniform(-0.18, 0.18, len(data))
    ax4c.scatter(np.ones(len(data))*(i+1)+jitter, data,
                 color=color, s=12, alpha=0.45, edgecolor='none')
ax4c.set_xticks([1,2,3]); ax4c.set_xticklabels(['Public','Private','Charter'])
ax4c.set_title('4C · Boxplot + Strip (jitter)\nDistribution per category')
ax4c.set_ylabel('Score')

# ── 4D: Heatmap with annotations ──────────────────────────────────────────────
ax4d = fig4.add_subplot(gs4[1, 0])
pivot = student_df.copy()
pivot['score_bin'] = pd.cut(pivot['score'], bins=[0,60,75,90,101], labels=['<60','60-75','75-90','90+'])
pivot['sleep_bin'] = pd.cut(pivot['sleep_hours'], bins=[3,6,7,8,11], labels=['<6','6-7','7-8','8+'])
pt = pd.crosstab(pivot['sleep_bin'], pivot['score_bin'])
sns.heatmap(pt, ax=ax4d, annot=True, fmt='d', cmap='YlOrRd',
            linewidths=0.5, cbar_kws={'shrink':0.8})
ax4d.set_title('4D · sns.heatmap\nFrequency cross-tabulation')
ax4d.set_xlabel('Score Bin'); ax4d.set_ylabel('Sleep Hours')

# ── 4E: Regression with confidence interval (seaborn style) ──────────────────
ax4e = fig4.add_subplot(gs4[1, 1])
for i, loc in enumerate(['Urban','Suburban','Rural']):
    sub = house_df[house_df['location']==loc]
    x = sub['sqft'].values; y = sub['price'].values/1000
    m,b,r,p,se = stats.linregress(x,y)
    xs = np.linspace(x.min(), x.max(), 100)
    ax4e.scatter(x, y, color=COLORS[i], s=18, alpha=0.4, edgecolor='none')
    ax4e.plot(xs, m*xs+b, color=COLORS[i], lw=2, label=f'{loc} (r={r:.2f})')
ax4e.set_title('4E · sns.lmplot style\nRegression by group (location)')
ax4e.set_xlabel('Sqft'); ax4e.set_ylabel('Price ($000s)')
ax4e.legend(fontsize=8)

# ── 4F: Multi-variable bubble plot ────────────────────────────────────────────
ax4f = fig4.add_subplot(gs4[1, 2])
sample = house_df.sample(120, random_state=1)
sc2 = ax4f.scatter(sample['sqft'], sample['price']/1000,
                   s=sample['bedrooms']*40,
                   c=sample['age'], cmap='plasma_r',
                   alpha=0.65, edgecolor='white', lw=0.5)
cbar2 = plt.colorbar(sc2, ax=ax4f)
cbar2.set_label('House Age (yrs)', fontsize=8)
for beds in [2,3,4]:
    ax4f.scatter([], [], s=beds*40, c='gray', alpha=0.6, label=f'{beds} beds')
ax4f.set_title('4F · Bubble chart (4 variables)\nSize=bedrooms, Color=age')
ax4f.set_xlabel('Sqft'); ax4f.set_ylabel('Price ($000s)')
ax4f.legend(fontsize=8, title='Bedrooms', title_fontsize=8)

fig4.savefig('/mnt/user-data/outputs/4_seaborn_gallery.png', bbox_inches='tight', dpi=140)
print("✓ Figure 4 saved — Seaborn Gallery")
plt.close(fig4)


# =============================================================================
# FIGURE 5 — CHEATSHEET / SUMMARY
# =============================================================================
fig5, ax = plt.subplots(figsize=(18, 10))
ax.axis('off')
fig5.patch.set_facecolor('#FAFAF8')

title_y = 0.97
ax.text(0.5, title_y, 'MATPLOTLIB & SEABORN — STATISTICAL ANALYSIS CHEATSHEET',
        transform=ax.transAxes, ha='center', va='top', fontsize=14,
        fontweight='bold', color='#1A1A1A')

sections = [
    ("UNIVARIATE\n(1 variable)", COLORS[0], [
        "plt.hist(data, bins=30, density=True)",
        "sns.histplot(data, kde=True)",
        "ax.boxplot(data, patch_artist=True)",
        "sns.boxplot(y='col', data=df)",
        "sns.violinplot(y='col', data=df)",
        "stats.gaussian_kde(data) — KDE",
        "stats.probplot(data) — Q-Q plot",
        "data.describe() — summary stats",
    ]),
    ("BIVARIATE\n(2 variables)", COLORS[2], [
        "ax.scatter(x, y, c='blue', alpha=0.5)",
        "sns.scatterplot(x='a', y='b', data=df)",
        "sns.regplot(x='a', y='b', data=df)",
        "stats.linregress(x, y) — regression",
        "stats.pearsonr(x, y) — correlation",
        "stats.spearmanr(x, y) — rank corr.",
        "stats.ttest_ind(g1, g2) — t-test",
        "stats.chi2_contingency(ctab) — χ²",
    ]),
    ("MULTIVARIATE\n(3+ variables)", COLORS[3], [
        "sns.heatmap(df.corr(), annot=True)",
        "sns.pairplot(df, hue='species')",
        "PCA(n_components=2).fit_transform(X)",
        "KMeans(n_clusters=k).fit(X)",
        "sns.clustermap(df, cmap='coolwarm')",
        "LinearRegression().fit(X, y)",
        "sns.FacetGrid(df, col='cat')",
        "ax.scatter(x,y, s=size, c=color)",
    ]),
]

col_positions = [0.04, 0.37, 0.70]
for col_idx, (title, color, items) in enumerate(sections):
    cx = col_positions[col_idx]
    # Section header
    fig5.add_axes([cx, 0.73, 0.27, 0.17]).set_axis_off()
    ax.text(cx + 0.135, 0.88, title, transform=ax.transAxes,
            ha='center', va='top', fontsize=11, fontweight='bold', color=color)
    # Items
    for j, item in enumerate(items):
        y_pos = 0.80 - j * 0.075
        ax.text(cx + 0.01, y_pos, '▸', transform=ax.transAxes,
                fontsize=9, color=color, va='top')
        ax.text(cx + 0.025, y_pos, item, transform=ax.transAxes,
                fontsize=8.5, va='top', fontfamily='monospace',
                color='#1A1A1A')

# Bottom: key math formulas
ax.plot([0.02, 0.98], [0.12, 0.12], color='#DDD', lw=0.8,
        transform=ax.transAxes)
formulas = [
    r'Mean: $\bar{x}=\frac{\sum x_i}{n}$',
    r'Variance: $s^2=\frac{\sum(x_i-\bar{x})^2}{n-1}$',
    r'Pearson $r=\frac{\sum(x-\bar{x})(y-\bar{y})}{(n-1)s_x s_y}$',
    r'Regression: $\hat{y}=\beta_0+\beta_1 x+...+\beta_k x_k$',
    r'Chi-sq: $\chi^2=\sum\frac{(O-E)^2}{E}$',
    r'F-stat: $F=\frac{MS_{between}}{MS_{within}}$',
]
for i, formula in enumerate(formulas):
    ax.text(0.02 + i * 0.165, 0.08, formula, transform=ax.transAxes,
            fontsize=8.5, va='top', ha='left')

ax.text(0.5, 0.02, 'Sample Datasets: student performance (n=200), house prices (n=300), iris (n=150)',
        transform=ax.transAxes, ha='center', fontsize=8.5, color='#777')

fig5.savefig('/mnt/user-data/outputs/5_cheatsheet.png', bbox_inches='tight', dpi=140)
print("✓ Figure 5 saved — Cheatsheet")
plt.close(fig5)

print("\n" + "="*60)
print("ALL 5 FIGURES SAVED SUCCESSFULLY")
print("="*60)
print("""
Files:
  1_univariate_analysis.png  — Histograms, boxplots, KDE, Q-Q, CDF
  2_bivariate_analysis.png   — Scatter, correlation, regression, ANOVA, Chi-sq
  3_multivariate_analysis.png — MLR, PCA, K-Means, clustering, heatmap
  4_seaborn_gallery.png      — Advanced seaborn visualizations
  5_cheatsheet.png           — Code + math formula reference
""")
