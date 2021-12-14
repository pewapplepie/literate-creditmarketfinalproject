"""
@author jeffrey
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

from statsmodels.regression.rolling import RollingOLS
import pandas as pd
from math import erfc
from sympy import det

from tables import Col


# Note that norm().cdf() from scipy is sometimes slow
def normcdf(x):
    '''Standard normal CDF'''
    return erfc(-x / np.sqrt(2)) / 2

# ==============================================================================
# Question 3
# ==============================================================================

# ---------- (a)
# Parameters
llambda = 0.3
R = 0.5
Lbar = 0.75
T = 5
# Prepare data
dfboeing = pd.read_csv('Boeing.csv')
dfboeing = dfboeing.dropna(axis=1, how='all')
dfboeing['date'] = pd.to_datetime(dfboeing['date'])
dfboeing.sort_values('date', inplace=True)


def get_cds(S, sigma_S, r, D, Lbar, llambda, R, T):
    '''Calculate CDS'''
    V0 = S + Lbar * D
    d = V0 / (Lbar * D) * np.exp(llambda**2)
    sigma = sigma_S * S / (S + Lbar * D)
    A0 = np.sqrt(sigma**2*0 + llambda**2)
    AT = np.sqrt(sigma**2*T + llambda**2)
    xi = (llambda/sigma)**2
    z = np.sqrt(0.25 + 2 * r / sigma**2)
    Gxi = d**(z+0.5) * \
        normcdf(-np.log(d)/(sigma*np.sqrt(xi)) - z*sigma*np.sqrt(xi)) + \
        d**(-z+0.5) * \
        normcdf(-np.log(d)/(sigma*np.sqrt(xi)) + z*sigma*np.sqrt(xi))
    GT = d**(z+0.5) * \
        normcdf(-np.log(d)/(sigma*np.sqrt(T)) - z*sigma*np.sqrt(T)) + \
        d**(-z+0.5) * \
        normcdf(-np.log(d)/(sigma*np.sqrt(T)) + z*sigma*np.sqrt(T))
    Txi = T + xi
    GTxi = d**(z+0.5) * \
        normcdf(-np.log(d)/(sigma*np.sqrt(Txi)) - z*sigma*np.sqrt(Txi)) + \
        d**(-z+0.5) * \
        normcdf(-np.log(d)/(sigma*np.sqrt(Txi)) + z*sigma*np.sqrt(Txi))
    q0 = normcdf(-A0/2+np.log(d)/A0) - d*normcdf(-A0/2-np.log(d)/A0)
    qT = normcdf(-AT/2+np.log(d)/AT) - d*normcdf(-AT/2-np.log(d)/AT) 
    HT = np.exp(r*xi) * (GTxi - Gxi)
    cds = 100 * r*(1-R) * (1-q0+HT) / (q0-qT*np.exp(-r*T)-HT)    # in %
    rpv01 = (q0 - qT*np.exp(-r*T) - HT) / r
    # Return variables other than cds for later use
    return {
        'cds': cds, 
        'rpv01': rpv01, 
        'q0': q0, 
        'qT': qT, 
        'GTxi': GTxi, 
        'GT': GT
    }


dfboeing['CGcds'] = [
    get_cds(x.S, x.sigma_stock, x.r, x.D, Lbar, llambda, R, T)['cds'] 
    for x in dfboeing.itertuples()
]
# Plot
fig, ax = plt.subplots()
dfboeing.set_index('date')[['spread5y', 'CGcds']].plot(ax=ax, linewidth=0.5)
ax.set_ylabel('CDS spread (%)')
fig.savefig('Q3_a.png', dpi=150)

# ---------- (b)


def get_pd_from_numdiff(S0, sigma_S, r, D, Lbar, llambda, R, T, dS):
    '''Calculate the partial derivative using numerical differentiation'''
    # Numerical differentiation with step size dS
    # Note that CDS spread from get_cds() is in %
    y1 = get_cds(S0+dS, sigma_S, r, D, Lbar, llambda, R, T)['cds']
    y0 = get_cds(S0-dS, sigma_S, r, D, Lbar, llambda, R, T)['cds']
    # Get hedge ratio
    return (y1-y0) / (2*dS)


def get_rpv01(S0, sigma_S, r, D, Lbar, llambda, R, T):
    '''Calculate RPV01'''
    return get_cds(S0, sigma_S, r, D, Lbar, llambda, R, T)['rpv01']


# Calculate RPV01
dfboeing['rpv01'] = [
    get_rpv01(x.S, x.sigma_stock, x.r, x.D, Lbar, llambda, R, T)
    for x in dfboeing.itertuples()
]
# Calculate partial derivative using numerical differentiation
dS = 0.1
dfboeing['partial deriv (numdiff)'] = [
    get_pd_from_numdiff(x.S, x.sigma_stock, x.r, x.D, Lbar, llambda, R, T, dS)
    for x in dfboeing.itertuples()
]
# Calculate partiald derivative using regression
winsize = 250
minobs = 125
dfboeing['dS'] = dfboeing['S'].diff(1)
dfboeing['dCGcds'] = dfboeing['CGcds'].diff(1)
dfboeing['const'] = 1
mod = RollingOLS(
    dfboeing['dCGcds'].to_numpy(), dfboeing[['const', 'dS']].to_numpy(), 
    window=winsize, min_nobs=minobs, expanding=True
)
dfboeing['partial deriv (reg)'] = mod.fit().params[:,1]
# Calculate hedge ratio
dfboeing['hedge ratio (numdiff)'] = dfboeing['partial deriv (numdiff)'] * dfboeing['rpv01']
dfboeing['hedge ratio (reg)'] = dfboeing['partial deriv (reg)'] * dfboeing['rpv01']
dfboeing[['date', 'hedge ratio (numdiff)', 'hedge ratio (reg)']]
# Plot
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(dfboeing['date'], dfboeing['hedge ratio (numdiff)'], linewidth=0.5)
ax.plot(dfboeing['date'], dfboeing['hedge ratio (reg)'], linewidth=0.5)
ax.set_ylabel('Hedge ratio')
ax.set_xlabel('date')
ax.legend(['From numerical differentiation', 'From rolling regression'])
# fig.subplots_adjust(left=0.2, right=0.95)
fig.savefig('Q3_b.png', dpi=150)

# ---------- (c)
# Strategy parameters
# threshold = 3
# maxholdingdays = 22
# winsize = 250
# minobs = 125
# # Difference
# dfboeing['diff'] = dfboeing['spread5y'] - dfboeing['CGcds']
# # Rolling mean and standard deviation
# dfboeing['diff_mean'] = dfboeing['diff'].rolling(window=winsize, min_periods=minobs).mean()
# dfboeing['diff_std'] = dfboeing['diff'].rolling(window=winsize, min_periods=minobs).std(doff=1)
# dfboeing['diff_z'] = (dfboeing['diff'] - dfboeing['diff_mean']) / dfboeing['diff_std']
# # Arrays used for the strategy
# diff_z = dfboeing['diff_z'].to_numpy()
# # Strategy if market CDS spread is high
# trigger1 = np.zeros(dfboeing.shape[0], dtype='int')
# pos1 = np.zeros(dfboeing.shape[0], dtype='int')
# for i in range(dfboeing.shape[0]):
#     if diff_z[i] > 3 and np.isfinite(diff_z[i]) and (i > 0 and trigger1[i-1] == 0):
#         trigger1[i] = 1
#         pos1[i] = 1
#     if i > 0 and pos1[i-1] != 0 and pos1[i-1] < maxholdingdays:
#         pos1[i] = pos1[i-1] + 1
#     if pos1[i] != 1:
#         trigger1[i] = 0
# # Strategy if market CDS spread is low
# trigger2 = np.zeros(dfboeing.shape[0], dtype='int')
# pos2 = np.zeros(dfboeing.shape[0], dtype='int')
# for i in range(dfboeing.shape[0]):
#     if diff_z[i] < -3 and np.isfinite(diff_z[i]) and (i > 0 and trigger1[i-1] == 0):
#         trigger2[i] = 1
#         pos2[i] = 1
#     if i > 0 and pos2[i-1] != 0 and pos2[i-1] < maxholdingdays:
#         pos2[i] = pos2[i-1] + 1
#     if pos2[i] != 1:
#         trigger2[i] = 0
# dfboeing['trigger1'] = trigger1
# dfboeing['trigger2'] = trigger2
# dfboeing['pos1'] = pos1
# dfboeing['pos2'] = pos2
# # Calculate cumulative stock return in the next month
# dfboeing['stkret'] = dfboeing['S']/dfboeing['S'].shift(1)-1
# dfboeing['fret1m'] = dfboeing.rolling(window=maxholdingdays)['stkret'].apply(lambda x: 100*((1+x).prod()-1)).shift(-maxholdingdays)
# # Calculate cumulative CDS return in the next month
# dfboeing['dcdsret'] = (dfboeing['spread5y'] - dfboeing['spread5y'].shift(1)) * dfboeing['rpv01']
# dfboeing['fcds1m'] = dfboeing.rolling(window=maxholdingdays)['dcdsret'].sum().shift(-maxholdingdays)
# # Calculate arbitrage portfolio return
# dfboeing['Arbitrage return'] = np.nan
# mask1 = dfboeing['trigger1'] == 1
# dfboeing.loc[mask1, 'caparb'] = dfboeing.loc[mask1, 'fret1m'] * dfboeing.loc[mask1, 'hedge ratio (reg)'] - dfboeing.loc[mask1, 'fcds1m']
# mask2 = dfboeing['trigger2'] == 1
# dfboeing.loc[mask2, 'caparb'] = -dfboeing.loc[mask2, 'fret1m'] * dfboeing.loc[mask2, 'hedge ratio (reg)'] + dfboeing.loc[mask2, 'fcds1m']
# # Print results
# outtab1 = dfboeing.dropna(subset=['caparb']).loc[mask1, 'caparb'].describe().to_frame('High market CDS')
# outtab2 = dfboeing.dropna(subset=['caparb']).loc[mask2, 'caparb'].describe().to_frame('Low market CDS')
# outtab = dfboeing.dropna(subset=['caparb'])['caparb'].describe().to_frame('Arbitrage return (%)')
# outtaball = pd.concat([outtab1, outtab2, outtab], axis=1)
# print(outtaball.to_markdown(tablefmt='psql'))

# dfboeing.to_csv('Boeing_ANSWER.csv')
#print(outtaball)

##################################################################
# Final Project
##################################################################


threshold = 3
maxholdingdays = 22
winsize = 250
minobs = 125
# Difference
dfboeing['diff'] = dfboeing['spread5y'] - dfboeing['CGcds']
# Rolling mean and standard deviation
dfboeing['diff_mean'] = dfboeing['diff'].rolling(window=winsize, min_periods=minobs).mean()
dfboeing['diff_std'] = dfboeing['diff'].rolling(window=winsize, min_periods=minobs).std(doff=1)
dfboeing['diff_z'] = (dfboeing['diff'] - dfboeing['diff_mean']) / dfboeing['diff_std']
# Arrays used for the strategy
diff_z = dfboeing['diff_z'].to_numpy()
# Strategy if market CDS spread is high
trigger1 = np.zeros(dfboeing.shape[0], dtype='int')
pos1 = np.zeros(dfboeing.shape[0], dtype='int')

def arb_ret(df, maxholdingdays, threshold):
    df_calc = df.copy()
    for i in range(df_calc.shape[0]):
        if diff_z[i] > threshold and np.isfinite(diff_z[i]) and (i > 0 and trigger1[i-1] == 0):
            trigger1[i] = 1
            pos1[i] = 1
        if i > 0 and pos1[i-1] != 0 and pos1[i-1] < maxholdingdays:
            pos1[i] = pos1[i-1] + 1
        if pos1[i] != 1:
            trigger1[i] = 0
# Strategy if market CDS spread is low
    trigger2 = np.zeros(df_calc.shape[0], dtype='int')
    pos2 = np.zeros(df_calc.shape[0], dtype='int')
    for i in range(df_calc.shape[0]):
        if diff_z[i] < -threshold and np.isfinite(diff_z[i]) and (i > 0 and trigger1[i-1] == 0):
            trigger2[i] = 1
            pos2[i] = 1
        if i > 0 and pos2[i-1] != 0 and pos2[i-1] < maxholdingdays:
            pos2[i] = pos2[i-1] + 1
        if pos2[i] != 1:
            trigger2[i] = 0
    df_calc['trigger1'] = trigger1
    df_calc['trigger2'] = trigger2
    df_calc['pos1'] = pos1
    df_calc['pos2'] = pos2
# Calculate cumulative stock return in the next month
    df_calc['stkret'] = df_calc['S']/df_calc['S'].shift(1)-1
    df_calc['fret1m'] = df_calc.rolling(window=maxholdingdays)['stkret'].apply(lambda x: 100*((1+x).prod()-1)).shift(-maxholdingdays)
# Calculate cumulative CDS return in the next month
    df_calc['dcdsret'] = (df_calc['spread5y'] - df_calc['spread5y'].shift(1)) * df_calc['rpv01']
    df_calc['fcds1m'] = df_calc.rolling(window=maxholdingdays)['dcdsret'].sum().shift(-maxholdingdays)
# Calculate arbitrage portfolio return
    df_calc['Arbitrage return'] = np.nan
    mask1 = df_calc['trigger1'] == 1
    df_calc.loc[mask1, 'caparb'] = df_calc.loc[mask1, 'fret1m'] * df_calc.loc[mask1, 'hedge ratio (reg)'] - df_calc.loc[mask1, 'fcds1m']
    mask2 = df_calc['trigger2'] == 1
    df_calc.loc[mask2, 'caparb'] = -df_calc.loc[mask2, 'fret1m'] * df_calc.loc[mask2, 'hedge ratio (reg)'] + df_calc.loc[mask2, 'fcds1m']
# Print results
    outtab1 = df_calc.dropna(subset=['caparb']).loc[mask1, 'caparb'].describe().to_frame('High market CDS')
    outtab2 = df_calc.dropna(subset=['caparb']).loc[mask2, 'caparb'].describe().to_frame('Low market CDS')
    outtab = df_calc.dropna(subset=['caparb'])['caparb'].describe().to_frame('Arbitrage return (%)')
    outtaball = pd.concat([outtab1, outtab2, outtab], axis=1)
# print(outtaball.to_markdown(tablefmt='psql'))
    temp1 = dfboeing.dropna(subset=['caparb']).loc[mask2, ['date', 'caparb']]
# dfboeing.to_csv('Boeing_ANSWER.csv')
    return (df_calc, outtaball)
#    return temp1

############################################################
# Data Mining Summary Table
## threshold 3
hold_period = pd.DataFrame(np.zeros((1, 6)), columns=['1w', '2w', '3w', '1m', '3m','6m'], index=['3'])
hold_period['1m'] = arb_ret(dfboeing,22, 3)[1]['Arbitrage return (%)']['mean']
hold_period['3m'] = arb_ret(dfboeing, 66, 3)[1]['Arbitrage return (%)']['mean']
hold_period['6m'] = arb_ret(dfboeing, 126, 3)[1]['Arbitrage return (%)']['mean']
hold_period['1w'] = arb_ret(dfboeing, 5, 3)[1]['Arbitrage return (%)']['mean']
hold_period['2w'] = arb_ret(dfboeing, 10, 3)[1]['Arbitrage return (%)']['mean']
hold_period['3w'] = arb_ret(dfboeing, 15, 3)[1]['Arbitrage return (%)']['mean']
## threhsold 1
thred1_1m = arb_ret(dfboeing,22, 1)[1]['Arbitrage return (%)']['mean']
thred1_3m = arb_ret(dfboeing, 66, 1)[1]['Arbitrage return (%)']['mean']
thred1_6m = arb_ret(dfboeing, 126, 1)[1]['Arbitrage return (%)']['mean']
thred1_1w = arb_ret(dfboeing, 5, 1)[1]['Arbitrage return (%)']['mean']
thred1_2w = arb_ret(dfboeing, 10, 1)[1]['Arbitrage return (%)']['mean']
thred1_3w = arb_ret(dfboeing, 15, 1)[1]['Arbitrage return (%)']['mean']
hold_period.loc['1'] = [thred1_1m, thred1_3m, thred1_6m, thred1_1w, thred1_2w, thred1_3w]
## threshold 0.5
thred05_1m = arb_ret(dfboeing,22, .5)[1]['Arbitrage return (%)']['mean']
thred05_3m = arb_ret(dfboeing, 66, .5)[1]['Arbitrage return (%)']['mean']
thred05_6m = arb_ret(dfboeing, 126, .5)[1]['Arbitrage return (%)']['mean']
thred05_1w = arb_ret(dfboeing, 5, .5)[1]['Arbitrage return (%)']['mean']
thred05_2w = arb_ret(dfboeing, 10, .5)[1]['Arbitrage return (%)']['mean']
thred05_3w = arb_ret(dfboeing, 15, .5)[1]['Arbitrage return (%)']['mean']
hold_period.loc['.5'] = [thred05_1m, thred05_3m, thred05_6m, thred05_1w, thred05_2w, thred05_3w]

# plt.plot(np.arange(1,6,1),hold_period.iloc[0,:])
# plt.barh(np.arange(1,6,1), hold_period.iloc[0,:], tick_label = hold_period.columns.tolist())
# plt.title('Arbitrage return (%) at different holding period')
# plt.show()

############################################################
# Figure 1
import seaborn as sns

hold_period_1m = arb_ret(dfboeing,22, 3)[0]
hold_period_3m = arb_ret(dfboeing,66, 3)[0]
hold_period_6m = arb_ret(dfboeing,126, 3)[0]
hold_period_1w = arb_ret(dfboeing,5, 3)[0]
hold_period_2w = arb_ret(dfboeing,10, 3)[0]
hold_period_3w = arb_ret(dfboeing,15, 3)[0]

exret_all_df = pd.concat(
    [hold_period_1m[['date','caparb']],
    hold_period_3m['caparb'],
    hold_period_6m['caparb'],
    hold_period_1w['caparb'],
    hold_period_2w['caparb'],
    hold_period_3w['caparb']],axis=1
    )
exret_all_df.columns = ['date', 'caparb1m','caparb3m','caparb6m','caparb1w','caparb2w','caparb3w']
exret_all_df = exret_all_df.fillna(0)
caparb_2018 = exret_all_df[exret_all_df['date'] < '2019'].copy()
caparb_2020 = exret_all_df[exret_all_df['date'] >= '2019'].copy()
#exret_all_df = exret_all_df.set_index('date')
#sns.set_theme()
sns.lineplot(
    x='date',y='value', hue='variable',
    data=pd.melt(caparb_2018,['date']))
# f2_x = arb_ret(22)['date']
# f2_y = arb_ret(44)['date']

# fig, ax = plt.subplots(figsize=(8,5))
# ax.plot(arb_ret(5)['date'], arb_ret(5)['caparb'], linewidth=0.5)
# ax.plot(arb_ret(10)['date'], arb_ret(10)['caparb'], linewidth=0.5)
# ax.plot(arb_ret(15)['date'], arb_ret(15)['caparb'], linewidth=0.5)
# ax.plot(arb_ret(22)['date'], arb_ret(22)['caparb'], linewidth=0.5)
# ax.plot(arb_ret(44)['date'], arb_ret(44)['caparb'], linewidth=0.5)
# ax.set_ylabel('Arbitrage return (%)')
# ax.set_xlabel('date')
# ax.legend(['1 week', '2 weeks', '3 weeks', '1 month', '2 months'])

############################################################
## Stategy Modification
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
import numpy_ext as npext
### price discovery
df_prcdis = dfboeing.copy()
df_prcdis = df_prcdis[['spread5y','CGcds']].copy()
df_prcdis.index = pd.DatetimeIndex(dfboeing['date'])
## VECM
## lag order selection
# lag_order = select_order(data=df_prcdis, maxlags=250, deterministic="ci")
# lag_order.summary()
# lag_order.aic, lag_order.bic, lag_order.fpe, lag_order.hqic
# vec_rank1 = vecm.select_coint_rank(df_prcdis, det_order = 1, k_ar_diff = 1, method = 'trace', signif=0.01)
# print(vec_rank.summary())
vecm_model = VECM(df_prcdis, coint_rank=1, k_ar_diff=250,deterministic='ci')
vecm_res = vecm_model.fit()
vecm_res.summary()

#### Getting Rolling Covariance Mat for unconditional vol
#cov_mats = df_prcdis.rolling(250).cov().groupby('date')
#chol_roots = []
# for n, gp in cov_mats:
#     if gp.isnull().values.any():
#         #print('ye',end=",")
#         chol_roots.append(0)
#     else:
#         #print("ll",end=",")
#         gp = gp.fillna(0)
#         chol_roots.append(np.linalg.cholesky(gp.values, use='pairwise'))
import warnings
warnings.filterwarnings("ignore")
def get_rolling_IS(dates, spread5y, CGcds):
    df = pd.DataFrame({'spread5y':spread5y,'CGcds':CGcds},index=dates)
    vecm_mod = VECM(df, coint_rank=1, k_ar_diff=250, deterministic='ci')
    vecm_res = vecm_mod.fit()
    cov_mat = pd.DataFrame(vecm_res.resid).cov()
    if cov_mat.isna().values().any():
        print('hi',end=",")
        return 0, 0
    chols = np.linalg.cholesky(cov_mat)
    m11, m12, m22 =chols[0,0], chols[1,0], chols[1,1]
    a1, a2 = vecm_res.alpha
    gamma1 = (a1/(a1-a2))[0]
    gamma2 = (a2/(a2-a1))[0]
    denom = ((gamma1*m11 + gamma2*m12)**2 + (gamma2*m22)**2)
    IS1 = (gamma1*m11 + gamma2)**2/denom
    IS2 = (gamma2*m22)**2/denom
    return IS1, IS2

IS1, IS2 = npext.rolling_apply(get_rolling_IS, 270, df_prcdis.index, df_prcdis.spread5y, df_prcdis.CGcds)
df_prcdis['IS1'] = IS1
df_prcdis['IS1'] = IS2

# Strategy if market CDS spread is high
trigger1 = np.zeros(dfboeing.shape[0], dtype='int')
pos1 = np.zeros(dfboeing.shape[0], dtype='int')

def arb_ret(df, maxholdingdays, threshold):
    df_calc = df.copy()
    for i in range(df_calc.shape[0]):
        if diff_z[i] > threshold and np.isfinite(diff_z[i]) and (i > 0 and trigger1[i-1] == 0):
            trigger1[i] = 1
            pos1[i] = 1
        if i > 0 and pos1[i-1] != 0 and pos1[i-1] < maxholdingdays:
            pos1[i] = pos1[i-1] + 1
        if pos1[i] != 1:
            trigger1[i] = 0
# Strategy if market CDS spread is low
    trigger2 = np.zeros(df_calc.shape[0], dtype='int')
    pos2 = np.zeros(df_calc.shape[0], dtype='int')
    for i in range(df_calc.shape[0]):
        if diff_z[i] < -threshold and np.isfinite(diff_z[i]) and (i > 0 and trigger1[i-1] == 0):
            trigger2[i] = 1
            pos2[i] = 1
        if i > 0 and pos2[i-1] != 0 and pos2[i-1] < maxholdingdays:
            pos2[i] = pos2[i-1] + 1
        if pos2[i] != 1:
            trigger2[i] = 0
    df_calc['trigger1'] = trigger1
    df_calc['trigger2'] = trigger2
    df_calc['pos1'] = pos1
    df_calc['pos2'] = pos2
# Calculate cumulative stock return in the next month
    df_calc['stkret'] = df_calc['S']/df_calc['S'].shift(1)-1
    df_calc['fret1m'] = df_calc.rolling(window=maxholdingdays)['stkret'].apply(lambda x: 100*((1+x).prod()-1)).shift(-maxholdingdays)
# Calculate cumulative CDS return in the next month
    df_calc['dcdsret'] = (df_calc['spread5y'] - df_calc['spread5y'].shift(1)) * df_calc['rpv01']
    df_calc['fcds1m'] = df_calc.rolling(window=maxholdingdays)['dcdsret'].sum().shift(-maxholdingdays)
# Calculate arbitrage portfolio return
    df_calc['Arbitrage return'] = np.nan
    mask1 = df_calc['trigger1'] == 1
    df_calc.loc[mask1, 'caparb'] = df_calc.loc[mask1, 'fret1m'] * df_calc.loc[mask1, 'hedge ratio (reg)'] - df_calc.loc[mask1, 'fcds1m']
    mask2 = df_calc['trigger2'] == 1
    df_calc.loc[mask2, 'caparb'] = -df_calc.loc[mask2, 'fret1m'] * df_calc.loc[mask2, 'hedge ratio (reg)'] + df_calc.loc[mask2, 'fcds1m']
# Print results
    outtab1 = df_calc.dropna(subset=['caparb']).loc[mask1, 'caparb'].describe().to_frame('High market CDS')
    outtab2 = df_calc.dropna(subset=['caparb']).loc[mask2, 'caparb'].describe().to_frame('Low market CDS')
    outtab = df_calc.dropna(subset=['caparb'])['caparb'].describe().to_frame('Arbitrage return (%)')
    outtaball = pd.concat([outtab1, outtab2, outtab], axis=1)
# print(outtaball.to_markdown(tablefmt='psql'))
    temp1 = dfboeing.dropna(subset=['caparb']).loc[mask2, ['date', 'caparb']]
# dfboeing.to_csv('Boeing_ANSWER.csv')
    return (df_calc, outtaball)
#    return temp1






