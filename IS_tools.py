"""
@author jeffrey
"""
import statsmodels.api as sm
import functools
from statsmodels.tsa.vector_ar import vecm
import pandas as pd
import numpy as np


def adfTest(x,autolag="BIC" ,regression='nc'):
    adf_results = sm.tsa.adfuller(x, regression=regression, autolag=autolag)
    title = ["statistic", "p", "usedlag"]
    _t = {title[i]: v for i, v in enumerate(adf_results[:len(title)])}
    _t.update(adf_results[4])
    return _t


batchADFTest = functools.partial(adfTest,autolag="BIC",regression="ct")


def johansen_Test(data,det_order,lagged_diff):

    results = vecm.coint_johansen(data, det_order, lagged_diff)
    format_res = []
    format_res.append(results.eig)
    format_res.append(results.lr2)
    cols = ["eig","max eig",'90%',"95%","90%"]
    df = pd.DataFrame(np.hstack((np.array(format_res).T,results.cvm)))
    df.columns=cols
    df.index=["H(0)","H(1)"]
    return df


def beforehand_test(df):

    from scipy import stats
    from statsmodels.sandbox.stats.diagnostic import acorr_ljungbox, het_arch

    def adf_sets(xs):
        adfs = {}
        for c in xs:
            adf = adfTest(xs[c])
            adfs[c] = adf
        return adfs

    def statistic(df,l_ar_order=[1,15]):

        cols = df.columns
        from scipy import stats
        skew = stats.skew(df)
        kuro = stats.kurtosis(df)

        ar_tvalues = {}
        for _order in l_ar_order:
            ar_tvalues["AR(%s)"%_order] = [sm.tsa.AR(df[c]).fit(_order).tvalues[-1] for c in df]

        ar_tvalues = pd.DataFrame(ar_tvalues,index=cols).T

        desb2 = pd.DataFrame(np.vstack((skew, kuro)), columns=cols, index=['skew', "kuro"])

        desb = pd.concat([df.describe(),desb2,ar_tvalues],axis=0)

        return desb

    accor = {}

    for i, c in enumerate(df):
        s = df[c]
        accor[c] = pd.Series([het_arch(s)[-1], \
                              (np.hsplit(np.array(acorr_ljungbox(s, 15)),15)[-1][-1])[0],
                              stats.jarque_bera(s)[1]],
                             index=['ARCH', 'LBQ', 'JB'])

    return pd.concat([statistic(df),\
                      pd.DataFrame(adf_sets(df)).loc[['p']].rename({"p":"ADF(p)"}),\
                      pd.DataFrame(accor)])


def pair_test(df):
    """
    :param df:
    :return:
    """

    if not df.shape[1] == 2:
        raise Exception("ndarray or dataframe must have 2 columns")

    r1 = df.iloc[:,0]
    r2 = df.iloc[:,1]
    from scipy import stats
    # 等均值检验
    em = stats.ttest_ind(r1,r2)
    # stats.bartlett()
    # 等方差检验
    ev = stats.levene(r1,r2)
    # 同分布检验
    ed = stats.ks_2samp(r1,r2)

    # 联合检验
    tests = pd.DataFrame(np.vstack((em,ev,ed)),index=['T-test','Levene','KS'],columns=['statistic','p'])

    return tests


def coint_test(df,max_lag=60,signif=0.01,deterministic='ci',output = "all",rule ="bic"):

    orders = vecm.select_order(np.array(df),max_lag,deterministic=deterministic)

    k_ar_diff = orders.__getattribute__(rule)

    k_ar_diff = k_ar_diff if k_ar_diff>0 else 1

    results = vecm.coint_johansen(df,0,k_ar_diff)

    possible_signif_values = [0.1, 0.05, 0.01]

    if output == "all":

        trace = np.vstack((results.cvt.T,results.lr1))
        MaxEig = np.vstack((results.cvm.T,results.lr2))

        return {"trace":trace,"MaxEig":MaxEig},k_ar_diff

    elif output == "standard":

        signif_index = possible_signif_values.index(signif)

        # crit_vals = results.cvt
        crit_vals = np.vstack((results.cvt[:,signif_index],results.cvm[:,signif_index])).T

        test_stat = np.vstack((results.lr1,results.lr2)).T

        masks = test_stat>crit_vals

        test_stat = np.round(pd.DataFrame(test_stat,columns=['trace',' Maxeig'],index=np.arange(1,df.shape[1]+1,1)),4)
        return test_stat.where(~masks,test_stat.astype(str)+"*"),orders
    else:
        # for i in range(len(possible_signif_values)):
        #     results.cvt[:,i]<results.lr1

        if any(results.lr1[0]>results.cvt[0]) or any(results.lr2[0]>results.cvm[0]):
            return True,k_ar_diff
        else:
            return False,k_ar_diff


