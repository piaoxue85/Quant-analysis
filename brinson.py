def fillHolding(d, nextd, tradeDates, df_holdings, df_holdingCash, df_totalReturnFactor):
    if nextd > d:
        holdingDates = tradeDates[(tradeDates >= d) & (tradeDates <= nextd)]

        df_holdings.ix[holdingDates] = np.tile(df_holdings.ix[d], (len(holdingDates), 1))

        df_holdingCash.ix[holdingDates] = np.tile(df_holdingCash.ix[d], (len(holdingDates), 1))

        df_holdings.ix[holdingDates] = df_holdings.ix[holdingDates] * (
        df_totalReturnFactor.ix[holdingDates] / df_totalReturnFactor.ix[d])

    result = {'holdings': df_holdings, 'holdingCash': df_holdingCash}

    return result


def SIMPLE_SIMULATE_DAILY_TRADE_CHN_STK(beginDate, endDate, initialHolding, df_targetPortfolioWgt,

                                        df_markToMarketPrice, df_totalReturnFactor, df_executePrice,

                                        df_execPriceReturn, df_tradeVolume, dict_tradingParam,

                                        dict_additionalTs):
    beginDate = beginDate[0]

    endDate = endDate[0]

    df_targetPortfolioWgt = df_targetPortfolioWgt.asMatrix()

    df_markToMarketPrice = df_markToMarketPrice.asMatrix()

    df_totalReturnFactor = df_totalReturnFactor.asMatrix()

    df_executePrice = df_executePrice.asMatrix()

    df_execPriceReturn = df_execPriceReturn.asMatrix()

    df_tradeVolume = df_tradeVolume.asMatrix()

    cashSymbol = gsUtils.getCashGid()

    godGid = gsUtils.getGodGid()

    allDates = df_markToMarketPrice.index

    if len(allDates[(allDates >= beginDate) & (allDates <= endDate)]) < 1:
        raise ValueError('no trading date falls between begindate and enddate')

    endDate = allDates[allDates <= endDate][-1]

    if beginDate > endDate:
        raise ValueError('beginDate should be less than endDate')

    initHldIsCash = True

    if isinstance(initialHolding, gftIO.GftTable):

        df_initialHolding = initialHolding.asMatrix()

        if df_initialHolding.shape[0] < 1:
            raise ValueError('no init holding is provided')

        initHldIsCash = False

        df_initialHolding = df_initialHolding.ix[-1:]

        beginDate = gsUtils.alignDate(df_initialHolding.index[0], allDates, method='ffill')

        if pd.isnull(beginDate):
            raise ValueError('do not have close price for the date of initHld')

        df_initialHolding.index = [beginDate]

    else:

        beginDate = gsUtils.alignDate(beginDate, allDates, method='bfill')

        if pd.isnull(beginDate):
            raise ValueError('beginDate should be less than the last trading date')

    if (df_targetPortfolioWgt < 0).any(axis=1).any():
        raise ValueError('Do not support stock short selling and cash borrowing')

    if (round(df_targetPortfolioWgt.sum(1), 4) > 1).any():
        raise ValueError('Total weight is greater than 1.')

    df_targetPortfolioWgt = df_targetPortfolioWgt.dropna(axis=[0, 1], how='all')

    sigDates = df_targetPortfolioWgt.index

    rebDates = sigDates

    if len(sigDates) > 0:

        execDelayPeriods = gsUtils.getParm(dict_tradingParam, "execDelayPeriods", 0)

        if execDelayPeriods > 0:

            idxs = np.array(gsUtils.alignDate(sigDates, allDates, method='bfill', returnidx=True), dtype=float)

            idxs = idxs + execDelayPeriods

            idxs[idxs > len(allDates)] = NA

            idxs[idxs == np.append(idxs[1:], NA)] = NA

            idxs_nonnan_flag = np.logical_not(np.isnan(idxs))

            if sum(idxs_nonnan_flag) < 1:
                raise ValueError("no trade date after the execute delay shift")

            df_targetPortfolioWgt = df_targetPortfolioWgt.ix[idxs_nonnan_flag]

            rebDates = allDates[np.array(idxs[idxs_nonnan_flag], dtype=int)]

            df_targetPortfolioWgt.index = rebDates

        if len(rebDates) > 0:

            if initHldIsCash:

                if gsUtils.getParm(dict_tradingParam, "shiftBeginDateToSignal", False):
                    beginDate = rebDates[rebDates >= beginDate][0]

                if pd.isnull(beginDate):
                    raise ValueError('beginDate is null after shift')

    tradeDates = allDates[(allDates >= beginDate) & (allDates <= endDate)]

    beginDate = tradeDates[0]

    endDate = tradeDates[-1]

    if beginDate > endDate:
        raise ValueError("Begin date is larger than end date after the date processing!")

    rebDates = rebDates[(rebDates >= beginDate) & (rebDates <= endDate)]

    df_targetPortfolioWgt = df_targetPortfolioWgt.ix[rebDates]

    allSymbols = np.unique(df_markToMarketPrice.columns)

    portfolioSymbols = np.unique(np.setdiff1d(df_targetPortfolioWgt.columns, cashSymbol))

    holdingSymbols = np.array([])

    if not (initHldIsCash):
        holdingSymbols = np.unique(np.setdiff1d(df_initialHolding.columns, cashSymbol))

    if len(np.setdiff1d(holdingSymbols, allSymbols)) > 0:
        raise ValueError("Initial Portfolio has non A-share stocks!")

    if len(np.setdiff1d(portfolioSymbols, allSymbols)) > 0:
        raise ValueError("Target Portfolio has non A-share stocks!")

    allSymbols = np.unique(
        np.setdiff1d(np.intersect1d(allSymbols, np.append(holdingSymbols, portfolioSymbols)), cashSymbol))

    priceDates = allDates[(allDates >= beginDate - datetime.timedelta(days=20)) & (allDates <= endDate)]

    df_markToMarketPrice = df_markToMarketPrice.reindex(priceDates, allSymbols, fill_value=NA)

    df_totalReturnFactor = df_totalReturnFactor.reindex(priceDates, allSymbols, fill_value=1.).fillna(1.)

    df_executePrice = df_executePrice.reindex(priceDates, allSymbols, fill_value=NA)

    df_execPriceReturn = df_execPriceReturn.reindex(priceDates, allSymbols, fill_value=NA)

    df_tradeVolume = df_tradeVolume.reindex(priceDates, allSymbols, fill_value=0.)

    if initHldIsCash:
        df_initialHolding = pd.DataFrame(initialHolding, index=[beginDate], columns=[cashSymbol])

    df_initialHolding = df_initialHolding.reindex(columns=np.append(allSymbols, cashSymbol)).fillna(0.)

    df_initialHoldingCash = df_initialHolding.ix[:, cashSymbol]

    df_initialHolding = df_initialHolding.ix[:, allSymbols]

    initHldValue = float((df_initialHolding * df_markToMarketPrice.ix[df_initialHolding.index]).sum(axis=1)) + \
                   df_initialHoldingCash.ix[0, 0]

    df_targetPortfolioWgt = df_targetPortfolioWgt.reindex(rebDates, allSymbols, fill_value=0.).fillna(0.)

    df_buyVolume = df_tradeVolume.copy().fillna(0)

    df_sellVolume = df_buyVolume.copy()

    if gsUtils.getParm(dict_tradingParam, "canTradeOnSuspend", 0) > 0:
        df_buyVolume[df_buyVolume < 1] = np.inf

        df_sellVolume[df_sellVolume < 1] = np.inf

    riseLimitThres = gsUtils.getParm(dict_tradingParam, "riseLimitThres", 0)

    if riseLimitThres > 0:
        riseLimit = df_execPriceReturn > riseLimitThres

        df_buyVolume[riseLimit] = 0

        df_sellVolume[riseLimit & (df_sellVolume > 0)] = np.inf

    fallLimitThres = gsUtils.getParm(dict_tradingParam, "fallLimitThres", 0)

    if fallLimitThres < 0:
        fallLimit = df_execPriceReturn < fallLimitThres

        df_buyVolume[fallLimit & (df_buyVolume > 0)] = np.inf

        df_sellVolume[fallLimit] = 0

    volumeLimitPct = gsUtils.getParm(dict_tradingParam, "volumeLimitPct", 0)

    if volumeLimitPct > 0:

        df_buyVolume = df_buyVolume * volumeLimitPct

        df_sellVolume = df_sellVolume * volumeLimitPct

    else:

        df_buyVolume[df_buyVolume > 0] = np.inf

        df_sellVolume[df_sellVolume > 0] = np.inf

    lotSize = gsUtils.getParm(dict_tradingParam, "lotSize", 0)

    df_buyVolume = gsUtils.roundToLot(df_buyVolume, lotSize)

    df_sellVolume = gsUtils.roundToLot(df_sellVolume, lotSize)

    buyCommission = gsUtils.getParm(dict_tradingParam, "buyCommission", 0)

    sellCommission = gsUtils.getParm(dict_tradingParam, "sellCommission", 0)

    df_holdings = pd.DataFrame(0., index=tradeDates, columns=allSymbols)

    df_weights = df_holdings.copy()

    df_execution = df_holdings.copy()

    df_holdingCash = pd.DataFrame(0., index=tradeDates, columns=cashSymbol)

    df_portfolioValue = pd.DataFrame(0., index=tradeDates, columns=godGid)

    df_cumRets = df_portfolioValue.copy()

    df_singlePeriodRets = df_portfolioValue.copy()

    df_turnoverPct = df_portfolioValue.copy()

    d = tradeDates[0]

    df_holdings.ix[d] = df_initialHolding.ix[d]

    df_holdingCash.ix[d] = df_initialHoldingCash.ix[0, 0]

    if len(rebDates) < 1:

        nextd = tradeDates[-1]

        ls_adjustedHoldings = fillHolding(d, nextd, tradeDates, df_holdings, df_holdingCash, df_totalReturnFactor)

        df_holdings = ls_adjustedHoldings['holdings']

        df_holdingCash = ls_adjustedHoldings['holdingCash']

    else:

        nextd = rebDates[0]

        ls_adjustedHoldings = fillHolding(d, nextd, tradeDates, df_holdings, df_holdingCash, df_totalReturnFactor)

        df_holdings = ls_adjustedHoldings['holdings']

        df_holdingCash = ls_adjustedHoldings['holdingCash']

        for i in range(len(rebDates)):

            d = rebDates[i]

            s_currentHoldingValue = df_holdings.ix[d] * df_executePrice.ix[d]

            totalValue = s_currentHoldingValue.sum() + df_holdingCash.ix[d, 0]

            s_currentHoldingWgt = s_currentHoldingValue / totalValue

            s_targetHoldingWgt = df_targetPortfolioWgt.ix[d]

            targetHoldingCashWgt = 1.0 - s_targetHoldingWgt.sum()

            s_orderWgt = s_targetHoldingWgt - s_currentHoldingWgt

            s_sellOrderWgt = s_orderWgt.copy()

            s_sellOrderWgt[s_sellOrderWgt > 0.] = 0.

            s_buyOrderWgt = s_orderWgt.copy()

            s_buyOrderWgt[s_buyOrderWgt < 0.] = 0.

            cashAvail = df_holdingCash.ix[d, 0]

            if (s_sellOrderWgt < 0).any():
                s_sellOrder = gsUtils.roundToLot(
                    s_sellOrderWgt / s_currentHoldingWgt.where(s_currentHoldingWgt > 0, 1.0) * df_holdings.ix[d],
                    lotSize)

                s_sellOrder = s_sellOrder.where(s_targetHoldingWgt > 0, -df_holdings.ix[d])

                s_sellExecution = s_sellOrder.copy()

                s_sellExecution = -pd.concat([s_sellExecution.fillna(0).abs(), df_sellVolume.ix[d]], axis=1).min(axis=1)

                cashAvail = cashAvail + (s_sellExecution.abs() * df_executePrice.ix[d]).sum() * (1 - sellCommission)

                df_execution.ix[d] += s_sellExecution

                df_holdings.ix[d] += s_sellExecution

            if (s_buyOrderWgt > 0).any():

                canBuyWgt = cashAvail / totalValue - targetHoldingCashWgt

                if canBuyWgt > 0:
                    s_buyOrder = gsUtils.roundToLot((min(canBuyWgt / s_buyOrderWgt.sum(),
                                                         1.0) * s_buyOrderWgt * totalValue / (1 + buyCommission) /
                                                     df_executePrice.ix[d]).fillna(0), lotSize)

                    s_buyExecution = s_buyOrder.copy()

                    s_buyExecution = pd.concat([s_buyExecution.fillna(0), df_buyVolume.ix[d]], axis=1).min(axis=1)

                    cashAvail = cashAvail - (s_buyExecution.abs() * df_executePrice.ix[d]).sum() * (1 + buyCommission)

                    df_execution.ix[d] += s_buyExecution

                    df_holdings.ix[d] += s_buyExecution

            df_holdingCash.ix[d] = cashAvail

            df_turnoverPct.ix[d] < - (df_execution.ix[d].abs() * df_executePrice.ix[d]).sum() / totalValue

            if i < (len(rebDates) - 1):

                nextd = rebDates[i + 1]

            else:

                nextd = tradeDates[-1]

            ls_adjustedHoldings = fillHolding(d, nextd, tradeDates, df_holdings, df_holdingCash, df_totalReturnFactor)

            df_holdings = ls_adjustedHoldings['holdings']

            df_holdingCash = ls_adjustedHoldings['holdingCash']

    df_portfolioValue.ix[:, 0] = (df_holdings * df_markToMarketPrice.ix[tradeDates]).sum(axis=1) + df_holdingCash.ix[:,
                                                                                                   0]

    df_weights = (df_holdings * df_markToMarketPrice.ix[tradeDates]).div(df_portfolioValue.ix[:, 0], axis=0)

    df_cumRets = df_portfolioValue / initHldValue - 1

    df_singlePeriodRets = df_portfolioValue / df_portfolioValue.shift(1) - 1

    df_singlePeriodRets.ix[0, 0] = df_portfolioValue.ix[0, 0] / initHldValue - 1

    result = {}

    result[gsConst.Const.Holding] = pd.concat([df_holdings.replace(0, NA), df_holdingCash], axis=1)

    result[gsConst.Const.PortfolioValue] = df_portfolioValue

    result[gsConst.Const.Weights] = df_weights.replace(0, NA)

    result[gsConst.Const.SinglePeriodReturn] = df_singlePeriodRets

    result[gsConst.Const.CumulativeReturn] = df_cumRets

    result[gsConst.Const.Turnover] = df_turnoverPct

    return result


import pandas as pd

from lib.gftTools import gftIO

from lib.gftTools import gsUtils

import numpy as np

from numpy import NaN

import warnings

warnings.filterwarnings("ignore")


def BrinsonAttribution(context, start_date, end_date, portfolio_wt, benchmark, close_price):
    '''

    parameter

    -----

    start_date: DateTimeIndex

    end_date: DateTimeIndex

    portfolio_wt: OOTV, portfolio weight with industry info,filling weight

    benchmark: OOTV, benchmark weight with industry info

    close_price：后复权收盘价,OTV



    '''

    if start_date >= end_date:
        raise ValueError('startdate should be earlier than enddate!')

    df_l_benchmark = benchmark.asColumnTab()

    df_l_portfolio_wt = portfolio_wt.asColumnTab()

    df_close_price = close_price.asMatrix()

    # rename

    df_l_portfolio_wt.columns = ['date', 'symbol', 'weight', 'ind']

    df_l_benchmark.columns = ['date', 'symbol', 'weight', 'ind']

    '''data preparation '''

    # prepare for slicing dates

    df_l_portfolio_wt = df_l_portfolio_wt.dropna(subset=['weight'])

    df_l_portfolio_wt = df_l_portfolio_wt[df_l_portfolio_wt['weight'] != 0]

    df_l_benchmark = df_l_benchmark.dropna(subset=['weight'])

    df_close_price = df_close_price.dropna(axis=1, how='all')

    businessdays = df_close_price.index

    if (min(businessdays) > end_date) | (max(businessdays) < start_date):
        raise ValueError('dates should be reset')

    end = max(businessdays[businessdays <= end_date])

    # prepare for slicing symbols

    portfolioSymbols = df_l_portfolio_wt['symbol'].unique()

    benchmarkSymbols = df_l_benchmark['symbol'].unique()

    allSymbols = np.unique(np.union1d(portfolioSymbols, benchmarkSymbols))

    cashSymbol = gsUtils.getCashGid()[0]

    # drop cash in portfolio_wt

    df_l_portfolio_wt = df_l_portfolio_wt[df_l_portfolio_wt['symbol'] != cashSymbol]

    df_l_portfolio_wt = df_l_portfolio_wt.sort_values('date')

    # slice dates

    start_date = gsUtils.alignDate(start_date, businessdays)

    begin = max(start_date, min(df_l_portfolio_wt['date']))

    df_l_portfolio_wt = df_l_portfolio_wt[(df_l_portfolio_wt['date'] >= begin) & (df_l_portfolio_wt['date'] <= end)]

    alldates = pd.to_datetime(df_l_portfolio_wt['date'].unique())

    alldates_benchmark = pd.to_datetime(df_l_benchmark['date'].unique())

    # if benchmark dates missing, find common dates between portfolio and benchmark

    if len(np.setdiff1d(alldates, alldates_benchmark)) > 0:
        df_l_portfolio_wt = df_l_portfolio_wt[df_l_portfolio_wt['date'].isin(alldates_benchmark)]

    df_l_benchmark = df_l_benchmark[(df_l_benchmark['date'] >= begin) & (df_l_benchmark['date'] <= end)]

    df_l_benchmark = df_l_benchmark.sort_values('date')

    # slice dates for closeprice

    priceDates = pd.to_datetime(df_l_benchmark['date'].unique())

    if not (end_date in priceDates):
        priceDates = priceDates.append(pd.Index([end]))  # add end_date if not included

    priceDates = priceDates.sort_values()

    # calculate return

    df_close_price = df_close_price.reindex(priceDates, allSymbols, fill_value=NaN)

    df_tot_return_index = df_close_price.pct_change(1).shift(-1).fillna(0.0)

    df_l_tot_return_index = gftIO.convertMatrix2ColumnTab(df_tot_return_index)

    df_l_tot_return_index.columns = ['date', 'symbol', 'return']

    df_l_tot_return_index = df_l_tot_return_index.sort_values('date')

    # calculate industry weight&industry return

    df_l_portwgt = df_l_portfolio_wt.groupby(['ind', 'date'], as_index=False).sum()  # industry weight

    df_l_portwgt = df_l_portwgt.sort_values('date')

    df_l_benchmark_wgt = df_l_benchmark.groupby(['ind', 'date'], as_index=False).sum()  # industry weight

    df_l_benchmark_wgt = df_l_benchmark_wgt.sort_values('date')

    df_l_portfolio_wt = pd.merge(df_l_portfolio_wt, df_l_portwgt, how='left', on=['ind', 'date'],
                                 suffixes=('', '_ind'))  # get industry weight

    df_l_benchmark = pd.merge(df_l_benchmark, df_l_benchmark_wgt, how='left', on=['ind', 'date'],
                              suffixes=('', '_ind'))  # get industry weight

    df_l_portfolio_wt['weight_ind_pct'] = df_l_portfolio_wt['weight'] / df_l_portfolio_wt[
        'weight_ind']  # stock weight/industry weight

    df_l_benchmark['weight_ind_pct'] = df_l_benchmark['weight'] / df_l_benchmark['weight_ind']

    df_l_portfolio_wt = pd.merge(df_l_portfolio_wt, df_l_tot_return_index, how='left', on=['symbol', 'date'],
                                 sort=False)

    df_l_benchmark = pd.merge(df_l_benchmark, df_l_tot_return_index, how='left', on=['symbol', 'date'], sort=False)

    df_l_portfolio_wt['return_pcg'] = df_l_portfolio_wt['weight_ind_pct'] * df_l_portfolio_wt['return']  # calculate w*r

    df_l_benchmark['return_pcg'] = df_l_benchmark['weight_ind_pct'] * df_l_benchmark['return']

    df_l_portret = df_l_portfolio_wt.groupby(['ind', 'date'], as_index=False)['return_pcg'].sum()  # get industry return

    df_l_benchret = df_l_benchmark.groupby(['ind', 'date'], as_index=False)['return_pcg'].sum()

    df_l_portret = df_l_portret.sort_values('date')

    df_l_benchret = df_l_benchret.sort_values('date')

    # Prepare for Q1,Q2,Q3,Q4( Q1:基准组合 Q2：积极资产配置组合 Q3.积极股票选择组合 Q4.实际组合 )

    Q1 = pd.merge(df_l_benchmark_wgt, df_l_benchret, how='left', on=['ind', 'date'])

    Q2 = pd.merge(df_l_portwgt, df_l_benchret, how='left', on=['ind', 'date'])

    Q3 = pd.merge(df_l_benchmark_wgt, df_l_portret, how='left', on=['ind', 'date'])

    Q4 = pd.merge(df_l_portwgt, df_l_portret, how='left', on=['ind', 'date'])

    # calculate cumulative return, p,aa,ss,b分别是基金实际组合、积极资产配置组合、积极股票选择组合以及基准组合的k期复合收益率

    Q1['b'] = Q1['weight'].mul(Q1['return_pcg'], axis=0)

    Q2['aa'] = Q2['weight'].mul(Q2['return_pcg'], axis=0)

    Q3['ss'] = Q3['weight'].mul(Q3['return_pcg'], axis=0)

    Q4['p'] = Q4['weight'].mul(Q4['return_pcg'], axis=0)

    b = Q1.groupby(['date'])['b'].sum()

    aa = Q2.groupby(['date'])['aa'].sum()

    ss = Q3.groupby(['date'])['ss'].sum()

    p = Q4.groupby(['date'])['p'].sum()

    b = np.cumprod(1 + b) - 1

    aa = np.cumprod(1 + aa) - 1

    ss = np.cumprod(1 + ss) - 1

    p = np.cumprod(1 + p) - 1

    b = pd.Series.to_frame(b).fillna(0)

    aa = pd.Series.to_frame(aa).fillna(0)

    ss = pd.Series.to_frame(ss).fillna(0)

    p = pd.Series.to_frame(p).fillna(0)

    Q1 = pd.merge(Q1, b, how='left', left_on=['date'], right_index=True)

    Q2 = pd.merge(Q2, aa, how='left', left_on=['date'], right_index=True)

    Q3 = pd.merge(Q3, ss, how='left', left_on=['date'], right_index=True)

    Q4 = pd.merge(Q4, p, how='left', left_on=['date'], right_index=True)

    Q1['b_y'] = Q1.groupby(['ind'])['b_y'].apply(lambda x: x.shift(1)).fillna(0)

    Q2['aa_y'] = Q2.groupby(['ind'])['aa_y'].apply(lambda x: x.shift(1)).fillna(0)

    Q3['ss_y'] = Q3.groupby(['ind'])['ss_y'].apply(lambda x: x.shift(1)).fillna(0)

    Q4['p_y'] = Q4.groupby(['ind'])['p_y'].apply(lambda x: x.shift(1)).fillna(0)

    Q1['b'] = Q1['weight'] * Q1['return_pcg'] * (Q1['b_y'] + 1)

    Q2['aa'] = Q2['weight'] * Q2['return_pcg'] * (Q2['aa_y'] + 1)

    Q3['ss'] = Q3['weight'] * Q3['return_pcg'] * (Q3['ss_y'] + 1)

    Q4['p'] = Q4['weight'] * Q4['return_pcg'] * (Q4['p_y'] + 1)

    q1 = Q1[['ind', 'date', 'b']]

    q2 = Q2[['ind', 'date', 'aa']]

    q3 = Q3[['ind', 'date', 'ss']]

    q4 = Q4[['ind', 'date', 'p']]

    summary0 = pd.merge(q1, q2, how='left', on=['ind', 'date'])

    summary1 = pd.merge(summary0, q3, how='left', on=['ind', 'date'])

    summary2 = pd.merge(summary1, q4, how='left', on=['ind', 'date'])

    summary = summary2.copy()

    summary.fillna(0, inplace=True)

    summary['AR'] = summary['aa'] - summary['b']  # Pure Sector Allocation

    summary['SR'] = summary['ss'] - summary['b']  # Within-Sector selection return

    summary['IR'] = summary['p'] - summary['ss'] - summary['aa'] + summary['b']

    summary['TotalValueAdded'] = summary['AR'] + summary['SR'] + summary['IR']

    summary = pd.merge(df_l_benchmark_wgt, summary, how='outer', on=['ind', 'date'])

    summary = pd.merge(summary, df_l_portwgt, how='outer', on=['ind', 'date'])

    summary.fillna(0, inplace=True)

    summary.columns = ['ind', 'date', 'bmweight', 'cumulativeBenRet', 'portreturncontribution',

                       'bmreturncontribution', 'cumulativePortRet', 'AssetAllocation', 'StockSelection', 'Interaction',
                       'TotalValueAdded', 'portweight']

    result = summary

    result = result[['ind', 'date', 'portweight', 'bmweight', 'portreturncontribution',

                     'bmreturncontribution', 'cumulativePortRet', 'cumulativeBenRet', 'AssetAllocation',
                     'StockSelection', 'Interaction', 'TotalValueAdded']]

    portfolio_length = len(df_l_portwgt['date'].unique())

    benchmark_length = len(df_l_benchmark_wgt['date'].unique())

    final_result = {'result': result, 'portfolio_length': portfolio_length, 'benchmark_length': benchmark_length}

    return final_result