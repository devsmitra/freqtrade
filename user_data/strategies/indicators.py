from numpy import nan
import talib.abstract as ta
import numpy as np
import pandas as pd

# -------------------------------- UTILS --------------------------------


def minimum(s1, s2) -> float:
    return np.minimum(s1, s2)


def maximum(s1, s2) -> float:
    return np.maximum(s1, s2)


def highestbars(series, n):
    return ta.MAXINDEX(series, n) - n + 1


def lowestbars(series, n):
    return ta.MININDEX(series, n) - n + 1


def nz(series):
    return series.fillna(method='backfill')


def na(series):
    return series is nan


def highest(series, n):
    return series.rolling(n).max()


def lowest(series, n):
    return series.rolling(n).min()


def series(value, df):
    return pd.series(value, index=df.index)

# -------------------------------- INDICATORS --------------------------------


def zlsma(df, period=50, offset=0, column='close'):
    src = df[column]
    lsma = ta.LINEARREG(src, period, offset)
    lsma2 = ta.LINEARREG(lsma, period, offset)
    eq = lsma - lsma2
    return lsma + eq


def chandelier_exit(df, timeperiod=14, multiplier=2, column='close'):
    close = df[column]
    high = df['ha_high']
    low = df['ha_low']
    atr = multiplier * ta.ATR(high, low, close, timeperiod=timeperiod)

    longStop = highest(close, timeperiod) - atr
    longStopPrev = nz(longStop)

    # shortStop = lowest(close, timeperiod) + atr
    # shortStopPrev = nz(shortStop)

    signal = pd.Series(1, index=df.index)
    # signal.loc[close > shortStopPrev] = 1
    signal.loc[close < longStopPrev] = -1

    return signal


def volatility_osc(df, timeperiod=100):
    spike = df['ha_close'] - df['ha_open']
    x = ta.STDDEV(spike, timeperiod)
    y = ta.STDDEV(spike, timeperiod) * -1
    return pd.DataFrame(index=df.index, data={'upper': x, 'lower': y, 'spike': spike})


def calculate_money_flow_volume_series(df):
    mfv = df['volume'] * (2 * df['close'] - df['high'] - df['low']) / \
                                    (df['high'] - df['low'])
    return mfv


def calculate_money_flow_volume(df, n: int = 20):
    return calculate_money_flow_volume_series(df).rolling(n).sum()


def calculate_chaikin_money_flow(df, n: int = 20):
    return calculate_money_flow_volume(df, n) / df['volume'].rolling(n).sum()


def poki(df, timeperiod=100, column='close'):
    pass
