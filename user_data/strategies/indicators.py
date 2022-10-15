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

# -------------------------------- INDICATORS --------------------------------


def zlsma(df, period=50, offset=0):
    src = df['close']
    lsma = ta.LINEARREG(src, period, offset)
    lsma2 = ta.LINEARREG(lsma, period, offset)
    eq = lsma - lsma2
    return lsma + eq


def chandelier_exit(df, timeperiod=14, multiplier=2):
    atr = multiplier * ta.ATR(df, period=timeperiod)
    close = df['close']

    longStop = highest(close, timeperiod) - atr
    longStopPrev = nz(longStop)

    shortStop = lowest(close, timeperiod) + atr
    shortStopPrev = nz(shortStop)

    signal = pd.Series(0, index=df.index)
    signal.loc[close > shortStopPrev] = 1
    signal.loc[close < longStopPrev] = -1

    return signal
