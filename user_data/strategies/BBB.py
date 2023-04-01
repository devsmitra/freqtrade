from numpy import nan
import numpy as np
import pandas as pd
import freqtrade.vendor.qtpylib.indicators as qtpylib
import math
import talib.abstract as ta

class Settings:
    def __init__(self, source, neighborsCount, maxBarsBack, featureCount, colorCompression, showExits, useDynamicExits):
        self.source = source
        self.neighborsCount = neighborsCount
        self.maxBarsBack = maxBarsBack
        self.featureCount = featureCount
        self.colorCompression = colorCompression
        self.showExits = showExits
        self.useDynamicExits = useDynamicExits

class Label:
    def __init__(self, long, short, neutral):
        self.long = long
        self.short = short
        self.neutral = neutral

class FeatureArrays:
    def __init__(self, f1, f2, f3, f4, f5):
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
        self.f5 = f5

class FeatureSeries:
    def __init__(self, f1, f2, f3, f4, f5):
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
        self.f5 = f5

class MLModel:
    def __init__(self, firstBarIndex, trainingLabels, loopSize, lastDistance, distancesArray, predictionsArray, prediction):
        self.firstBarIndex = firstBarIndex
        self.trainingLabels = trainingLabels
        self.loopSize = loopSize
        self.lastDistance = lastDistance
        self.distancesArray = distancesArray
        self.predictionsArray = predictionsArray
        self.prediction = prediction

class FilterSettings:
    def __init__(self, useVolatilityFilter, useRegimeFilter, useAdxFilter, regimeThreshold, adxThreshold):
        self.useVolatilityFilter = useVolatilityFilter
        self.useRegimeFilter = useRegimeFilter
        self.useAdxFilter = useAdxFilter
        self.regimeThreshold = regimeThreshold
        self.adxThreshold = adxThreshold

class Filter:
    def __init__(self, volatility, regime, adx):
        self.volatility = volatility
        self.regime = regime
        self.adx = adx


def series(value, df):
    return pd.Series(value, index=df.index)

def nz(series):
    return series.fillna(method='backfill')

def normalizeDeriv(src, quadraticMeanLength):
    deriv = src - src[2]
    quadraticMean = math.sqrt(np.nan_to_num(np.sum(np.power(deriv, 2), quadraticMeanLength) / quadraticMeanLength))
    nDeriv = deriv / quadraticMean
    return nDeriv

def normalize(src, minimum, maximum):
    _historicMin =  10e10
    _historicMax = -10e10
    _historicMin = min(nz(src, _historicMin), _historicMin)
    _historicMax = max(nz(src, _historicMax), _historicMax)
    return minimum + (maximum - minimum) * (src - _historicMin) / max(_historicMax - _historicMin, 10e-10)

def rescale(src, oldMin, oldMax, newMin, newMax):
    return newMin + (newMax - newMin) * (src - oldMin) / max(oldMax - oldMin, 10e-10)

def n_rsi(src, n1, n2):
    return rescale(
        ta.EMA(
            ta.RSI(src, n1), n2
        ), 0, 100, 0, 1
    )

def n_cci(src, n1, n2):
    return normalize(ta.EMA(ta.cci(src, n1), n2), 0, 1)

def n_wt(src, n1=10, n2=11):
    ema1 = ta.EMA(src, n1)
    ema2 = ta.EMA(np.abs(src - ema1), n1)
    ci = (src - ema1) / (0.015 * ema2)
    wt1 = ta.EMA(ci, n2)  # tci
    wt2 = ta.SMA(wt1, 4)
    return normalize(wt1 - wt2, 0, 1)

def n_adx(highSrc, lowSrc, closeSrc, n1):
    th = 20
    tr = np.maximum(np.maximum(highSrc - lowSrc, np.abs(highSrc - np.roll(closeSrc, 1))), np.abs(lowSrc - np.roll(closeSrc, 1)))
    directionalMovementPlus = np.where(highSrc - np.roll(highSrc, 1) > np.roll(lowSrc, 1) - lowSrc, np.maximum(highSrc - np.roll(highSrc, 1), 0), 0)
    negMovement = np.where(np.roll(lowSrc, 1) - lowSrc > highSrc - np.roll(highSrc, 1), np.maximum(np.roll(lowSrc, 1) - lowSrc, 0), 0)
    trSmooth = pd.Series(tr).rolling(window=n1).mean().values
    smoothDirectionalMovementPlus = pd.Series(directionalMovementPlus).rolling(window=n1).mean().values
    smoothnegMovement = pd.Series(negMovement).rolling(window=n1).mean().values
    diPositive = smoothDirectionalMovementPlus / trSmooth * 100
    diNegative = smoothnegMovement / trSmooth * 100
    dx = np.abs(diPositive - diNegative) / (diPositive + diNegative) * 100 
    adx = RMA(dx, n1)
    return rescale(adx, 0, 100, 0, 1)

def filter_volatility(df, minLength, maxLength, useVolatilityFilter):
    recentAtr = ta.ATR(df, minLength)
    historicalAtr = ta.ATR(df, maxLength)
    return recentAtr > historicalAtr if useVolatilityFilter else True

def rational_quadratic(src, lookback, relative_weight, start_at_bar=0):
    current_weight = 0.0
    cumulative_weight = 0.0
    size = len(src)
    for i in range(start_at_bar, size):
        y = src[i]
        w = np.power(1 + (np.power(i, 2) / ((np.power(lookback, 2) * 2 * relative_weight))), -relative_weight)
        current_weight += y * w
        cumulative_weight += w
    yhat = current_weight / cumulative_weight
    return yhat

def gaussian(src, lookback, start_at_bar=0):
    current_weight = 0.0
    cumulative_weight = 0.0
    size = len(src)
    for i in range(start_at_bar, size):
        y = src[i]
        w = np.exp(-np.power(i, 2) / (2 * np.power(lookback, 2)))
        current_weight += y * w
        cumulative_weight += w
    yhat = current_weight / cumulative_weight
    return yhat        

def series_from(feature_string, _close, _high, _low, _hlc3, f_paramA, f_paramB):
    if feature_string == "RSI":
        return n_rsi(_close, f_paramA, f_paramB)
    elif feature_string == "WT":
        return n_wt(_hlc3, f_paramA, f_paramB)
    elif feature_string == "CCI":
        return n_cci(_close, f_paramA, f_paramB)
    elif feature_string == "ADX":
        return n_adx(_high, _low, _close, f_paramA)

def get_lorentzian_distance(i, featureCount, featureSeries, featureArrays):
    if featureCount == 5:
        return (
            math.log(1 + abs(featureSeries.f1 - featureArrays.f1[i])) +
            math.log(1 + abs(featureSeries.f2 - featureArrays.f2[i])) +
            math.log(1 + abs(featureSeries.f3 - featureArrays.f3[i])) +
            math.log(1 + abs(featureSeries.f4 - featureArrays.f4[i])) +
            math.log(1 + abs(featureSeries.f5 - featureArrays.f5[i]))
        )
    elif featureCount == 4:
        return (math.log(1 + abs(featureSeries.f1 - featureArrays.f1[i])) +
               math.log(1 + abs(featureSeries.f2 - featureArrays.f2[i])) +
               math.log(1 + abs(featureSeries.f3 - featureArrays.f3[i])) +
               math.log(1 + abs(featureSeries.f4 - featureArrays.f4[i])))
    elif featureCount == 3:
        return (math.log(1 + abs(featureSeries.f1 - featureArrays.f1[i])) +
               math.log(1 + abs(featureSeries.f2 - featureArrays.f2[i])) +
               math.log(1 + abs(featureSeries.f3 - featureArrays.f3[i])))
    elif featureCount == 2:
        return (
            math.log(1 + abs(featureSeries.f1 - featureArrays.f1[i])) +
            math.log(1 + abs(featureSeries.f2 - featureArrays.f2[i]))
        )


showTradeStats = True
useWorstCase = False

filterSettings = FilterSettings(
    useVolatilityFilter = True,
    useRegimeFilter = True,
    useAdxFilter = False,
    regimeThreshold = -0.1,
    adxThreshold = 20
)



f1_string = "RSI"
f1_paramA = 14
f1_paramB = 2
f2_string = "WT"
f2_paramA = 10
f2_paramB = 11
f3_string = "CCI"
f3_paramA = 20
f3_paramB = 1
f4_string = "ADX"
f4_paramA = 20
f4_paramB = 2
f5_string = "RSI"
f5_paramA = 9
f5_paramB = 1



# // FeatureSeries Object: Calculated Feature Series based on Feature Variables
# featureSeries = 
#  FeatureSeries.new(
#    series_from(f1_string, close, high, low, hlc3, f1_paramA, f1_paramB), // f1
#    series_from(f2_string, close, high, low, hlc3, f2_paramA, f2_paramB), // f2 
#    series_from(f3_string, close, high, low, hlc3, f3_paramA, f3_paramB), // f3
#    series_from(f4_string, close, high, low, hlc3, f4_paramA, f4_paramB), // f4
#    series_from(f5_string, close, high, low, hlc3, f5_paramA, f5_paramB)  // f5
#  )

def RMA(src, period):
    return ta.EMA(src, period)

def indicator(df):
    close = df['close']
    high = df['high']
    low = df['low']
    open = df['open']
    hlc3 = (high + low + close) / 3
    ohlc4 = (open + high + low + close) / 4

    settings = Settings(
        source = close,
        neighborsCount = 8,
        maxBarsBack = 2000,
        featureCount = 5,
        colorCompression = 1,
        showExits = False,
        useDynamicExits = False
    )

    def regime_filter(src, threshold, useRegimeFilter):
        # Calculate the slope of the curve.
        value1 = series(0.0, src)
        value2 = series(0.0, src)
        klmf = series(0.0, src)
        value1 = 0.2 * (src - src.shift(1)) + 0.8 * value1.shift(1)
        value2 = 0.1 * (high - low) + 0.8 * value2.shift(1)
        omega = abs(value1 / value2)
        alpha = (-omega.pow(2) + np.sqrt(omega.pow(4) + 16 * omega.pow(2))) / 8 
        klmf = alpha * src + (1 - alpha) * klmf.shift(1)
        absCurveSlope = abs(klmf - klmf.shift(1))
        exponentialAverageAbsCurveSlope = 1.0 * ta.EMA(absCurveSlope, 200)
        normalized_slope_decline = (absCurveSlope - exponentialAverageAbsCurveSlope) / exponentialAverageAbsCurveSlope
        # Calculate the slope of the curve.
        return normalized_slope_decline >= threshold if useRegimeFilter else True

    def filter_adx(src, length, adxThreshold, useAdxFilter):
        tr = pd.DataFrame({
            'high': high,
            'low': low,
            'src': src.shift(1)
        }).apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['src']), abs(x['low'] - x['src'])), axis=1)
        directionalMovementPlus = np.where(high - high.shift(1) > low.shift(1) - low, 
                                        high - high.shift(1), 0)
        negMovement = np.where(low.shift(1) - low > high - high.shift(1),
                            low.shift(1) - low, 0)
        trSmooth = pd.Series(tr).rolling(length).mean()
        smoothDirectionalMovementPlus = pd.Series(directionalMovementPlus).rolling(length).mean()
        smoothnegMovement = pd.Series(negMovement).rolling(length).mean()
        diPositive = smoothDirectionalMovementPlus / trSmooth * 100
        diNegative = smoothnegMovement / trSmooth * 100
        dx = abs(diPositive - diNegative) / (diPositive + diNegative) * 100
        adx = RMA(dx, length)
        return adx > adxThreshold if useAdxFilter else True

    filter = Filter(
        volatility = filter_volatility(df, 1, 10, filterSettings.useVolatilityFilter),
        regime = regime_filter(ohlc4, filterSettings.regimeThreshold, filterSettings.useRegimeFilter),
        adx = filter_adx(settings.source, 14, filterSettings.adxThreshold, filterSettings.useAdxFilter)
    )

    featureSeries = FeatureSeries(
        series_from(f1_string, close, high, low, hlc3, f1_paramA, f1_paramB), # f1
        series_from(f2_string, close, high, low, hlc3, f2_paramA, f2_paramB), # f2
        series_from(f3_string, close, high, low, hlc3, f3_paramA, f3_paramB), # f3
        series_from(f4_string, close, high, low, hlc3, f4_paramA, f4_paramB), # f4
        series_from(f5_string, close, high, low, hlc3, f5_paramA, f5_paramB)  # f5
    )
    featureArrays = FeatureArrays(
        featureSeries.f1, # f1
        featureSeries.f2, # f2
        featureSeries.f3, # f3
        featureSeries.f4, # f4
        featureSeries.f5  # f5
    )

    direction = Label(
        long=1,
        short=-1,
        neutral=0
    )

    maxBarsBackIndex = last_bar_index - settings.maxBarsBack if last_bar_index >= settings.maxBarsBack else 0
    src = settings.source
    y_train_series = direction.short if src[4] < src[0] else direction.long if src[4] > src[0] else direction.neutral
    y_train_array = []

    predictions = []
    prediction = 0.
    signal = direction.neutral
    distances = []

    y_train_array.append(y_train_series)

    lastDistance = -1.0
    size = min(settings.maxBarsBack-1, len(y_train_array)-1)
    sizeLoop = min(settings.maxBarsBack-1, size)

    if bar_index >= maxBarsBackIndex:
        for i in range(0, sizeLoop):
            d = get_lorentzian_distance(i, settings.featureCount, featureSeries, featureArrays)
            if d >= lastDistance and i%4:
                lastDistance = d
                distances.append(d)
                predictions.append(round(y_train_array[i]))
                if len(predictions) > settings.neighborsCount:
                    lastDistance = distances[round(settings.neighborsCount*3/4)]
                    distances.pop(0)
                    predictions.pop(0)
        prediction = sum(predictions)

    filter_all = direction.long if prediction > 0 and filter_all else direction.short if prediction < 0 and filter_all else signal[1]

    isDifferentSignalType = ta.change(signal)
    isBuySignal = signal == direction.long
    isSellSignal = signal == direction.short
    isNewBuySignal = isBuySignal and isDifferentSignalType
    isNewSellSignal = isSellSignal and isDifferentSignalType

    startLongTrade = isNewBuySignal
    startShortTrade = isNewSellSignal

    # plotshape(low if startLongTrade else na, 'Buy', shape.labelup, location.belowbar, color=color_green(prediction), size=size.small, offset=0)`