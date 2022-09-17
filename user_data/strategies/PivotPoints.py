# --- Do not remove these libs ---
from datetime import datetime
from typing import Any, Optional
from freqtrade.strategy import IStrategy, informative, stoploss_from_absolute
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from technical.pivots_points import pivots_points
import talib
# --------------------------------


class PivotPoints(IStrategy):
    cache: Any = {}

    INTERFACE_VERSION: int = 3
    process_only_new_candles: bool = False
    # Optimal timeframe for the strategy
    timeframe = '1h'

    minimal_roi = {
        "0": 1
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.05
    use_custom_stoploss = True

    @informative('1w')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        data = pivots_points(dataframe, timeperiod=1)
        dataframe['pivot'] = data['pivot']
        for i in range(1, 3):
            dataframe["s" + str(i)] = data["s" + str(i)]
            dataframe["r" + str(i)] = data["r" + str(i)]
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        sl = 4.1
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = dataframe.iloc[-1].squeeze()

        def get_stoploss(atr):
            return stoploss_from_absolute(current_rate - (
                candle['atr'] * atr), current_rate, is_short=trade.is_short
            )

        return get_stoploss(sl) * -1

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        if self.wallets is None:
            return proposed_stake
        return self.wallets.get_total_stake_amount() * .06

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        # dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['pattern'] = self.find_pattern(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        def touched_pivot(dataframe, key):
            for i in range(0, 3):
                low = dataframe.shift(i)['low']
                close = dataframe.shift(i)['close']
                return (low < dataframe[key]) & (close > dataframe[key])

        crossed = touched_pivot(dataframe, 'pivot_1w')
        for i in range(1, 3):
            crossed = crossed | touched_pivot(dataframe, "s" + str(i) + '_1w')
            crossed = crossed | touched_pivot(dataframe, "r" + str(i) + '_1w')

        dataframe.loc[
            (
                crossed &
                dataframe['pattern'] &
                (dataframe['rsi'] > 50)
            ),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        crossed = False
        for i in range(1, 3):
            crossed = crossed | qtpylib.crossed_below(
                dataframe['close'], dataframe["r" + str(i) + '_1w']
            )

        dataframe.loc[crossed, 'exit_long'] = 1
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        profit = trade.calc_profit_ratio(rate)
        if (((exit_reason == 'force_exit') | (exit_reason == 'exit_signal')) and (profit < 0)):
            return False
        return True

    def find_pattern(self, df: DataFrame):
        data = {}
        data['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        data['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        data['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        data['dragonfly_doji'] = talib.CDLDRAGONFLYDOJI(
            df['open'], df['high'], df['low'], df['close']
        )
        # Three white soldiers
        data['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(
            df['open'], df['high'], df['low'], df['close']
        )
        return (
            (data['morning_star'] == 100) |
            (data['engulfing'] == 100) |
            (data['hammer'] == 100) |
            (data['dragonfly_doji'] == 100) |
            (data['three_white_soldiers'] == 100)
        )
