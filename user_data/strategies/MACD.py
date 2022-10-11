# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, stoploss_from_absolute
from pandas import DataFrame
from freqtrade.persistence import Trade
from datetime import datetime
from typing import Optional

# --------------------------------
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import madrid as mad


class MACD(IStrategy):
    INTERFACE_VERSION: int = 3

    minimal_roi = {
        "0":  1
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.1
    use_custom_stoploss = False

    # Optimal timeframe for the strategy
    timeframe = '1h'

    # Cache
    cache: dict = {}

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        if self.wallets is None:
            return proposed_stake
        return self.wallets.get_total_stake_amount() * .06

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['atr'] = ta.ATR(df)
        df['rsi'] = ta.RSI(df)
        mad.madrid(df)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        condition = (
            ((df['ma5'] > df['ma100']) & qtpylib.crossed_above(df['ma5'], df['ma10'])) |
            (qtpylib.crossed_above(df['ma5'], df['ma100']) & (df['ma5'] > df['ma10']))
        ) & (df['rsi'] > 50)

        enter_long = (
            condition &
            (df['val5'] > 0) & (df['val10'] > 0) & (df['val15'] > 0) &
            (df['val20'] > 0) & (df['val25'] > 0) & (df['val30'] > 0) &
            (df['val35'] > 0) & (df['val40'] > 0) & (df['val45'] > 0) &
            (df['val50'] > 0) & (df['val55'] > 0) & (df['val60'] > 0) &
            (df['val65'] > 0) & (df['val70'] > 0) & (df['val75'] > 0) &
            (df['val80'] > 0) & (df['val85'] > 0) & (df['val90'] > 0)
        )

        df.loc[enter_long, 'enter_long'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            False &
            qtpylib.crossed_below(df['ma5'], df['ma10']),
            'exit_long'
        ] = 1
        return df

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        profit = trade.calc_profit_ratio(rate)
        if (((exit_reason == 'force_exit') | (exit_reason == 'exit_signal')) and (profit < 0)):
            return False
        if pair in self.cache:
            del self.cache[pair]
        return True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = df.iloc[-1].squeeze()
        atr = candle['atr']

        def get_stoploss(multiplier):
            return stoploss_from_absolute(current_rate - (
               atr * multiplier), current_rate, is_short=trade.is_short
            ) * -1

        if current_profit > .1:
            return get_stoploss(1.1)
        return get_stoploss(4.1)

    def custom_exit(self, pair: str, trade: Trade, current_time: 'datetime',
                    current_rate: float, current_profit: float, **kwargs):

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = df.iloc[-1].squeeze()

        if pair not in self.cache:
            self.cache[pair] = df['low'].rolling(3).min().iloc[-1]

        if pair in self.cache:
            diff = (trade.open_rate - self.cache[pair]) * 1.1
            if (((trade.open_rate + diff * 1.5) < current_rate) & (candle['open'] < candle['ma5'])):
                del self.cache[pair]
                return 'Profit Booked'
            if (trade.open_rate - diff) > current_rate:
                del self.cache[pair]
                return 'Stop Loss Hit'

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 6
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 6,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": True
            },
        ]
