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
    use_custom_stoploss = True

    # Optimal timeframe for the strategy
    timeframe = '1h'

    # Cache
    # cache = {}

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        if self.wallets is None:
            return proposed_stake
        return self.wallets.get_total_stake_amount() * .06

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['atr'] = ta.ATR(df)
        df['adx'] = ta.ADX(df)
        mad.madrid(df)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        con = (
            (df.shift()['val5'] <= 00) | (df.shift()['val10'] <= 0) | (df.shift()['val15'] <= 0) |
            (df.shift()['val20'] <= 0) | (df.shift()['val25'] <= 0) | (df.shift()['val30'] <= 0) |
            (df.shift()['val35'] <= 0) | (df.shift()['val40'] <= 0) | (df.shift()['val45'] <= 0) |
            (df.shift()['val50'] <= 0) | (df.shift()['val55'] <= 0) | (df.shift()['val60'] <= 0) |
            (df.shift()['val65'] <= 0) | (df.shift()['val70'] <= 0) | (df.shift()['val75'] <= 0) |
            (df.shift()['val80'] <= 0) | (df.shift()['val85'] <= 0) | (df.shift()['val90'] <= 0)
        ) | qtpylib.crossed_above(df['adx'], 20)

        enter_long = (
            con &
            (df['val5'] > 0) & (df['val10'] > 0) & (df['val15'] > 0) &
            (df['val20'] > 0) & (df['val25'] > 0) & (df['val30'] > 0) &
            (df['val35'] > 0) & (df['val40'] > 0) & (df['val45'] > 0) &
            (df['val50'] > 0) & (df['val55'] > 0) & (df['val60'] > 0) &
            (df['val65'] > 0) & (df['val70'] > 0) & (df['val75'] > 0) &
            (df['val80'] > 0) & (df['val85'] > 0) & (df['val90'] > 0)
        )

        # self.cache[metadata['pair']] = df['low'].rolling(5).min()
        df.loc[enter_long, 'enter_long'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                qtpylib.crossed_below(df['adx'], 25) |
                (
                    (df['val5'] <= 0) & (df['val10'] <= 0) & (df['val15'] <= 0) &
                    (df['val20'] <= 0) & (df['val25'] <= 0) & (df['val30'] <= 0) &
                    (df['val35'] <= 0) & (df['val40'] <= 0) & (df['val45'] <= 0) &
                    (df['val50'] <= 0) & (df['val55'] <= 0) & (df['val60'] <= 0) &
                    (df['val65'] <= 0) & (df['val70'] <= 0) & (df['val75'] <= 0) &
                    (df['val80'] <= 0) & (df['val85'] <= 0) & (df['val90'] <= 0)
                )
            ),
            'exit_long'
        ] = 1
        return df

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        if (exit_reason == 'force_exit'):
            return False
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

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime',
                    current_rate: float, current_profit: float, **kwargs):

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = df.iloc[-1].squeeze()

        if current_profit > 0:
            atr = candle['atr']
            if ((candle['open'] + (atr * 6.1)) < current_rate):
                return 'Profit Booked'
