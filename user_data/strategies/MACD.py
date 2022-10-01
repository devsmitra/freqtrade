# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, stoploss_from_absolute
from pandas import DataFrame
from freqtrade.persistence import Trade
from datetime import datetime
from typing import Optional

# --------------------------------
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import support as sup


class MACD(IStrategy):
    INTERFACE_VERSION: int = 3

    minimal_roi = {
        "0":  1
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.1
    use_custom_stoploss = True

    # Optimal timeframe for the strategy
    timeframe = '15m'

    # cache
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

        time = df.iloc[-1]['date']
        key = metadata['pair']
        if key in self.cache and self.cache[key]['Time'] == time:
            df['Trend'] = self.cache[key]['Trend']
        else:
            sup.identify_df_trends(df, 'close')
            self.cache[key] = {
                'Trend': df['Trend'],
                'Time': time
            }
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            qtpylib.crossed_above(df['Trend'], 0),
            'enter_long'
        ] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (df['Trend'] == -1),
            'exit_long'
        ] = 1
        return df

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        profit = trade.calc_profit_ratio(rate)
        if (((exit_reason == 'force_exit') | (exit_reason == 'exit_signal')) and (profit < 0)):
            return False
        return True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = df.iloc[-1].squeeze()

        alt = candle['atr']

        def get_stoploss(multiplier):
            return stoploss_from_absolute(current_rate - (
               alt * multiplier), current_rate, is_short=trade.is_short
            ) * -1

        if ((candle['open'] + (alt * 6.1)) < current_rate):
            return get_stoploss(1)
        if current_profit > .1:
            return get_stoploss(1.1)
        if current_profit > .05:
            return get_stoploss(2.1)
        return get_stoploss(4.1)
