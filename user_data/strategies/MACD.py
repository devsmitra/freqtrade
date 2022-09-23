# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, stoploss_from_absolute
from pandas import DataFrame
from freqtrade.persistence import Trade
from datetime import datetime
from typing import Optional

# --------------------------------
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta


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

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        if self.wallets is None:
            return proposed_stake
        return self.wallets.get_total_stake_amount() * .06

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(df)
        df['macd'] = macd['macd']
        df['macdsignal'] = macd['macdsignal']
        df['rsi'] = ta.RSI(df)
        df['atr'] = ta.ATR(df)
        df['adx'] = ta.ADX(df)
        df['trend'] = ta.SMA(df, timeperiod=200)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                (df['adx'] > 20) & (
                    (
                        qtpylib.crossed_above(df['macd'], df['macdsignal']) &
                        (df['rsi'] > 50)
                    ) | (
                        (df['macd'] > df['macdsignal']) &
                        qtpylib.crossed_above(df['rsi'], 50)
                    )
                )
            ),
            'enter_long'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                (df['macd'] < df['macdsignal']) &
                (df['rsi'] < 50)
            ),
            'exit_long'] = 1
        return df

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        profit = trade.calc_profit_ratio(rate)
        if (((exit_reason == 'force_exit') | (exit_reason == 'exit_signal')) and (profit < 0)):
            return False
        return True

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = df.iloc[-1].squeeze()
        if current_profit > 0:
            atr = candle['atr']
            if ((candle['open'] + (atr * 6.1)) < current_rate):
                return 'Profit Booked'

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = df.iloc[-1].squeeze()

        def get_stoploss(atr):
            return stoploss_from_absolute(current_rate - (
                candle['atr'] * atr), current_rate, is_short=trade.is_short
            ) * -1

        if current_profit > .1:
            return get_stoploss(1.1)
        if current_profit > .05:
            return get_stoploss(2.1)
        return get_stoploss(4.1)
