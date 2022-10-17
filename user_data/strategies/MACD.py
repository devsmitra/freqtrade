# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, stoploss_from_absolute
from pandas import DataFrame
from freqtrade.persistence import Trade
from datetime import datetime
from typing import Optional

# --------------------------------
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import indicators as indicators


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
        heikinashi = qtpylib.heikinashi(df)
        df['ha_close'] = heikinashi['close']
        df['ha_open'] = heikinashi['open']
        df['ha_high'] = heikinashi['high']
        df['ha_low'] = heikinashi['low']

        df['atr'] = ta.ATR(df, timeperiod=14)
        df['zlsma'] = indicators.zlsma(df, period=50, offset=0, column='ha_close')
        df['chandelier_exit'] = indicators.chandelier_exit(
            df, timeperiod=14, multiplier=1.85, column='ha_close')

        vol = indicators.volatility_osc(df)
        df['upper'] = vol['upper']
        df['lower'] = vol['lower']
        df['spike'] = vol['spike']
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # enter_long = (
        #     ((df['ha_close'] > df['zlsma']) & qtpylib.crossed_above(df['spike'], df['upper'])) |
        #     (qtpylib.crossed_above(df['ha_close'], df['zlsma']) & (df['spike'] > df['upper']))
        # )
        enter_long = (
            (df['ha_close'] > df['zlsma']) &
            qtpylib.crossed_above(df['chandelier_exit'], 0)
        )
        df.loc[enter_long, 'enter_long'] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            # (
            #     (df['spike'] < 0) & qtpylib.crossed_below(df['ha_close'], df['zlsma']) |
            #     qtpylib.crossed_below(df['spike'], 0) & (df['ha_close'] < df['zlsma'])
            # ),
            (
                (df['chandelier_exit'] < 1) & qtpylib.crossed_below(df['ha_close'], df['zlsma']) |
                qtpylib.crossed_below(df['chandelier_exit'], 1) & (df['ha_close'] < df['zlsma'])
            ),
            'exit_long'] = 1
        return df

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = df.iloc[-1].squeeze()
        atr = candle['atr']

        def get_stoploss(multiplier):
            return stoploss_from_absolute(current_rate - (
               atr * multiplier), current_rate, is_short=trade.is_short
            ) * -1

        if (current_profit > .1):
            if (candle['chandelier_exit'] != 1):
                return get_stoploss(1.1)
            return get_stoploss(2.1)
        if (current_profit > .05):
            return get_stoploss(3.1)
        return get_stoploss(4.1)

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
