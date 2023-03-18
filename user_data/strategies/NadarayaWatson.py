# --- Do not remove these libs ---
from datetime import datetime
from typing import Any, Optional
from freqtrade.strategy import IStrategy, informative, stoploss_from_absolute
from pandas import DataFrame
import talib.abstract as ta
import indicators as indicators
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade

# --------------------------------


class NadarayaWatson(IStrategy):
    INTERFACE_VERSION: int = 3

    minimal_roi = {
        "0":  1
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.1
    use_custom_stoploss = False

    # Optimal timeframe for the strategy
    timeframe = '1h'

    custom_info: dict = {}

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                        proposed_stake: float, min_stake: Optional[float], max_stake: float,
                        leverage: float, entry_tag: Optional[str], side: str,
                        **kwargs) -> float:
        if self.wallets is None:
            return proposed_stake
        return self.wallets.get_total_stake_amount() * .06

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        data = indicators.Nadaraya_Watson(df)
        df['yhat'] = data['yhat']
        df['yhat2'] = data['yhat2']
        # df['up'] = data['up']
        # df['dn'] = data['dn']

        df['rsi'] = ta.RSI(df, timeperiod=14)
        df['ema_rsi'] = ta.EMA(df['rsi'], timeperiod=14)

        df['ema_200'] = ta.EMA(df, timeperiod=50)
        df['atr'] = ta.ATR(df, timeperiod=14)
        df['atr_low'] = df['close'] - (df['atr'] * 2)
        df['atr_high'] = df['close'] + (df['atr'] * 4)
        # df['swing_low'] = df['low'].rolling(20).min()
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        # if pair in self.custom_info:
        #     del self.custom_info[pair]
        df.loc[
            ( 
                # (df['close'] > df['ema_200']) &
                
                (df['rsi'] > df['ema_rsi']) &
                (df['rsi'] > 50) &
                qtpylib.crossed_above(df['yhat2'], df['yhat'])
            ),
            'enter_long'
        ] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            qtpylib.crossed_below(df['yhat2'], df['yhat']) & False,
            'exit_long'
        ] = 1
        return df

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        # if pair not in self.custom_info:
        #     self.custom_info[pair] = {
        #         'last_candle': last_candle,
        #     }

        # last_candle = self.custom_info[pair]['last_candle']
        atr = last_candle['atr_low']

        def get_stoploss(multiplier):
            return stoploss_from_absolute(atr, trade.open_rate, is_short=trade.is_short) * -1

        return get_stoploss(2.1)

    
    def custom_exit(self, pair: str, trade: Trade, current_time: 'datetime',
                    current_rate: float, current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if pair not in self.custom_info:
            self.custom_info[pair] = {
                'last_candle': last_candle,
            }

        last_candle = self.custom_info[pair]['last_candle']

        if (current_rate < last_candle['atr_low']):
            del self.custom_info[pair]
            return 'Stop Loss Hit'

        pt = trade.open_rate + last_candle['atr_high']
        if (pt < current_rate):
            del self.custom_info[pair]
            return 'Profit Booked'