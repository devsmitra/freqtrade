# --- Do not remove these libs ---
from datetime import datetime
from typing import Any, Optional
from freqtrade.strategy import IStrategy, informative, stoploss_from_absolute
from pandas import DataFrame
import talib.abstract as ta
import indicators as indicators
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade

import BBB as bbb


# --------------------------------


class NadarayaWatson(IStrategy):
    INTERFACE_VERSION: int = 3

    minimal_roi = {
        "0":  1
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.1
    use_custom_stoploss = True

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
        # bbb.indicator(df)
        # data1 = indicators.Nadaraya_Watson_Envelope(df)
        # df['up'] = data1['up']
        # df['dn'] = data1['dn']

        data = indicators.Nadaraya_Watson(df)
        df['yhat'] = data['yhat']
        df['yhat2'] = data['yhat2']

        df['ema'] = ta.EMA(df, timeperiod=200)
        data = ta.MACD(df)
        df['macd'] = data['macd']
        df['macdsignal'] = data['macdsignal']

        df['atr'] = ta.ATR(df, timeperiod=14)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        # if pair in self.custom_info:
        #     del self.custom_info[pair]
        df.loc[
            ( 
                (df['close'] > df['ema']) &
                (df['macd'] < 0) &
                ((
                    qtpylib.crossed_above(df['yhat2'], df['yhat']) &
                   (df['macd'] >= df['macdsignal'])
                ) | (
                    (df['yhat2'] >= df['yhat']) &
                    qtpylib.crossed_above(df['macd'], df['macdsignal'])
                ))
            ),
            'enter_long'
        ] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                qtpylib.crossed_below(df['yhat2'], df['yhat']) &
                False
            ),
            'exit_long'
        ] = 1
        return df

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = dataframe.iloc[-1].squeeze()

        if pair not in self.custom_info:
            self.custom_info[pair] = {
                'last_candle': candle,
            }

        candle = self.custom_info[pair]['last_candle']

        def get_stoploss(atr):
            return stoploss_from_absolute(trade.open_rate - (
                candle['atr'] * atr), trade.open_rate, is_short=trade.is_short
            ) * -1

        sl = 2.1
        if (self.profit_target(pair, trade, current_rate, dataframe)):
            if ((candle['macd'] < candle['macdsignal']) | (candle['yhat2'] < candle['yhat'])):
                return get_stoploss(.001)

        return get_stoploss(sl)

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        profit = trade.calc_profit_ratio(rate)
        if (((exit_reason == 'force_exit') | (exit_reason == 'exit_signal')) and (profit < 0.005)):
            return False

        if pair in self.custom_info:
            del self.custom_info[pair]
        return True

    def profit_target(self, pair, trade, current_rate, df):
        last_candle = df.iloc[-1].squeeze()
        if pair not in self.custom_info:
            self.custom_info[pair] = {
                'last_candle': last_candle,
            }
        last_candle = self.custom_info[pair]['last_candle']
        pt = trade.open_rate + (last_candle['atr'] * 3)
        return (pt < current_rate)

    # def custom_exit(self, pair: str, trade: Trade, current_time: 'datetime',
    #                 current_rate: float, current_profit: float, **kwargs):
    #     df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     if (self.profit_target(pair, trade, current_rate, df)):
    #         last_candle = df.iloc[-1].squeeze()
    #         if ((last_candle['macd'] < last_candle['macdsignal']) | (last_candle['yhat2'] < last_candle['yhat'])):
    #             return 'Profit Booked'