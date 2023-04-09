# # --- Do not remove these libs ---
# from datetime import datetime
# from typing import Any, Optional
# from freqtrade.strategy import IStrategy, informative, stoploss_from_absolute
# from pandas import DataFrame
# import talib.abstract as ta
# import indicators as indicators
# import freqtrade.vendor.qtpylib.indicators as qtpylib
# from freqtrade.persistence import Trade
# # --------------------------------

# class NadarayaWatson(IStrategy):
#     INTERFACE_VERSION: int = 3

#     minimal_roi = { "0":  1 }

#     # Optimal stoploss designed for the strategy
#     sl = 4.1
#     stoploss = -0.1
#     use_custom_stoploss = True

#     # Optimal timeframe for the strategy
#     timeframe = '1h'

#     custom_info: dict = {}

#     def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
#                         proposed_stake: float, min_stake: Optional[float], max_stake: float,
#                         leverage: float, entry_tag: Optional[str], side: str,
#                         **kwargs) -> float:
#         if self.wallets is None:
#             return proposed_stake
#         return self.wallets.get_total_stake_amount() * .06

#     def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
#         data = indicators.Nadaraya_Watson(df['close'], loop_back=20)
#         df['yhat_fast'] = data['yhat']
#         data = indicators.Nadaraya_Watson(df['close'], loop_back=32)
#         df['yhat_slow'] = data['yhat']

#         df['ema'] = indicators.smma(df, timeperiod=32)

#         df['atr'] = ta.ATR(df, timeperiod=14)
#         atr = indicators.Nadaraya_Watson(df['atr'], loop_back=60)
#         df['atr_yhat'] = atr['yhat']



#         # dt = indicators.Nadaraya_Watson_Envelope(df)
#         df['up'] = df['yhat_slow'] + (df['atr_yhat'] * 3)
#         df['dn'] = df['yhat_slow'] - (df['atr_yhat'] * 3)
#         return df

#     def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
#         pair = metadata['pair']
#         df.loc[
#             (
#                 # (df['close'] > df['ema']) &
#                 # (df['atr_yhat'] > df['atr_yhat'].shift(1)) &

#                 # (df['yhat_slow'] >  df['yhat_slow'].shift(1)) &
#                 # qtpylib.crossed_above(df['yhat_fast'], df['yhat_slow'])

#                 ((df['close'].diff() / df['close'])  < 0.05) &
#                 (df['yhat_fast'] > df['yhat_fast'].shift(1)) &
#                 qtpylib.crossed_above(df['yhat_slow'], df['yhat_slow'].shift(1))
#             ),
#             'enter_long'
#         ] = 1
#         return df

#     def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
#         df.loc[
#             (qtpylib.crossed_below(df['yhat_slow'], df['yhat_slow'].shift(1)) & False),
#             'exit_long'
#         ] = 1
#         return df

#     def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
#                            rate: float, time_in_force: str, exit_reason: str,
#                            current_time: datetime, **kwargs) -> bool:
#         profit = trade.calc_profit_ratio(rate)
#         if (exit_reason == 'force_exit'):
#             return False
#         if pair in self.custom_info:
#             del self.custom_info[pair]
#         return True

#     def trade_candle(self, pair, trade, current_rate):
#         if pair not in self.custom_info:
#             df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
#             # rolling min
#             # low = df['low'].rolling(24).min().iloc[-1].squeeze()
#             # diff = trade.open_rate - low

#             last_candle = df.iloc[-1].squeeze()
#             diff = last_candle['atr'] * 4.1

#             sl = (diff * 1.1)
#             self.custom_info[pair] = {
#                 'sl': trade.open_rate - sl,
#                 'pt': trade.open_rate + (sl * 1.5),
#                 'diff': diff,
#             }
#         return self.custom_info[pair]

#     def custom_exit(self, pair: str, trade: Trade, current_time: 'datetime',
#                     current_rate: float, current_profit: float, **kwargs):
#         last_candle = self.trade_candle(pair, trade, current_rate)

#         pt = last_candle['pt']
#         if (pt < current_rate):
#             return 'Profit Booked'

#         diff = last_candle['diff']
#         if ((trade.open_rate + (diff * 1.1)) < current_rate):
#             self.custom_info[pair]['sl'] = trade.open_rate * (diff * .1)
        
#         sl = last_candle['sl']
#         if (sl > current_rate):
#             return 'Stop Loss Hit'

#     def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
#                     current_rate: float, current_profit: float, **kwargs) -> float:
#         low = self.trade_candle(pair, trade, current_rate)
#         diff = trade.open_rate - low['sl']
#         sl = diff * 1.1
#         # pt = trade.open_rate + (sl * 1.5)

#         def get_stoploss(sl):
#             return stoploss_from_absolute(
#                 sl, 
#                 current_rate, 
#                 is_short=trade.is_short
#             ) * -1


#         # diff = current_rate - trade.open_rate
#         # diff = diff if diff > 0 else diff * -1
#         # pt = trade.open_rate + (diff * 1.5)

#         pt = trade.open_rate + sl
#         if (pt < current_rate):
#             return get_stoploss(trade.open_rate)
#         return 1