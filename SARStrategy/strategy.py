import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime
from moomoo import OpenQuoteContext

# Parameters
STEP = 0.05
LIMIT = 0.1
THRESHOLD = 0.1
SMA_PERIOD = 14
SMA_THRESHOLD = 7


def SMA(data, period, window=7, distance=1):
    data['short_sma'] = data['close'].rolling(window=period).mean()
    data['long_sma'] = data['close'].rolling(window=period * 2).mean()
    data['delta'] = data['short_sma'] - data['long_sma']


def SAR(data, step=0.02, limit=0.2, threshold=0.2):
    trend = 1  # 1 = uptrend, -1 = downtrend
    af = step
    ep = data['high'].iloc[0]

    sar = [np.nan] * len(data)
    sar[1] = data['low'].iloc[0]

    signals = [0] * len(data)
    trend_list = [np.nan] * len(data)
    trend_list[1] = trend

    for i in range(2, len(data)):
        prev_sar = sar[i - 1]

        if trend == 1:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], data['low'].iloc[i - 1], data['low'].iloc[i - 2])
            if sar[i] + threshold >= data['open'].iloc[i]:
                trend = -1
                sar[i] = ep
                ep = data['open'].iloc[i]
                af = step
                signals[i] = -1
            else:
                if data['open'].iloc[i] > ep:
                    ep = data['high'].iloc[i]
                    af = min(af + step, limit)
        else:
            sar[i] = prev_sar - af * (prev_sar - ep)
            sar[i] = max(sar[i], data['high'].iloc[i - 1], data['high'].iloc[i - 2])
            if sar[i] - threshold <= data['open'].iloc[i]:
                trend = 1
                sar[i] = ep
                ep = data['open'].iloc[i]
                af = step
                signals[i] = 1
            else:
                if data['open'].iloc[i] < ep:
                    ep = data['open'].iloc[i]
                    af = min(af + step, limit)

        trend_list[i] = trend

    data['signals'] = signals
    data['sar'] = sar
    data['trend'] = trend_list
    return data


def main():
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    start_date = '2025-04-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    symbol = 'HK.01810'

    all_data = pd.DataFrame()
    page_req_key = None

    while True:
        ret, data, page_req_key = quote_ctx.request_history_kline(
            symbol,
            start=start_date,
            end=end_date,
            ktype='K_30M',
            autype='qfq',
            max_count=1000,
            page_req_key=page_req_key
        )

        if ret != 0:
            print(f"Error fetching data: {data}")
            return

        all_data = pd.concat([all_data, data], ignore_index=True)

        if page_req_key is None:
            break

    quote_ctx.close()

    all_data['time_key'] = pd.to_datetime(all_data['time_key'])
    all_data.set_index('time_key', inplace=True)

    all_data = SAR(all_data, STEP, LIMIT, THRESHOLD)
    SMA(all_data, SMA_PERIOD, 7, SMA_THRESHOLD)

    # Clean invalid rows before plotting
    all_data = all_data[~all_data.isin(["", None]).any(axis=1)]
    all_data.dropna(subset=['sar', 'short_sma', 'long_sma'], inplace=True)

    # Fix 0.0 SAR placeholder
    all_data['sar'] = all_data['sar'].replace(0.0, np.nan)

    # Buy/sell signals
    buy_signals = all_data['signals'].copy()
    sell_signals = all_data['signals'].copy()
    buy_signals[buy_signals != 1] = np.nan
    sell_signals[sell_signals != -1] = np.nan
    buy_plot = buy_signals * all_data['low'] * 0.9995
    sell_plot = sell_signals * all_data['high'] * -1.0005

    # Separate SAR dots by trend
    uptrend_sar = all_data.apply(lambda row: row['sar'] if row['trend'] == 1 else np.nan, axis=1)
    downtrend_sar = all_data.apply(lambda row: row['sar'] if row['trend'] == -1 else np.nan, axis=1)

    # Save for inspection
    all_data.to_csv("sar_sma.csv", index=False)

    apdict = [
        mpf.make_addplot(buy_plot, type='scatter', markersize=100, marker='^', color='green'),
        mpf.make_addplot(sell_plot, type='scatter', markersize=100, marker='v', color='red'),
        mpf.make_addplot(uptrend_sar, type='scatter', markersize=20, marker='.', color='green'),
        mpf.make_addplot(downtrend_sar, type='scatter', markersize=20, marker='.', color='red'),
        mpf.make_addplot(all_data['short_sma'], color='blue', width=1.2, label='Short SMA'),
        mpf.make_addplot(all_data['long_sma'], color='orange', width=1.2, label='Long SMA'),
    ]

    mpf.plot(all_data,
             type='candle',
             style='yahoo',
             title='SAR + SMA Signals',
             ylabel='Price',
             figsize=(10, 5),
             addplot=apdict)


if __name__ == "__main__":
    main()
