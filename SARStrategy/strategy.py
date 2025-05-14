import pandas as pd
import numpy as np
import mplfinance as mpf
from mplfinance.plotting import make_addplot
from datetime import datetime
from moomoo import OpenQuoteContext

# parameters
step = 0.01
limit = 1


short_ema = 10
long_ema = 20
threshold = 0

def calculate_atr(data, window=14):
    # Calculate True Range (TR)
    previous_close = data['close'].iloc[0]
    
    data['high_low'] = data['high'] - data['low']
    data['high_close'] = abs(data['high'] - previous_close)
    data['low_close'] = abs(data['low'] - previous_close)
    
    # True Range (TR) is the maximum of the 3 values
    data['true_range'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)
    
    # ATR is the moving average of True Range
    data['atr'] = data['true_range'].rolling(window=window).mean()
    
    # Drop intermediate columns
    data.drop(columns=['high_low', 'high_close', 'low_close'], inplace=True)
    
    return data

def SAR(data, step=0.02, limit=0.2, threshold=0.2):
    

    trend = 1  # 1 = uptrend, -1 = downtrend
    af = step
    ep = data['high'].iloc[0]
    
    # Initialize SAR list
    sar = [0.0] * len(data)
    sar[1] = data['low'].iloc[0]
    
    # Initialize columns
    signals = [0] * len(data)

    


    for i in range(2, len(data)):
        prev_sar = sar[i - 1]

        if trend == 1:  # Uptrend
            sar[i] = prev_sar + af * (ep - prev_sar)
            # Enforce SAR cannot be above prior 2 lows
            sar[i] = min(sar[i], data['low'].iloc[i - 1], data['low'].iloc[i - 2])

            # Reversal condition
            if sar[i] + threshold >= data['low'].iloc[i]:
                trend = -1
                sar[i] = ep
                ep = data['low'].iloc[i]
                af = step
                signals[i] = -1
                continue

            # Update EP and AF if a new high is made
            if data['high'].iloc[i] > ep:
                ep = data['high'].iloc[i]
                af = min(af + step, limit)

        else:  # Downtrend
            sar[i] = prev_sar - af * (prev_sar - ep)
            # Enforce SAR cannot be below prior 2 highs
            sar[i] = max(sar[i], data['high'].iloc[i - 1], data['high'].iloc[i - 2])

            # Reversal condition
            if sar[i] - threshold <= data['high'].iloc[i]:
                trend = 1
                sar[i] = ep
                ep = data['high'].iloc[i]
                af = step
                signals[i] = 1
                continue

            # Update EP and AF if a new low is made
            if data['low'].iloc[i] < ep:
                ep = data['low'].iloc[i]
                af = min(af + step, limit)
    data['signals'] = signals
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

    # Prepare data for plot
    all_data['time_key'] = pd.to_datetime(all_data['time_key'])
    all_data.set_index('time_key', inplace=True)

    SAR(all_data, step, limit, threshold)
    # Create buy/sell signal series for plotting
    buy_signals = all_data['signals'].copy()
    sell_signals = all_data['signals'].copy()

    # Keep only signal locations
    buy_signals[buy_signals != 1] = np.nan
    sell_signals[sell_signals != -1] = np.nan

    buy_marker_prices = all_data['low'] * 0.9995  # 0.5% below low for buy arrows
    sell_marker_prices = all_data['high'] * -1.0005  # 0.5% above high for sell arrows

    # Combine signals with marker prices
    buy_plot = buy_signals * buy_marker_prices  # NaN where no buy signal
    sell_plot = sell_signals * sell_marker_prices  # NaN where no sell signal


    apdict = [
    # Buy signals: green upward arrows
    mpf.make_addplot(buy_plot, type='scatter', markersize=100, marker='^', color='green'),
    # Sell signals: red downward arrows
    mpf.make_addplot(sell_plot, type='scatter', markersize=100, marker='v', color='red'),
    ]

            # Plot with mplfinance
    mpf.plot(all_data,
            type='candle',
            style='yahoo',
            title='Signals',
            ylabel='Price',
            figsize=(8,4),
            addplot=apdict)

if __name__ == "__main__":
    main()