import pandas as pd
import numpy as np
import mplfinance as mpf
from mplfinance.plotting import make_addplot
from datetime import datetime
from moomoo import OpenQuoteContext

# parameters
step = 0.01
limit = 0.2

def SAR(df_original, s, l):
    df = df_original.copy()

    signals = np.zeros(len(df))
    sar = np.zeros(len(df))
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    trend = 1  # 1 for uptrend, -1 for downtrend
    accel = s
    ep = low[0]  # Extreme Point
    sar[0] = high[0]  # Initial SAR

    for i in range(1, len(df)):
        if trend == 1:  # Uptrend
            sar[i] = sar[i-1] + accel * (ep - sar[i-1])
            # Constraint
            previous_low = low[i - 1]
            second_previous_low = low[i - 1]
            if i >= 2:
                second_previous_low = low[i - 2]
            sar[i] = max(sar[i], previous_low, second_previous_low)
            
            # Reversal check
            if close[i] < sar[i]:
                signals[i] = -1 # Sell signal
                trend = -1
                sar[i] = ep  # Set SAR to EP (highest high of uptrend)
                ep = low[i]  # New EP for downtrend
                accel = s
            else:
                # Update EP and accel
                if high[i] > ep:
                    ep = high[i]
                    accel = min(accel + s, l)
        else:  # Downtrend
            sar[i] = sar[i-1] - accel * (ep - sar[i-1])
            # Constraint

            previous_high = high[i - 1]
            second_previous_high = high[i - 1]
            if i >= 2:
                second_previous_high = high[i - 2]
            sar[i] = min(sar[i], previous_high, second_previous_high)
            # Reversal check
            if close[i] >= sar[i]:
                signals[i] = 1  # buy signal
                trend = 1
                sar[i] = ep  # Set SAR to EP (lowest low of downtrend)
                ep = high[i]  # New EP for uptrend
                accel = s
            else:
                # Update EP and accel
                if low[i] < ep:
                    ep = low[i]
                    accel = min(accel + s, l)
    df_original['signals'] = signals
    return 0

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
            ktype='K_60M',
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

    SAR(all_data, step, limit)

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