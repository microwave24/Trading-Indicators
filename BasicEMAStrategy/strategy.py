import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from moomoo import OpenQuoteContext

# EMA periods
short_term_ema = 2
long_term_ema = 5

def calculate_ema(df, period):
    return df['close'].ewm(span=period, adjust=False).mean()

def find_signal(df, short_ema_col, long_ema_col):
    df = df[[short_ema_col, long_ema_col, 'time_key', 'close']].dropna().copy()
    df['delta'] = df[short_ema_col] - df[long_ema_col]
    df['prev_delta'] = df['delta'].shift(1)

    reverse_signals = True  # Set to True to reverse the signals

    if not reverse_signals:
        # Buy: short crosses above long
        df['buy_signal'] = np.where((df['delta'] > 0) & (df['prev_delta'] <= 0), 1, 0)

        # Sell: short crosses below long
        df['sell_signal'] = np.where((df['delta'] < 0) & (df['prev_delta'] >= 0), 1, 0)
    else:
        # Reversed logic
        df['buy_signal'] = np.where((df['delta'] < 0) & (df['prev_delta'] >= 0), 1, 0)
        df['sell_signal'] = np.where((df['delta'] > 0) & (df['prev_delta'] <= 0), 1, 0)

    return df

def main():
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    start_date = '2025-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    symbol = 'HK.01810'

    all_data = pd.DataFrame()
    page_req_key = None

    while True:
        ret, data, page_req_key = quote_ctx.request_history_kline(
            symbol,
            start=start_date,
            end=end_date,
            ktype='K_15M',
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

    # Calculate EMAs
    all_data['ema_short'] = calculate_ema(all_data, short_term_ema)
    all_data['ema_long'] = calculate_ema(all_data, long_term_ema)

    # signals
    all_data = find_signal(all_data, 'ema_short', 'ema_long')

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(all_data['time_key'], all_data['close'], label='Close Price')
    plt.plot(all_data['time_key'], all_data['ema_short'], label=f'EMA {short_term_ema}', linestyle='--')
    plt.plot(all_data['time_key'], all_data['ema_long'], label=f'EMA {long_term_ema}', linestyle='--')

    # Buy signals (green up arrows)
    buy_signals = all_data[all_data['buy_signal'] == 1]
    plt.scatter(buy_signals['time_key'], buy_signals['close'], color='green', label='Buy Signal', marker='^', s=80)

    # Sell signals (red down arrows)
    sell_signals = all_data[all_data['sell_signal'] == 1]
    plt.scatter(sell_signals['time_key'], sell_signals['close'], color='red', label='Sell Signal', marker='v', s=80)

    plt.legend()
    plt.ylabel('Price')
    plt.title(f'{symbol} | EMA Crossover Signals')
    plt.xticks([], [])  # Optional: hide x-axis labels
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
