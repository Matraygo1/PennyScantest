import tkinter as tk
from tkinter import messagebox, ttk
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import robin_stocks.robinhood as r
from datetime import datetime
from sklearn.utils import shuffle

class StockPredictor:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', max_iter=2000, warm_start=True)
        self.classes = np.array([0, 1])  # Initialize classes for partial_fit

    def train(self, stock_data: pd.DataFrame):
        features, labels = self.prepare_data(stock_data)
        if len(features) == 0 or len(labels) == 0:
            raise ValueError("No data available for training")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
        self.model.fit(X_train, y_train.ravel())
        accuracy = self.model.score(X_test, y_test.ravel())
        return accuracy

    def partial_train(self, stock_data: pd.DataFrame):
        features, labels = self.prepare_data(stock_data)
        if len(features) == 0 or len(labels) == 0:
            raise ValueError("No data available for training")
        X_train, y_train = shuffle(features, labels)
        self.model.partial_fit(X_train, y_train.ravel(), classes=self.classes)

    def prepare_data(self, stock_data: pd.DataFrame):
        stock_data['RSI'] = self.compute_rsi(stock_data['Close'])
        stock_data['MACD'], stock_data['MACD_signal'] = self.compute_macd(stock_data['Close'])
        stock_data['SMA'] = self.compute_sma(stock_data['Close'])
        stock_data['EMA'] = self.compute_ema(stock_data['Close'])
        stock_data['Stochastic'] = self.compute_stochastic(stock_data)

        stock_data.dropna(inplace=True)

        features = stock_data[['Close', 'RSI', 'MACD', 'MACD_signal', 'SMA', 'EMA', 'Stochastic']].values
        labels = (stock_data['Close'].pct_change() > 0).astype(int).values
        
        return features, labels
    def compute_rsi(self, price, period=14):
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_macd(self, price, short_window=12, long_window=26, signal_window=9):
        short_ema = price.ewm(span=short_window, adjust=False).mean()
        long_ema = price.ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, macd_signal

    def compute_sma(self, price, window=20):
        return price.rolling(window=window).mean()

    def compute_ema(self, price, span=20):
        return price.ewm(span=span, adjust=False).mean()

    def compute_stochastic(self, stock_data, k_window=14, d_window=3):
        high = stock_data['High']
        low = stock_data['Low']
        close = stock_data['Close']

        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        stoch_d = stoch_k.rolling(window=d_window).mean()

        return stoch_k

    def calculate_expected_time_until_surge(self, stock_data):
        time_deltas = stock_data.index.to_series().diff().dt.total_seconds()
        average_time = time_deltas.mean() / 3600  # convert to hours
        return average_time
    def predict_price_surge(self, stock_symbol: str):
        stock_data = self.fetch_stock_data(stock_symbol)
        if stock_data is not None:
            features = self.prepare_data(stock_data)[0]
            if len(features) == 0:
                return {'surge_percent': 0, 'reason': 'No data available', 'timestamp': '', 'indicators': [], 'expected_time_until_surge': 0}
            predicted_surge = self.model.predict(features)
            predicted_surge_percent = np.mean(predicted_surge) * 100
            reasons = "Surge predicted due to strong historical price movement and technical indicators."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            indicators_used = ['RSI', 'MACD', 'SMA', 'EMA', 'Stochastic']
            expected_time_until_surge = self.calculate_expected_time_until_surge(stock_data)
            return {'surge_percent': predicted_surge_percent, 'reason': reasons, 'timestamp': timestamp, 'indicators': indicators_used, 'expected_time_until_surge': expected_time_until_surge}
        return {'surge_percent': 0, 'reason': 'No data available', 'timestamp': '', 'indicators': [], 'expected_time_until_surge': 0}

    def fetch_stock_data(self, stock_symbol: str):
        try:
            stock_data = yf.download(stock_symbol, period="1mo", interval="1h", progress=False)
            return stock_data
        except Exception as e:
            print(f"Error fetching data for {stock_symbol}: {e}")
            return None

    def plot_stock_data(self, stock_data, prediction_window):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plotting the stock data
        ax1.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.xticks(rotation=45)

        # Adding the secondary axis for RSI
        ax2 = ax1.twinx()
        ax2.plot(stock_data.index, stock_data['RSI'], label='RSI', color='red')
        ax2.set_ylabel('RSI', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(70, color='green', linestyle='--', alpha=0.5)

        fig.tight_layout()
        plt.title('Stock Price and RSI')
        plt.show()

        # Plot with Plotly for interactive graph
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=stock_data.index,
            open=stock_data["Open"],
            high=stock_data["High"],
            low=stock_data["Low"],
            close=stock_data["Close"],
            name="Candlestick"
        ))

        stock_data["SMA"] = stock_data["Close"].rolling(window=20).mean()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["SMA"], mode="lines", name="SMA (20)"))

        stock_data["RSI"] = self.compute_rsi(stock_data["Close"])
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["RSI"], mode="lines", name="RSI (14)", yaxis="y2"))

        fig.update_layout(
            title="Stock Price with Indicators",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis2=dict(
                title="RSI",
                overlaying="y",
                side="right",
                range=[0, 100]
            ),
            template="plotly_dark",
        )

        fig.show()
class AnalystPriceTargets:
    def __init__(self):
        self.analyst_targets = {
            'AAPL': {'target': 200, 'reason': "Strong earnings and upcoming product launches."},
            'MSFT': {'target': 350, 'reason': "Cloud growth and AI innovations."},
        }

    def get_analyst_target(self, stock_symbol):
        return self.analyst_targets.get(stock_symbol, {'target': None, 'reason': 'No target available'})

class RobinhoodPortfolio:
    def __init__(self, username, password):
        self.login(username, password)

    def login(self, username, password):
        r.login(username, password)

    def get_account_info(self):
        funds = r.stocks.load_account()
        available_funds = funds['cash']
        total_value = funds['equity']
        return available_funds, total_value

    def get_holdings(self):
        holdings = r.stocks.get_all_positions()
        return holdings

    def get_order_history(self):
        orders = r.stocks.get_all_open_orders()
        return orders

    def get_buy_sell_history(self):
        history = r.stocks.get_all_completed_orders()
        return history

class HomeWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Penny Stock Trader")
        self.master.geometry("800x600")

        self.is_live_mode = tk.BooleanVar(value=False)

        self.create_widgets()
    def show_results(self):
        results_window = tk.Toplevel(self.master)
        results_window.title("Scan Results")

        canvas = tk.Canvas(results_window)
        scrollbar = ttk.Scrollbar(results_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        stock_symbols = ['TNXP', 'BNGO', 'LDTC', 'TSLA', 'AMZN']
        stock_predictor = StockPredictor()

        for stock_symbol in stock_symbols:
            stock_data = stock_predictor.fetch_stock_data(stock_symbol)
            if stock_data is not None and not stock_data.empty:
                try:
                    accuracy = stock_predictor.train(stock_data)
                    prediction = stock_predictor.predict_price_surge(stock_symbol)
                    result_text = f"Ticker: {stock_symbol}\nPredicted Surge: {prediction['surge_percent']}%\nReason: {prediction['reason']}\nTimestamp: {prediction['timestamp']}\nIndicators: {', '.join(prediction['indicators'])}\nExpected Time Until Surge: {prediction['expected_time_until_surge']} hours\n\n"
                except ValueError as e:
                    result_text = f"Ticker: {stock_symbol}\nError: {e}\n\n"
            else:
                result_text = f"Ticker: {stock_symbol}\nNo data available\n\n"

            if scrollable_frame.winfo_exists():
                result_label = tk.Label(scrollable_frame, text=result_text, justify="left")
                result_label.pack(anchor="w", padx=10, pady=5)

                # Plotting enhanced graphs
                stock_predictor.plot_stock_data(stock_data, scrollable_frame)
            else:
                print(f"Error: scrollable_frame for {stock_symbol} does not exist.")
    def create_widgets(self):
        self.live_sim_toggle = tk.Checkbutton(self.master, text="Live Mode", variable=self.is_live_mode)
        self.live_sim_toggle.grid(row=0, column=0, padx=20, pady=10)

        self.start_button = tk.Button(self.master, text="Start Scanning", command=self.start_scanning)
        self.start_button.grid(row=1, column=0, padx=20, pady=10)

        self.stop_button = tk.Button(self.master, text="Stop Scanning", command=self.stop_scanning)
        self.stop_button.grid(row=2, column=0, padx=20, pady=10)

        self.results_button = tk.Button(self.master, text="Show Results", command=self.show_results)
        self.results_button.grid(row=3, column=0, padx=20, pady=10)

        self.portfolio_button = tk.Button(self.master, text="Show Portfolio", command=self.show_portfolio)
        self.portfolio_button.grid(row=4, column=0, padx=20, pady=10)

    def start_scanning(self):
        self.show_progress_window()
        live_mode = self.is_live_mode.get()
        
        stock_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "NFLX", "SPY", "QQQ"]
        total_stocks = len(stock_symbols)
        
        stock_predictor = StockPredictor()
        for i, stock_symbol in enumerate(stock_symbols):
            if live_mode:
                stock_data = stock_predictor.fetch_stock_data(stock_symbol)
                if stock_data is not None and not stock_data.empty:
                    try:
                        accuracy = stock_predictor.train(stock_data)
                        prediction = stock_predictor.predict_price_surge(stock_symbol)
                    except ValueError as e:
                        print(f"Skipping {stock_symbol}: {e}")
            self.update_progress(i + 1, total_stocks)
        
        messagebox.showinfo("Scan Completed", "Scanning for potential penny stocks is complete.")

    def show_progress_window(self):
        self.progress_window = tk.Toplevel(self.master)
        self.progress_window.title("Scanning Stocks...")

        self.progress_label = tk.Label(self.progress_window, text="Scanning in Progress...")
        self.progress_label.grid(row=0, column=0, padx=20, pady=10)

        self.progress_bar = ttk.Progressbar(self.progress_window, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.grid(row=1, column=0, padx=20, pady=10)

        self.progress_bar["value"] = 0

    def update_progress(self, current, total):
        progress = (current / total) * 100
        self.progress_bar["value"] = progress
        self.progress_window.update_idletasks()

    def stop_scanning(self):
        messagebox.showinfo("Scan Stopped", "Scan process has been stopped.")
        if hasattr(self, "progress_window"):
            self.progress_window.destroy()

    def show_results(self):
        results_window = tk.Toplevel(self.master)
        results_window.title("Scan Results")

        canvas = tk.Canvas(results_window)
        scrollbar = ttk.Scrollbar(results_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        stock_symbols = ['TNXP', 'BNGO', 'LDTC', 'TSLA', 'AMZN']
        stock_predictor = StockPredictor()

        for stock_symbol in stock_symbols:
            stock_data = stock_predictor.fetch_stock_data(stock_symbol)
            if stock_data is not None and not stock_data.empty:
                try:
                    accuracy = stock_predictor.train(stock_data)
                    prediction = stock_predictor.predict_price_surge(stock_symbol)
                    result_text = f"Ticker: {stock_symbol}\nPredicted Surge: {prediction['surge_percent']}%\nReason: {prediction['reason']}\nTimestamp: {prediction['timestamp']}\nIndicators: {', '.join(prediction['indicators'])}\nExpected Time Until Surge: {prediction['expected_time_until_surge']} hours\n\n"
                except ValueError as e:
                    result_text = f"Ticker: {stock_symbol}\nError: {e}\n\n"
            else:
                result_text = f"Ticker: {stock_symbol}\nNo data available\n\n"
            
            result_label = tk.Label(scrollable_frame, text=result_text, justify="left")
            result_label.pack(anchor="w", padx=10, pady=5)

            # Plotting enhanced graphs
            stock_predictor.plot_stock_data(stock_data, scrollable_frame)

    def show_portfolio(self):
        robinhood_account = RobinhoodPortfolio(username="your_username", password="your_password")
        
        available_funds, total_value = robinhood_account.get_account_info()
        holdings = robinhood_account.get_holdings()
        pending_orders = robinhood_account.get_order_history()
        buy_sell_history = robinhood_account.get_buy_sell_history()
        
        portfolio_window = tk.Toplevel(self.master)
        portfolio_window.title("Portfolio Status")

        available_funds_label = tk.Label(portfolio_window, text=f"Available Funds: ${available_funds}")
        available_funds_label.grid(row=0, column=0, padx=10, pady=5)

        total_value_label = tk.Label(portfolio_window, text=f"Total Investment: ${total_value}")
        total_value_label.grid(row=1, column=0, padx=10, pady=5)

        holdings_label = tk.Label(portfolio_window, text="Holdings:")
        holdings_label.grid(row=2, column=0, padx=10, pady=5)
        row = 3
        for holding in holdings:
            symbol = holding["instrument"]["symbol"]
            quantity = holding["quantity"]
            price = holding["average_buy_price"]
            holding_label = tk.Label(portfolio_window, text=f"{symbol}: {quantity} shares at ${price}")
            holding_label.grid(row=row, column=0, padx=10, pady=5)
            row += 1

        pending_orders_label = tk.Label(portfolio_window, text="Pending Orders:")
        pending_orders_label.grid(row=row, column=0, padx=10, pady=5)
        row += 1
        for order in pending_orders:
            order_id = order["id"]
            instrument = order["instrument"]["symbol"]
            quantity = order["quantity"]
            price = order["price"]
            order_label = tk.Label(portfolio_window, text=f"Order {order_id}: {instrument} x {quantity} at ${price}")
            order_label.grid(row=row, column=0, padx=10, pady=5)
            row += 1

        buy_sell_history_label = tk.Label(portfolio_window, text="Buy/Sell History:")
        buy_sell_history_label.grid(row=row, column=0, padx=10, pady=5)
        row += 1
        for history in buy_sell_history:
            action = history["side"]
            instrument = history["instrument"]["symbol"]
            quantity = history["quantity"]
            price = history["price"]
            history_label = tk.Label(portfolio_window, text=f"{action.capitalize()} {quantity} {instrument} at ${price}")
            history_label.grid(row=row, column=0, padx=10, pady=5)

def main():
    root = tk.Tk()
    home_window = HomeWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()