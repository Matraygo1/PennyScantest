import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import robin_stocks.robinhood as r

# Stock data and analysis prediction class
class StockPredictor:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=300)
        
    def train(self, stock_data: pd.DataFrame):
        # Prepare the features (technical indicators)
        features, labels = self.prepare_data(stock_data)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = self.model.score(X_test, y_test)
        return accuracy

    def prepare_data(self, stock_data: pd.DataFrame):
        # Adding technical indicators to the stock data
        stock_data['RSI'] = self.compute_rsi(stock_data['Close'])
        stock_data['MACD'], stock_data['MACD_signal'] = self.compute_macd(stock_data['Close'])
        stock_data['SMA'] = self.compute_sma(stock_data['Close'])
        stock_data['EMA'] = self.compute_ema(stock_data['Close'])
        stock_data['Stochastic'] = self.compute_stochastic(stock_data)

        stock_data.dropna(inplace=True)  # Drop rows with NaN values

        # Use technical indicators and price as features
        features = stock_data[['Close', 'RSI', 'MACD', 'MACD_signal', 'SMA', 'EMA', 'Stochastic']].values
        labels = (stock_data['Close'].pct_change() > 0).astype(int).values  # Price movement direction (1: up, 0: down)
        
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
        # Placeholder implementation, replace with your actual logic
        # Example: Calculate the average time between surges
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
        # Fetch stock data from Yahoo Finance
        try:
            stock_data = yf.download(stock_symbol, period="3mo", interval="15m")
            return stock_data
        except Exception as e:
            print(f"Error fetching data for {stock_symbol}: {e}")
            return None
        # Analyst price targets class (for simplicity, simulated data)
class AnalystPriceTargets:
    def __init__(self):
        self.analyst_targets = {
            'AAPL': {'target': 200, 'reason': "Strong earnings and upcoming product launches."},
            'MSFT': {'target': 350, 'reason': "Cloud growth and AI innovations."},
            # Add more stocks and analyst predictions
        }

    def get_analyst_target(self, stock_symbol):
        return self.analyst_targets.get(stock_symbol, {'target': None, 'reason': 'No target available'})


# Robinhood Portfolio Integration
class RobinhoodPortfolio:
    def __init__(self, username, password):
        # Login to Robinhood account
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
    # Home window with GUI for controlling scan and training
class HomeWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Penny Stock Trader")
        self.master.geometry("800x600")

        self.is_live_mode = tk.BooleanVar(value=False)  # Live Mode toggle (default to False)

        self.create_widgets()

    def create_widgets(self):
        # Live or Simulated Mode Toggle
        self.live_sim_toggle = tk.Checkbutton(self.master, text="Live Mode", variable=self.is_live_mode)
        self.live_sim_toggle.grid(row=0, column=0, padx=20, pady=10)

        # Start Scanning Button
        self.start_button = tk.Button(self.master, text="Start Scanning", command=self.start_scanning)
        self.start_button.grid(row=1, column=0, padx=20, pady=10)

        # Stop Scanning Button
        self.stop_button = tk.Button(self.master, text="Stop Scanning", command=self.stop_scanning)
        self.stop_button.grid(row=2, column=0, padx=20, pady=10)

        # Results Window Button (for stocks that match criteria)
        self.results_button = tk.Button(self.master, text="Show Results", command=self.show_results)
        self.results_button.grid(row=3, column=0, padx=20, pady=10)

        # Portfolio Status Section
        self.portfolio_button = tk.Button(self.master, text="Show Portfolio", command=self.show_portfolio)
        self.portfolio_button.grid(row=4, column=0, padx=20, pady=10)
        def start_scanning(self):
        # Start real scanning process
         self.show_progress_window()
        live_mode = self.is_live_mode.get()
        
        # Example stock symbols
        stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX', 'SPY', 'QQQ']
        total_stocks = len(stock_symbols)
        
        for i, stock_symbol in enumerate(stock_symbols):
            if live_mode:
                stock_predictor = StockPredictor()
                stock_data = stock_predictor.fetch_stock_data(stock_symbol)
                if stock_data is not None:
                    accuracy = stock_predictor.train(stock_data)
                    prediction = stock_predictor.predict_price_surge(stock_symbol)
            self.update_progress(i + 1, total_stocks)
        
        messagebox.showinfo("Scan Completed", "Scanning for potential penny stocks is complete.")

    def show_progress_window(self):
        self.progress_window = tk.Toplevel(self.master)
        self.progress_window.title("Scanning Stocks...")

        # Create a progress bar
        self.progress_label = tk.Label(self.progress_window, text="Scanning in Progress...")
        self.progress_label.grid(row=0, column=0, padx=20, pady=10)

        self.progress_bar = ttk.Progressbar(self.progress_window, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.grid(row=1, column=0, padx=20, pady=10)

        self.progress_bar["value"] = 0  # Initialize progress

    def update_progress(self, current, total):
        progress = (current / total) * 100
        self.progress_bar["value"] = progress
        self.progress_window.update_idletasks()  # Update the progress bar

    def stop_scanning(self):
        # Placeholder for stopping the scanning process
        messagebox.showinfo("Scan Stopped", "Scan process has been stopped.")
        if hasattr(self, 'progress_window'):
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

        stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
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

    def show_portfolio(self):
        # Get Robinhood account information
        robinhood_account = RobinhoodPortfolio(username="your_username", password="your_password")
        
        available_funds, total_value = robinhood_account.get_account_info()
        holdings = robinhood_account.get_holdings()
        pending_orders = robinhood_account.get_order_history()
        buy_sell_history = robinhood_account.get_buy_sell_history()
        
        # Display portfolio information in a new window
        portfolio_window = tk.Toplevel(self.master)
        portfolio_window.title("Portfolio Status")

        # Available Funds
        available_funds_label = tk.Label(portfolio_window, text=f"Available Funds: ${available_funds}")
        available_funds_label.grid(row=0, column=0, padx=10, pady=5)

        # Total Investment Value
        total_value_label = tk.Label(portfolio_window, text=f"Total Investment: ${total_value}")
        total_value_label.grid(row=1, column=0, padx=10, pady=5)

        # Holdings
        holdings_label = tk.Label(portfolio_window, text="Holdings:")
        holdings_label.grid(row=2, column=0, padx=10, pady=5)
        row = 3
        for holding in holdings:
            symbol = holding['instrument']['symbol']
            quantity = holding['quantity']
            price = holding['average_buy_price']
            holding_label = tk.Label(portfolio_window, text=f"{symbol}: {quantity} shares at ${price}")
            holding_label.grid(row=row, column=0, padx=10, pady=5)
            row += 1

        # Pending Orders
        pending_orders_label = tk.Label(portfolio_window, text="Pending Orders:")
        pending_orders_label.grid(row=row, column=0, padx=10, pady=5)
        row += 1
        for order in pending_orders:
            order_id = order['id']
            instrument = order['instrument']['symbol']
            quantity = order['quantity']
            price = order['price']
            order_label = tk.Label(portfolio_window, text=f"Order {order_id}: {instrument} x {quantity} at ${price}")
            order_label.grid(row=row, column=0, padx=10, pady=5)
            row += 1

        # Buy/Sell History
        buy_sell_history_label = tk.Label(portfolio_window, text="Buy/Sell History:")
        buy_sell_history_label.grid(row=row, column=0, padx=10, pady=5)
        row += 1
        for history in buy_sell_history:
            action = history['side']
            instrument = history['instrument']['symbol']
            quantity = history['quantity']
            price = history['price']
            history_label = tk.Label(portfolio_window, text=f"{action.capitalize()} {quantity} {instrument} at ${price}")
            history_label.grid(row=row, column=0, padx=10, pady=5)
            row += 1
            def train_on_stock(self, stock_symbol):
        # Train the model on the selected stock and show the prediction window
             stock_predictor = StockPredictor()
        stock_data = stock_predictor.fetch_stock_data(stock_symbol)
        if stock_data is not None and not stock_data.empty:
            accuracy = stock_predictor.train(stock_data)
            prediction = stock_predictor.predict_price_surge(stock_symbol)
            self.display_prediction_window(stock_symbol, prediction)

    def display_prediction_window(self, stock_symbol, predictions):
        prediction_window = tk.Toplevel(self.master)
        prediction_window.title(f"Prediction Window - {stock_symbol}")
        
        # Display predictions and analysis
        prediction_label = tk.Label(prediction_window, text=f"Predicted Surge: {predictions['surge_percent']}%\nReason: {predictions['reason']}\nTimestamp: {predictions['timestamp']}\nIndicators: {', '.join(predictions['indicators'])}\nExpected Time Until Surge: {predictions['expected_time_until_surge']} hours")
        prediction_label.grid(row=0, column=0, padx=10, pady=10)

        # Display analyst predictions and price targets
        analyst_targets = AnalystPriceTargets()
        target_data = analyst_targets.get_analyst_target(stock_symbol)
        target_label = tk.Label(prediction_window, text=f"Analyst Target: ${target_data['target']}\nReason: {target_data['reason']}")
        target_label.grid(row=1, column=0, padx=10, pady=10)

        # Display a plot of the stock price with technical indicators
        stock_predictor = StockPredictor()
        stock_data = stock_predictor.fetch_stock_data(stock_symbol)
        self.plot_stock_data(stock_data, prediction_window)
        def plot_stock_data(self, stock_data, prediction_window):
            fig = go.Figure()

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='Candlestick'
        ))

        # Add moving average (SMA) to the chart
        stock_data['SMA'] = stock_data['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA'], mode='lines', name='SMA (20)'))

        # Add RSI plot
        stock_data['RSI'] = self.compute_rsi(stock_data['Close'])
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI (14)', yaxis="y2"))

        # Customize the layout
        fig.update_layout(
            title=f"Stock Price with Indicators",
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

        # Show the plot in the results window
        fig.show()

# Main program to run the GUI
def main():
    root = tk.Tk()
    home_window = HomeWindow(root)
    root
