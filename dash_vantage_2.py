import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import plotly.express as px
import plotly.graph_objects as go

from alpha_vantage.timeseries import TimeSeries
from dotenv import find_dotenv, load_dotenv
import os

import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html

# --- Configuration ---
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

VANTAGE_API_KEY = os.environ.get("VANTAGE_API_KEY")

# Default configuration (will be updated by Dash inputs)
config = {
    "alpha_vantage": {
        "key": VANTAGE_API_KEY,
        "symbol": "AAPL", # Default symbol
        "outputsize": "full",
        "key_close": "4. close",
    },
    "data": {
        "window_size": 20, # Hyperparameter
        "train_split_size": 0.80,
    },
    "plots": {
        "color_actual": "#001f3f",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1,
        "num_lstm_layers": 2, # Hyperparameter
        "lstm_size": 32,      # Hyperparameter
        "dropout": 0.2,       # Hyperparameter
    },
    "training": {
        "device": "cpu", # Set to "cuda" if GPU is available
        "batch_size": 64,
        "num_epoch": 100, # Hyperparameter
        "learning_rate": 0.01, # Hyperparameter
        "scheduler_step_size": 40,
    }
}

# --- Helper Classes & Functions ---

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, 2)
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        x = self.linear_1(x)
        x = self.relu(x)

        lstm_out, (h_n, c_n) = self.lstm(x)

        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]


# Cache for downloaded data to avoid hitting API limits repeatedly
_cached_data = {}

def download_data(current_config):
    """
    Download data from Alpha Vantage and cache it to avoid hitting API limits.
    Will return cached data if it has been downloaded before.

    Parameters
    ----------
    current_config : dict
        A dictionary with configuration parameters for data downloading.

    Returns
    -------
    tuple
        A tuple of (data_date, data_close_price, num_data_points, display_date_range).
    """
    symbol = current_config["alpha_vantage"]["symbol"]
    if symbol in _cached_data:
        print(f"Using cached data for {symbol}")
        return _cached_data[symbol]

    ts = TimeSeries(key=current_config["alpha_vantage"]["key"])
    try:
        data, meta_data = ts.get_daily(symbol, outputsize=current_config["alpha_vantage"]["outputsize"])
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}. Check API key or symbol.")
        return None, None, None, None # Indicate failure

    if not data:
        print(f"No data received for {symbol}. Check API key or symbol.")
        return None, None, None, None # Indicate failure

    data_date = [date for date in data.keys()]
    data_date.reverse() # Ensure chronological order

    data_close_price = [float(data[date][current_config["alpha_vantage"]["key_close"]]) for date in data.keys()]
    data_close_price.reverse() # Ensure chronological order
    data_close_price = np.array(data_close_price)

    num_data_points = len(data_date)
    display_date_range = f"from {data_date[0]} to {data_date[num_data_points-1]}"
    print(f"Downloaded {num_data_points} data points for {symbol} {display_date_range}")

    _cached_data[symbol] = (data_date, data_close_price, num_data_points, display_date_range)
    return _cached_data[symbol]



def prepare_data_x(x, window_size):
    """
    Prepare data for training and validation.

    Given a time series `x` and a window size `window_size`, this function
    creates the sequences of data points that are used for training and
    validation. The last sequence is used for making predictions about the
    next day.

    Parameters
    ----------
    x : array_like
        The time series data
    window_size : int
        The size of the window

    Returns
    -------
    tuple
        A tuple of two arrays. The first array contains the sequences of data
        points used for training and validation, and the second array contains
        the last sequence used for making predictions about the next day.
    """
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1] # sequences for training/validation, last sequence for next-day prediction
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1] # sequences for training/validation, last sequence for next-day prediction


def prepare_data_y(x, window_size):
    output = x[window_size:] # The price immediately following each window
    return output


def run_epoch(dataloader, model, criterion, optimizer, scheduler, current_config, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        x = x.to(current_config["training"]["device"])
        y = y.to(current_config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / x.shape[0])

    # Scheduler step is usually done once per epoch after all batches
    if is_training and scheduler:
        scheduler.step()

    lr = scheduler.get_last_lr()[0] if scheduler else current_config["training"]["learning_rate"]

    return epoch_loss, lr


# --- Dash App Layout ---
app = dash.Dash(__name__)
server = app.server # This is for Gunicorn/Heroku deployment

# Stock Symbols
company_options = [
    {'label': 'Apple (AAPL)', 'value': 'AAPL'},
    {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
    {'label': 'Alphabet (GOOGL)', 'value': 'GOOGL'},
    {'label': 'Amazon (AMZN)', 'value': 'AMZN'},
    {'label': 'NVIDIA (NVDA)', 'value': 'NVDA'},
    {'label': 'Meta Platforms (META)', 'value': 'META'},
    {'label': 'Tesla (TSLA)', 'value': 'TSLA'},
    {'label': 'Netflix (NFLX)', 'value': 'NFLX'},
    {'label': 'Johnson & Johnson (JNJ)', 'value': 'JNJ'},
    {'label': 'Coca-Cola (KO)', 'value': 'KO'},
    {'label': 'Procter & Gamble (PG)', 'value': 'PG'},
    {'label': 'UnitedHealth Group (UNH)', 'value': 'UNH'},
    {'label': 'JPMorgan Chase & Co. (JPM)', 'value': 'JPM'},
    {'label': 'Visa Inc. (V)', 'value': 'V'},
    {'label': 'Mastercard Inc. (MA)', 'value': 'MA'},
    {'label': 'Walt Disney (DIS)', 'value': 'DIS'},
    {'label': 'Home Depot (HD)', 'value': 'HD'},
    {'label': 'McDonald\'s (MCD)', 'value': 'MCD'},
    {'label': 'Nike (NKE)', 'value': 'NKE'},
    {'label': 'Salesforce (CRM)', 'value': 'CRM'},
    {'label': 'Adobe (ADBE)', 'value': 'ADBE'},
    {'label': 'PayPal (PYPL)', 'value': 'PYPL'},
    {'label': 'Intel (INTC)', 'value': 'INTC'},
    {'label': 'Verizon (VZ)', 'value': 'VZ'},
    {'label': 'AT&T (T)', 'value': 'T'},
    {'label': 'Comcast (CMCSA)', 'value': 'CMCSA'},
    {'label': 'PepsiCo (PEP)', 'value': 'PEP'},
    {'label': 'Costco (COST)', 'value': 'COST'},
    {'label': 'Texas Instruments (TXN)', 'value': 'TXN'},
    {'label': 'Broadcom (AVGO)', 'value': 'AVGO'},
    {'label': 'Qualcomm (QCOM)', 'value': 'QCOM'},
    {'label': 'Cisco (CSCO)', 'value': 'CSCO'},
    {'label': 'Oracle (ORCL)', 'value': 'ORCL'},
    {'label': 'Walmart (WMT)', 'value': 'WMT'},
]

app.layout = html.Div([
    html.H1("Stock Price Prediction with LSTM", style={'textAlign': 'center', 'color': config["plots"]["color_actual"]}),

    html.Div([ # Controls Container
        html.Div([ # Company Selector
            html.Label("Select Company:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='company-selector',
                options=company_options,
                value='AAPL',
                clearable=False,
                style={'marginBottom': '10px'}
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),

        html.Div([ # Training Epochs
            html.Label("Training Epochs:", style={'fontWeight': 'bold'}),
            dcc.Input(
                id='num-epoch-input',
                type='number',
                value=config["training"]["num_epoch"],
                min=1,
                max=500,
                step=1,
                style={'width': '100%'}
            ),
            html.Small("Number of times the model sees the entire training dataset. Too few: underfitting; Too many: overfitting.", style={'color': 'gray', 'fontSize': '0.8em'})
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),

        html.Div([ # Window Size
            html.Label("Window Size (days):", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='window-size-slider',
                min=5, max=60, step=5,
                value=config["data"]["window_size"],
                marks={i: str(i) for i in range(5, 61, 5)},
            ),
            html.Small("Number of past days' prices the model uses to predict the next day. Adjust to capture short-term vs. long-term dependencies.", style={'color': 'gray', 'fontSize': '0.8em'})
        ], style={'width': '30%', 'display': 'inline-block'}),

        html.Div([ # LSTM Hidden Layer Size
            html.Label("LSTM Hidden Size:", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='lstm-size-slider',
                min=8, max=128, step=8,
                value=config["model"]["lstm_size"],
                marks={i: str(i) for i in [8, 16, 32, 64, 128]},
            ),
            html.Small("Capacity of each LSTM layer. Larger values can learn more complex patterns but risk overfitting.", style={'color': 'gray', 'fontSize': '0.8em'})
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%', 'marginTop': '20px'}),

        html.Div([ # Number of LSTM Layers
            html.Label("Number of LSTM Layers:", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='num-lstm-layers-slider',
                min=1, max=5, step=1,
                value=config["model"]["num_lstm_layers"],
                marks={i: str(i) for i in range(1, 6)},
            ),
            html.Small("Number of stacked LSTM layers. More layers can capture hierarchical features but may increase complexity and training time.", style={'color': 'gray', 'fontSize': '0.8em'})
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%', 'marginTop': '20px'}),

        html.Div([ # Learning Rate
            html.Label("Learning Rate:", style={'fontWeight': 'bold'}),
            dcc.Input(
                id='learning-rate-input',
                type='text',
                value=config["training"]["learning_rate"],
                min=0.00001, max=0.1, step=0.0001,
                style={'width': '100%'}
            ),
            html.Small("Step size the optimizer takes to adjust weights. Too high: model diverges; Too low: slow training.", style={'color': 'gray', 'fontSize': '0.8em'})
        ], style={'width': '30%', 'display': 'inline-block', 'marginTop': '20px'}),

        html.Div([ # Dropout Rate
            html.Label("Dropout Rate:", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='dropout-slider',
                min=0.0, max=0.5, step=0.05,
                value=config["model"]["dropout"],
                marks={i/10: str(i/10) for i in range(0, 6, 1)},
            ),
            html.Small("Percentage of neurons randomly deactivated during training to prevent overfitting.", style={'color': 'gray', 'fontSize': '0.8em'})
        ], style={'width': '30%', 'display': 'inline-block', 'marginTop': '20px', 'marginRight': '68%'}), # Use margin-right to push button to new line


        html.Button(
            'Run Prediction',
            id='run-button',
            n_clicks=0,
            style={
                'width': '100%',
                'padding': '15px',
                'fontSize': '20px',
                'backgroundColor': config["plots"]["color_pred_test"],
                'color': 'white',
                'border': 'none',
                'borderRadius': '8px',
                'cursor': 'pointer',
                'marginTop': '30px'
            }
        ),

    ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'marginBottom': '30px', 'backgroundColor': '#f9f9f9'}),


    dcc.Loading(
        id="loading-output",
        type="circle", # You can choose 'graph', 'cube', 'dot', 'default'
        children=html.Div([
            html.H3(id='predicted-next-day-price-output', style={'textAlign': 'center', 'marginTop': '30px', 'fontSize': '28px', 'color': config["plots"]["color_pred_test"]}),
            dcc.Graph(id='combined-prediction-graph', style={'height': '500px'}),
            dcc.Graph(id='next-day-prediction-graph', style={'height': '500px'})
        ])
    )
], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1400px', 'margin': 'auto', 'padding': '30px', 'backgroundColor': '#fff'})


# --- Dash App Callbacks ---
@app.callback(
    Output('combined-prediction-graph', 'figure'),
    Output('next-day-prediction-graph', 'figure'),
    Output('predicted-next-day-price-output', 'children'),
    Input('run-button', 'n_clicks'),
    State('company-selector', 'value'),
    State('num-epoch-input', 'value'),
    State('window-size-slider', 'value'),
    State('lstm-size-slider', 'value'),
    State('num-lstm-layers-slider', 'value'),
    State('learning-rate-input', 'value'),
    State('dropout-slider', 'value')
)
def update_graphs(n_clicks, symbol, num_epoch, window_size, lstm_size, num_lstm_layers, learning_rate, dropout):
    # This ensures the callback doesn't run on initial load before the button is clicked
    # and returns empty figures/message.
    if n_clicks == 0:
        return go.Figure(), go.Figure(), "Click 'Run Prediction' to generate forecasts."

    # Update config based on user input for this specific run
    current_config = config.copy()
    current_config["alpha_vantage"]["symbol"] = symbol
    current_config["data"]["window_size"] = window_size
    current_config["model"]["num_lstm_layers"] = num_lstm_layers
    current_config["model"]["lstm_size"] = lstm_size
    current_config["model"]["dropout"] = dropout
    current_config["training"]["num_epoch"] = num_epoch
    current_config["training"]["learning_rate"] = float(learning_rate)

    # --- 1. Data Download ---
    data_date, data_close_price, num_data_points, display_date_range = download_data(current_config)

    if data_date is None:
        # Return empty figures and an error message if data download fails
        return go.Figure().update_layout(title="Data Download Error"), \
               go.Figure().update_layout(title="Data Download Error"), \
               f"Error: Could not download data for {symbol}. Please check your API key or try again later."


    # --- 2. Data Preprocessing ---
    scaler = Normalizer()
    normalized_data_close_price = scaler.fit_transform(data_close_price)

    # Check if data_x and data_y can be formed with the given window_size
    if len(normalized_data_close_price) < current_config["data"]["window_size"] + 1:
        return go.Figure().update_layout(title="Data Error"), \
               go.Figure().update_layout(title="Data Error"), \
               f"Error: Not enough data points ({len(normalized_data_close_price)}) for the selected window size ({current_config['data']['window_size']}). Please choose a smaller window size or a different stock."

    data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=current_config["data"]["window_size"])
    data_y = prepare_data_y(normalized_data_close_price, window_size=current_config["data"]["window_size"])

    split_index = int(data_y.shape[0]*current_config["data"]["train_split_size"])

    # Ensure there's enough data for both train and validation splits
    if split_index == 0 or split_index >= len(data_y):
        return go.Figure().update_layout(title="Data Split Error"), \
               go.Figure().update_layout(title="Data Split Error"), \
               f"Error: Not enough data for valid train/validation split with current window size and train split ratio ({current_config['data']['train_split_size']}). Adjust window size or ratio."


    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

    # --- 3. Model Initialization & Training ---
    train_dataloader = DataLoader(dataset_train, batch_size=current_config["training"]["batch_size"], shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=current_config["training"]["batch_size"], shuffle=True)

    model = LSTMModel(input_size=current_config["model"]["input_size"],
                      hidden_layer_size=current_config["model"]["lstm_size"],
                      num_layers=current_config["model"]["num_lstm_layers"],
                      output_size=1,
                      dropout=current_config["model"]["dropout"])
    model = model.to(current_config["training"]["device"])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=current_config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=current_config["training"]["scheduler_step_size"], gamma=0.1)

    # The actual training loop
    for epoch in range(current_config["training"]["num_epoch"]):
        run_epoch(train_dataloader, model, criterion, optimizer, scheduler, current_config, is_training=True)
        run_epoch(val_dataloader, model, criterion, optimizer, scheduler, current_config, is_training=False)


    # --- 4. Make Predictions ---
    model.eval() # Set model to evaluation mode for consistent predictions

    # Re-initialize dataloaders without shuffling for sequential predictions required for plotting
    train_dataloader_eval = DataLoader(dataset_train, batch_size=current_config["training"]["batch_size"], shuffle=False)
    val_dataloader_eval = DataLoader(dataset_val, batch_size=current_config["training"]["batch_size"], shuffle=False)

    predicted_train = np.array([])
    for idx, (x, y) in enumerate(train_dataloader_eval):
        x = x.to(current_config["training"]["device"])
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_train = np.concatenate((predicted_train, out))

    predicted_val = np.array([])
    for idx, (x, y) in enumerate(val_dataloader_eval):
        x = x.to(current_config["training"]["device"])
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_val = np.concatenate((predicted_val, out))

    # Inverse transform predictions to original price scale for plotting
    to_plot_data_y_train_pred = np.zeros(num_data_points)
    to_plot_data_y_val_pred = np.zeros(num_data_points)

    to_plot_data_y_train_pred[current_config["data"]["window_size"]:split_index+current_config["data"]["window_size"]] = scaler.inverse_transform(predicted_train)
    to_plot_data_y_val_pred[split_index+current_config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

    # Plotly handles None values by creating gaps in lines, which is desired for empty sections
    to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

    # --- Generate Graph 1: Combined Predicted vs. Actual Prices ---
    df_plot_combined = pd.DataFrame({
        'Date': data_date,
        'Actual Price': data_close_price,
        'Predicted Train Price': to_plot_data_y_train_pred,
        'Predicted Validation Price': to_plot_data_y_val_pred
    })
    df_plot_combined_melted = df_plot_combined.melt(id_vars=['Date'], var_name='Type', value_name='Price').dropna(subset=['Price'])

    color_map_combined = {
        'Actual Price': current_config["plots"]["color_actual"],
        'Predicted Train Price': current_config["plots"]["color_pred_train"],
        'Predicted Validation Price': current_config["plots"]["color_pred_val"]
    }

    fig_combined = px.line(df_plot_combined_melted, x='Date', y='Price', color='Type',
                           color_discrete_map=color_map_combined,
                           title=f"Predicted vs. Actual Prices for {symbol}")
    fig_combined.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )


    # --- Generate Graph 2: Recent Performance + Next Day Prediction ---
    x_unseen_tensor = torch.tensor(data_x_unseen).float().to(current_config["training"]["device"]).unsqueeze(0).unsqueeze(2)
    prediction_tomorrow_normalized = model(x_unseen_tensor)
    predicted_tomorrow_price = scaler.inverse_transform(prediction_tomorrow_normalized.cpu().detach().numpy())[0]

    plot_range = 10 # Days to show in the zoomed-in plot
    actual_val_last_segment = scaler.inverse_transform(data_y_val)[-plot_range+1:]
    predicted_val_last_segment = scaler.inverse_transform(predicted_val)[-plot_range+1:]

    plot_date_test = data_date[-plot_range+1:]
    plot_date_test.append("Tomorrow") # Label for the predicted future point

    fig_next_day = go.Figure()

    # Add Actual prices trace (last part of validation)
    fig_next_day.add_trace(go.Scatter(
        x=plot_date_test[:-1], # Dates up to the last actual day
        y=actual_val_last_segment,
        mode='lines+markers',
        name='Actual prices',
        marker=dict(size=10, color=current_config["plots"]["color_actual"]),
        line=dict(color=current_config["plots"]["color_actual"]),
        hovertemplate="<b>Date</b>: %{x}<br><b>Actual Price</b>: %{y:.2f}<extra></extra>"
    ))

    # Add Past predicted prices trace (last part of validation)
    fig_next_day.add_trace(go.Scatter(
        x=plot_date_test[:-1], # Dates up to the last actual day
        y=predicted_val_last_segment,
        mode='lines+markers',
        name='Past predicted prices',
        marker=dict(size=10, color=current_config["plots"]["color_pred_val"]),
        line=dict(color=current_config["plots"]["color_pred_val"]),
        hovertemplate="<b>Date</b>: %{x}<br><b>Predicted Price (Past)</b>: %{y:.2f}<extra></extra>"
    ))

    # Add Predicted price for next day trace
    fig_next_day.add_trace(go.Scatter(
        x=[plot_date_test[-1]], # Only 'Tomorrow'
        y=[predicted_tomorrow_price],
        mode='markers', # Only marker, no line connecting to previous points
        name='Predicted price for next day',
        marker=dict(size=20, color=current_config["plots"]["color_pred_test"]),
        hovertemplate="<b>Date</b>: %{x}<br><b>Predicted Price (Tomorrow)</b>: %{y:.2f}<extra></extra>"
    ))

    fig_next_day.update_layout(
        title=f"Predicted Close Price for {symbol} (Next Trading Day)",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig_next_day.update_xaxes(type='category') # Treat 'Tomorrow' as a category for correct display

    return fig_combined, fig_next_day, f"Predicted close price for next trading day: ${predicted_tomorrow_price:.2f}"


if __name__ == '__main__':
    app.run(debug=True)