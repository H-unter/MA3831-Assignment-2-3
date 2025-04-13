import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import ast
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import SimpleRNN, Dense, concatenate, Input
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import pandas as pd
import os



price_data = pd.read_parquet('data/stock_data.parquet')
sentiment_data = pd.read_csv('data/sentiment_data.csv', converters={'stock_codes': ast.literal_eval})
sentiment_data['datetime'] = pd.to_datetime(sentiment_data['date_string'])
sentiment_data

def get_stock_sentiment(stock_ticker, sentiment_data):
  search_term = f'ASX:{stock_ticker.upper()}'  # Create search term
  filtered_data = sentiment_data[sentiment_data['stock_codes'].apply(lambda codes: search_term in codes)]
  stock_sentiment_df = filtered_data[['datetime', 'vader_negative', 'vader_neutral', 'vader_positive', 'vader_compound', 'bert_sentiment_score']].copy()
  stock_sentiment_df.set_index('datetime', inplace=True)
  return stock_sentiment_df

def get_raw_stock_data(stock_ticker, df=price_data):
  """given the format BHP, returns df['BHP.AX']"""
  return df[f'{stock_ticker.upper()}.AX'].copy()

def prepare_data(stock_name, sentiment_data, control_columns, sentiment_columns):
    """Merge stock price and sentiment data and create dataset variants."""
    sentiment_df = get_stock_sentiment(stock_name, sentiment_data)
    price_df = get_raw_stock_data(stock_name)
    merged_df = pd.merge(price_df, sentiment_df, left_index=True, right_index=True, how='left').fillna(0)
    
    data = {}
    data['Control'] = merged_df[control_columns].copy()
    for key, cols in sentiment_columns.items():
        data[key] = merged_df[control_columns + cols].copy()
    return data

def create_dataset(dataset, lookback, forecast):
    """Create lookback dataset for time series prediction."""
    X, Y = [], []
    max_index = len(dataset) - lookback - forecast
    for i in range(max_index):
        X.append(dataset.iloc[i:i + lookback].values)
        Y.append(dataset['Close'].iloc[i + lookback + forecast - 1])
    return np.array(X), np.array(Y)

def train_test_split_data(data_df, train_proportion, lookback, forecast):
    """Split, scale, and create lookback datasets."""
    train_size = int(len(data_df) * train_proportion)
    train_data = data_df.iloc[:train_size].copy()
    test_data = data_df.iloc[train_size:].copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_scaled = pd.DataFrame(scaler.transform(train_data), columns=train_data.columns, index=train_data.index)
    test_scaled = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns, index=test_data.index)
    
    X_train, Y_train = create_dataset(train_scaled, lookback, forecast)
    X_test, Y_test = create_dataset(test_scaled, lookback, forecast)
    return X_train, Y_train, X_test, Y_test, scaler

def create_lstm_model(input_shape):
    """Create and compile an LSTM model."""
    optimizer = Adam(learning_rate=0.001)
    model = Sequential()
    model.add(LSTM(64, activation="tanh", return_sequences=True, input_shape=input_shape))
    model.add(LSTM(32, activation="tanh"))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_and_train_model(X_train, Y_train, input_shape, max_epochs, patience):
    """Train an LSTM model with early stopping."""
    model = create_lstm_model(input_shape)
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(X_train, Y_train, epochs=max_epochs, batch_size=64,
                        validation_split=0.3, callbacks=[early_stop], verbose=1)
    return model, history

def make_predictions(model, X, Y_true):
    """Predict values using a trained model."""
    Y_pred = model.predict(X).reshape(-1)
    return Y_true, Y_pred

def calculate_metrics(y_true, y_pred):
    """Calculate MSE, RMSE, and MAE."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, rmse, mae

def plot_training_loss(loss_dict, stock_name, output_dir):
    """Plot the training loss vs epochs for each model."""
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_dict.get('Control', []), label='Control Model', linewidth=2, color='black')
    plt.semilogy(loss_dict.get('Vader', []), label='Vader Model', linewidth=2, color='red')
    plt.semilogy(loss_dict.get('BERT', []), label='Bert Model', linewidth=2, color='blue')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.gca().yaxis.set_minor_formatter(plt.NullFormatter())
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss of Different RNN Models')
    plt.legend()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'{stock_name}_model_training_loss.pdf'), format='pdf')
    plt.show()

def plot_test_results(test_dates, actual, predictions, stock_name, output_dir):
    """Create a 3-way subplot for test results: actual vs predicted, residuals over time, and residual KDE."""
    # Compute residuals for each model
    residuals = {}
    for key, pred in predictions.items():
        residuals[key] = actual - pred

    # Create a DataFrame for KDE plotting
    residuals_df = pd.DataFrame(residuals)
    melted_res = residuals_df.melt(var_name='Model', value_name='Residuals')
    model_colours = {'Control': 'blue', 'Vader': 'green', 'BERT': 'red'}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Actual vs Predicted (Testing)
    ax1 = axes[0]
    ax1.plot(test_dates, actual, label='Actual Price', color='black')
    ax1.plot(test_dates, predictions.get('Control', []), label='Control Model', color='blue')
    ax1.plot(test_dates, predictions.get('Vader', []), label='Vader Model', color='green')
    ax1.plot(test_dates, predictions.get('BERT', []), label='BERT Model', color='red')
    ax1.set_title(f'{stock_name} - Test Data - Actual vs Predicted')
    ax1.set_ylabel('Stock Price')
    ax1.grid(True)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=11)
    
    # Plot 2: Residuals over time (Testing)
    ax2 = axes[1]
    ax2.axhline(0, color='black', linestyle='-', label='Ideal Residual')
    ax2.plot(test_dates, residuals.get('Control', []), label='Residual Control', color='blue')
    ax2.plot(test_dates, residuals.get('Vader', []), label='Residual Vader', color='green')
    ax2.plot(test_dates, residuals.get('BERT', []), label='Residual BERT', color='red')
    ax2.set_title(f'{stock_name} - Test Data - Residuals Over Time')
    ax2.set_ylabel('Residual (Actual - Predicted)')
    ax2.grid(True)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=11)
    
    # Plot 3: KDE of Residuals (Testing)
    ax3 = axes[2]
    sns.kdeplot(
        data=melted_res,
        x='Residuals',
        hue='Model',
        fill=True,
        alpha=0.1,
        palette=model_colours,
        ax=ax3
    )
    for model, color in model_colours.items():
        mean_res = residuals_df[model].mean()
        ax3.axvline(mean_res, color=color, linestyle='--', label=f'{model} (Mean: {mean_res:.4f})')
    ax3.set_title(f'{stock_name} - Test Data - Residual Distribution')
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Density')
    ax3.grid(True)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=11)
    
    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'residuals_{stock_name}.pdf'), format='pdf')
    plt.show()

def train_and_evaluate_model(stock_name, sentiment_data, max_epochs=20, patience=10,
                             train_proportion=0.7, lookback=30, forecast=1,
                             control_columns=['Close', 'Volume'],
                             sentiment_columns={'Vader': ['vader_negative', 'vader_neutral', 'vader_positive', 'vader_compound'],
                                                'BERT': ['bert_sentiment_score']},
                             output_dir='results'):
    """Train and evaluate models using different dataset variants."""
    data_variants = prepare_data(stock_name, sentiment_data, control_columns, sentiment_columns)
    
    performance = []
    loss_dict = {}
    pred_dict = {}
    control_Y_true = None
    test_dates = None

    # Process each dataset variant
    for key, dataset in data_variants.items():
        X_train, Y_train, X_test, Y_test, scaler = train_test_split_data(dataset, train_proportion, lookback, forecast)
        input_shape = (X_train.shape[1], X_train.shape[2])
        model, history = build_and_train_model(X_train, Y_train, input_shape, max_epochs, patience)
        loss_dict[key] = history.history['loss']
        
        Y_true, Y_pred = make_predictions(model, X_test, Y_test)
        # For the Control variant, store test dates and actual values
        if key == 'Control':
            train_size = int(len(dataset) * train_proportion)
            test_data = dataset.iloc[train_size:]
            test_dates = test_data.index[lookback + forecast - 1:][:len(Y_test)]
            control_Y_true = Y_true
        
        mse, rmse, mae = calculate_metrics(Y_true, Y_pred)
        performance.append({'Model': key + '_' + stock_name, 'MSE': mse, 'RMSE': rmse, 'MAE': mae})
        pred_dict[key] = Y_pred

    plot_training_loss(loss_dict, stock_name, output_dir)
    plot_test_results(test_dates, control_Y_true, pred_dict, stock_name, output_dir)
    
    return pd.DataFrame(performance)

# example use
performance = train_and_evaluate_model("BHP", sentiment_data, max_epochs=20)