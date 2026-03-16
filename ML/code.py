import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from statsmodels.tsa.stattools import adfuller, acf, kpss
from matplotlib import pyplot
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PowerTransformer
#import xgboost as xgb
import concurrent.futures
from sklearn.multioutput import MultiOutputClassifier
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import time
from datetime import timedelta
# from multiprocessing import Process, Manager, cpu_count
import multiprocessing as mp
from functools import partial
from prophet import Prophet
from colorama import init as colorama_init
from colorama import Fore 
from colorama import Style
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from scipy.stats import entropy
from scipy.signal import find_peaks
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import pywt

start_total = time.time()

# data_diff = data_series.diff().dropna()
# result = adfuller(data_diff)
# print("ADF Statistic:", result[0])
# print("p-value:", result[1])
# if result[1] > 0.05:
#     print("Data is likely non-stationary; consider differencing or transformations.")



# Load the dataset and preprocess data
def preprocess_data(data):

    data = data.drop(['Volume', 'High', 'Low', 'Close', 'Name'], axis=1)

    window_size = 5
    std_dev_threshold = 3

    # Calculate rolling mean and standard deviation
    rolling_mean = data['Open'].rolling(window=window_size, center=True).mean()
    rolling_std = data['Open'].rolling(window=window_size, center=True).std()

    # Detect outliers
    outliers = (data['Open'] - rolling_mean).abs() > std_dev_threshold * rolling_std

    # Replace outliers with interpolated values
    data['value_corrected'] = data['Open'].copy()
    data.loc[outliers, 'value_corrected'] = np.nan  # Set outliers to NaN
    data['Open'] = data['value_corrected'].interpolate() 
    data = data.drop('value_corrected', axis=1)

    # Convert the date column to datetime and set it as the index
    data['Open'] = data['Open'].apply(lambda x: x if x > 1e-10 else 1e-10)
    data = data[data["Open"] > 0]
    data.plot()
    pyplot.show()
    print("NaN values:", data["Open"].isna().sum())
    print("Infinite values:", np.isinf(data["Open"]).sum())
    data["Open"] = np.log(data['Open'] + 1).diff().dropna()


    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data.asfreq('D', method="ffill")
    data = prepare_data(data)
    data.plot()
    pyplot.show()
    return data

def preprocess_series(series):
    pt = PowerTransformer(method='yeo-johnson')
    data_transformed = pt.fit_transform(series.values.reshape(-1, 1)).flatten()
    data_transformed_series = pd.Series(data_transformed, index=series.index)
    data_transformed_series = data_transformed_series.diff().dropna()
    return data_transformed_series

# result = adfuller(data_transformed_series)

def prepare_data(df):
    """
    Prepare data by selecting only numeric columns and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Cleaned DataFrame with only numeric columns
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    
    # Handle any missing values
    df_numeric = df_numeric.fillna(method='ffill').fillna(method='bfill')
    
    return df_numeric

def run_stationarity_tests(data, column_name='Open'):
    """
    Run ADF and KPSS tests on the time series data.
    
    Parameters:
    data (pd.Series or np.array): Time series data
    column_name (str): Name of the column being tested
    """
    # Ensure data is numeric and handle NaN values
    if isinstance(data, pd.DataFrame):
        data = data.select_dtypes(include=[np.number]).iloc[:, 0]
    data = pd.to_numeric(data, errors='coerce')
    data = data.dropna()
    
    # ADF Test
    adf_result = adfuller(data)
    print(f'\nAugmented Dickey-Fuller Test Results for {column_name}:')
    print(f'ADF Statistic: {adf_result[0]:.4f}')
    print(f'p-value: {adf_result[1]:.4f}')
    print('Critical values:')
    for key, value in adf_result[4].items():
        print(f'\t{key}: {value:.4f}')
    
    # KPSS Test
    kpss_result = kpss(data)
    print(f'\nKPSS Test Results for {column_name}:')
    print(f'KPSS Statistic: {kpss_result[0]:.4f}')
    print(f'p-value: {kpss_result[1]:.4f}')
    print('Critical values:')
    for key, value in kpss_result[3].items():
        print(f'\t{key}: {value:.4f}')



# Function to create rolling windows from the time series
def create_dynamic_windows(series, min_window=50, max_window=200, step=25, volatility_threshold=0.1):
    """
    Create dynamic windows based on local volatility.
    
    Parameters:
    - series: Time series data (pd.Series).
    - min_window: Minimum window size.
    - max_window: Maximum window size.
    - step: Step size for moving the window.
    - volatility_threshold: Threshold for determining high volatility.
    
    Returns:
    - windows: List of windows (pd.Series).
    """
    windows = []
    i = 0
    while i < len(series) - min_window + 1:
        # Calculate volatility in the current region
        current_volatility = series.iloc[i:i + min_window].std()
        
        # Adjust window size based on volatility
        if current_volatility > volatility_threshold:
            window_size = min_window  # Smaller window for high volatility
        else:
            window_size = max_window  # Larger window for low volatility
        
        # Ensure the window does not exceed the series length
        window_size = min(window_size, len(series) - i)
        
        # Extract the window
        window = series.iloc[i:i + window_size]
        windows.append(window)
        
        # Move the window by the step size
        i += step
    
    return windows

# Feature extraction function
def extract_features(window):
    """
    Enhanced feature extraction with additional time series characteristics
    """
    features = {}
    
    # Statistical features
    features['mean'] = np.mean(window)
    features['std'] = np.std(window)
    features['max'] = np.max(window)
    features['min'] = np.min(window)
    features['range'] = features['max'] - features['min']
    features['skewness'] = skew(window)
    features['kurtosis'] = kurtosis(window)


    features['first_diff'] = window.iloc[-1] - window.iloc[-2] if len(window) > 1 else 0
    features['rate_of_change'] = (window.iloc[-1] - window.iloc[0]) / window.iloc[0] if window.iloc[0] != 0 else 0
    
    # Rolling statistics
    features['rolling_mean_7'] = window.rolling(window=7).mean().iloc[-1]
    features['rolling_std_7'] = window.rolling(window=7).std().iloc[-1]
    features['rolling_mean_30'] = window.rolling(window=30).mean().iloc[-1]
    features['rolling_std_30'] = window.rolling(window=30).std().iloc[-1]
    
    # Momentum and volatility
    features['momentum_5'] = window.iloc[-1] - window.iloc[-5] if len(window) >= 5 else 0
    features['momentum_10'] = window.iloc[-1] - window.iloc[-10] if len(window) >= 10 else 0
    features['volatility_7'] = window.rolling(window=7).std().mean() if len(window) >= 7 else 0
    features['volatility_30'] = window.rolling(window=30).std().mean() if len(window) >= 30 else 0
    
    x = np.arange(len(window))
    slope = np.polyfit(x, window, 1)[0]
    features['slope'] = slope

    # Trend features
    x = np.arange(len(window))
    poly_fit = np.polyfit(x, window, 2)
    features['trend_quadratic'] = poly_fit[0]
    features['trend_linear'] = poly_fit[1]
    
    # Stationarity features
    adf_result = adfuller(window)
    features['adf_stat'] = adf_result[0]
    features['adf_pvalue'] = adf_result[1]

    fft_values = np.abs(fft(window))
    features['fft_1'] = fft_values[1]  # First harmonic
    features['fft_2'] = fft_values[2]  # Second harmonic

    features['entropy'] = entropy(window.value_counts(normalize=True))

    peaks, _ = find_peaks(window)
    features['num_peaks'] = len(peaks)
    features['avg_peak_height'] = np.mean(window.iloc[peaks]) if len(peaks) > 0 else 0
        
    coeffs = pywt.wavedec(window, 'db1', level=2)
    features['wavelet_energy_level_1'] = np.sum(np.square(coeffs[0]))
    features['wavelet_energy_level_2'] = np.sum(np.square(coeffs[1]))

    features['lag_1'] = window.iloc[-2] if len(window) >= 2 else 0
    features['lag_2'] = window.iloc[-3] if len(window) >= 3 else 0
    features['lag_3'] = window.iloc[-4] if len(window) >= 4 else 0

    # Autocorrelation features
    try:
        acf_values = acf(window, nlags=5)
        for i in range(1, min(6, len(acf_values))):
            features[f'acf_lag_{i}'] = acf_values[i]
    except:
        for i in range(1, 6):
            features[f'acf_lag_{i}'] = 0

    # Exponential moving averages
    features['ema_7'] = window.ewm(span=7, adjust=False).mean().iloc[-1]
    features['ema_30'] = window.ewm(span=30, adjust=False).mean().iloc[-1]

    # Seasonal decomposition
    decomposition = seasonal_decompose(window, period=7)
    features['seasonal_strength'] = np.std(decomposition.seasonal)
    features['residual_std'] = np.std(decomposition.resid)

    # Seasonality indicator (basic periodicity detection)
    features['seasonal_strength'] = acf(window, nlags=12)[12] if len(window) > 12 else 0

    # Extract date-based features
    if isinstance(window.index, pd.DatetimeIndex):
        features['month'] = window.index[-1].month
        features['day_of_week'] = window.index[-1].dayofweek
            
    # Return distances
    features['last_value'] = window.iloc[-1]
    features['second_last_value'] = window.iloc[-2] if len(window) >= 2 else window.iloc[-1]
    features['third_last_value'] = window.iloc[-3] if len(window) >= 3 else window.iloc[-1]
    
    return features

# Apply forecasting models and get the best model based on a simple error measure (e.g., RMSE)
# For simplicity, using ARIMA and HWES as examples of models; more can be added


def evaluate_models(window):
    errors = {}
    predictions = {}
    
    # ARIMA with optimized parameters
    try:
        model_arima = ARIMA(window, order=(1, 1, 1))
        model_arima_fit = model_arima.fit()
        pred_arima = model_arima_fit.forecast(steps=1) 
        error_arima = np.sqrt(mean_squared_error(window[-1:], pred_arima))
        errors['ARIMA'] = error_arima
        print(f"{Fore.GREEN}ARIMA Done{Style.RESET_ALL}")
    except:
        errors['ARIMA'] = float('inf')
        print(f"{Fore.RED}ERROR HAPPENED AT ARIMA{Style.RESET_ALL}")
    # HWES with optimized parameters
    try:
        model_hwes = ExponentialSmoothing(window, 
                                         seasonal='add', 
                                         seasonal_periods=7)
        model_hwes_fit = model_hwes.fit(optimized=True)
        pred_hwes = model_hwes_fit.forecast(1)
        error_hwes = np.sqrt(mean_squared_error(window[-1:], [pred_hwes]))
        errors['HWES'] = error_hwes
        predictions['HWES'] = pred_hwes
        print(f"{Fore.GREEN}HWES Done{Style.RESET_ALL}")
    except:
        errors['HWES'] = float('inf')
        predictions['HWES'] = None
        print(f"{Fore.RED}ERROR HAPPENED AT HWES{Style.RESET_ALL}")
    
    # Prophet with enhanced configuration
    try:
        prophet_df = pd.DataFrame({'ds': window.index, 'y': window.values})
        prophet_model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode='multiplicative',
            daily_seasonality=True
        )
        prophet_model.fit(prophet_df)
        future = prophet_model.make_future_dataframe(periods=1)
        forecast = prophet_model.predict(future)
        pred_prophet = forecast['yhat'].iloc[-1]
        error_prophet = np.sqrt(mean_squared_error(window[-1:], [pred_prophet]))
        errors['Prophet'] = error_prophet
        predictions['Prophet'] = pred_prophet
        print(f"{Fore.GREEN}PROPHET Done{Style.RESET_ALL}")
    except:
        errors['Prophet'] = float('inf')
        predictions['Prophet'] = None
        print(f"{Fore.RED}ERROR HAPPENED AT PROPHET{Style.RESET_ALL}")
    
    # Enhanced model selection logic
    valid_models = {k: v for k, v in errors.items() if v != float('inf')}
    if not valid_models:
        return 'ARIMA'  # Default to ARIMA if all models fail
    
    # Weight recent performance more heavily
    best_model = min(valid_models.items(), key=lambda x: x[1])[0]
    return best_model

def process_window(window):
    features = extract_features(window)
    best_model = evaluate_models(window)
    return features, best_model

# def process_dataset(windows, shared_dict, dataset_id):
#     """Same as before - processes single dataset"""
#     training_data, labels = [], []
#     for window in windows:
#         features, best_model = process_window(window)
#         training_data.append(features)
#         labels.append(best_model)
    
#     shared_dict[f'training_data{dataset_id}'] = training_data
#     shared_dict[f'labels{dataset_id}'] = labels

# def parallel_process_datasets(windows_list):
#     num_cpus = cpu_count()
#     max_processes = max(1, num_cpus - 1)  # Leave one CPU free for system
#     num_datasets = len(windows_list)
    
#     print(f"System has {num_cpus} CPUs. Using {max_processes} for processing.")
#     print(f"Processing {num_datasets} datasets in batches...")
    
#     manager = Manager()
#     shared_dict = manager.dict()
    
#     for batch_start in range(0, num_datasets, max_processes):
#         batch_end = min(batch_start + max_processes, num_datasets)
#         current_batch = windows_list[batch_start:batch_end]
        
#         print(f"Processing datasets {batch_start + 1} to {batch_end}")
        
#         processes = []
#         for i, windows in enumerate(current_batch, batch_start + 1):
#             p = Process(
#                 target=process_dataset,
#                 args=(windows, shared_dict, i)
#             )
#             processes.append(p)
        
#         for p in processes:
#             p.start()
#         for p in processes:
#             p.join()
    
#     training_data_list = []
#     labels_list = []
#     for i in range(1, num_datasets + 1):
#         training_data_list.append(shared_dict[f'training_data{i}'])
#         labels_list.append(shared_dict[f'labels{i}'])
    
#     return training_data_list, labels_list

def process_single_window(window_tuple):
    """
    Process a single window and return features and best model.
    Args:
        window_tuple: Tuple of (dataset_idx, window)
    """
    _, window = window_tuple  # Unpack the tuple
    features, best_model = process_window(window)
    return features, best_model

def flatten_windows(window_lists):
    """Flatten all windows into a single list with their dataset indices."""
    flat_windows = []
    for dataset_idx, windows in enumerate(window_lists):
        for window in windows:
            flat_windows.append((dataset_idx, window))
    return flat_windows



preprocess_start = time.time()

data1 = pd.read_csv("CSCO.csv")
data1 = preprocess_data(data1)  
data_series1 = data1['Open']  # Adjust as necessary
data_series1 = preprocess_series(data_series1)  # Apply transformations and differencing

# Load and preprocess Dataset 2
data2 = pd.read_csv("AXP.csv")
data2 = preprocess_data(data2)  # Apply same preprocessing steps
data_series2 = data2['Open']  # Adjust as necessary
data_series2 = preprocess_series(data_series2)

# Load and preprocess Dataset 3
data3 = pd.read_csv("GOOGLE.csv")
data3 = preprocess_data(data3)  # Assumes prepare_data does necessary preprocessing steps
data_series3 = data3['Open']  # Adjust as necessary
data_series3 = preprocess_series(data_series3)  # Apply transformations and differencing

# Load and preprocess Dataset 4
data4 = pd.read_csv("IBM.csv")
data4 = preprocess_data(data4)  # Apply same preprocessing steps
data_series4 = data4['Open']  # Adjust as necessary
data_series4 = preprocess_series(data_series4)

# Load and preprocess Dataset 5
data5 = pd.read_csv("MCD.csv")
data5 = preprocess_data(data5)  # Apply same preprocessing steps
data_series5 = data5['Open']  # Adjust as necessary
data_series5 = preprocess_series(data_series5)

# Load and preprocess Dataset 6
data6 = pd.read_csv("CAT.csv")
data6 = preprocess_data(data6)  # Apply same preprocessing steps
data_series6 = data6['Open']  # Adjust as necessary
data_series6 = preprocess_series(data_series6)

# Load and preprocess Dataset 7
data7 = pd.read_csv("BA.csv")
data7 = preprocess_data(data7)  # Apply same preprocessing steps
data_series7 = data7['Open']  # Adjust as necessary
data_series7 = preprocess_series(data_series7)

# Load and preprocess Dataset 8
data8 = pd.read_csv("AMZN.csv")
data8 = preprocess_data(data8)  # Apply same preprocessing steps
data_series8 = data8['Open']  # Adjust as necessary
data_series8 = preprocess_series(data_series8)

#Generate features and labels for each window
data9 = pd.read_csv("NKE.csv")
data9 = preprocess_data(data9)  # Apply same preprocessing steps
data_series9 = data9['Open']  # Adjust as necessary
data_series9 = preprocess_series(data_series9)

data10 = pd.read_csv("JPM.csv")
data10 = preprocess_data(data10)  # Apply same preprocessing steps
data_series10 = data10['Open']  # Adjust as necessary
data_series10 = preprocess_series(data_series10)

preprocess_end = time.time()
elapsed_time_preprocess = preprocess_end - preprocess_start
formatted_prepro_time = str(timedelta(seconds=elapsed_time_preprocess))


#Generate features and labels for each window
feat_gen_start = time.time()

windows1 = create_dynamic_windows(data_series1)
windows2 = create_dynamic_windows(data_series2)
windows3 = create_dynamic_windows(data_series3)
windows4 = create_dynamic_windows(data_series4)
windows5 = create_dynamic_windows(data_series5)
windows6 = create_dynamic_windows(data_series6)
windows7 = create_dynamic_windows(data_series7)
windows8 = create_dynamic_windows(data_series8)
windows9 = create_dynamic_windows(data_series9)
windows10 = create_dynamic_windows(data_series10)

feat_gen_end = time.time()
elapsed_time_feat_gen = feat_gen_end - feat_gen_start
formatted_feat_gen_time = str(timedelta(seconds=elapsed_time_feat_gen))

# Feature extraction for all datasets
feat_extr_start = time.time()

training_data1, labels1 = [], []
for window in windows1:
    features, best_model = process_window(window)
    training_data1.append(features)
    labels1.append(best_model)

training_data2, labels2 = [], []
for window in windows2:
    features, best_model = process_window(window)
    training_data2.append(features)
    labels2.append(best_model)

training_data3, labels3 = [], []
for window in windows3:
    features, best_model = process_window(window)
    training_data3.append(features)
    labels3.append(best_model)

training_data4, labels4 = [], []
for window in windows4:
    features, best_model = process_window(window)
    training_data4.append(features)
    labels4.append(best_model)

training_data5, labels5 = [], []
for window in windows5:
    features, best_model = process_window(window)
    training_data5.append(features)
    labels5.append(best_model)

training_data6, labels6 = [], []
for window in windows6:
    features, best_model = process_window(window)
    training_data6.append(features)
    labels6.append(best_model)

    
training_data7, labels7 = [], []
for window in windows7:
    features, best_model = process_window(window)
    training_data7.append(features)
    labels7.append(best_model)

training_data8, labels8 = [], []
for window in windows8:
    features, best_model = process_window(window)
    training_data8.append(features)
    labels8.append(best_model)
    
training_data9, labels9 = [], []
for window in windows1:
    features, best_model = process_window(window)
    training_data9.append(features)
    labels9.append(best_model)
    
training_data10, labels10 = [], []
for window in windows10:
    features, best_model = process_window(window)
    training_data10.append(features)
    labels10.append(best_model)

# windows_list = [
#     windows1, windows2, windows3, windows4, windows5,
#     windows6, windows7, windows8, windows9, windows10
# ]

# # Process all datasets
# training_data_list, labels_list = parallel_process_datasets(windows_list)

# # Unpack results if needed
# (training_data1, training_data2, training_data3, training_data4, training_data5,
#     training_data6, training_data7, training_data8, training_data9, training_data10) = training_data_list

# (labels1, labels2, labels3, labels4, labels5,
#     labels6, labels7, labels8, labels9, labels10) = labels_list

# all_windows = [windows1, windows2, windows3, windows4, windows5,
#                windows6, windows7, windows8, windows9, windows10]

# # Flatten all windows into a single list
# flat_windows = flatten_windows(all_windows)

# # Process all windows in parallel
# with mp.Pool(processes=mp.cpu_count()) as pool:
#     results = pool.map(process_single_window, flat_windows)



feat_extr_end = time.time()
elapsed_time_feat_extr = feat_extr_end - feat_extr_start
formatted_feat_extr_time = str(timedelta(seconds=elapsed_time_feat_extr))

# data_log = np.log(data_series)
# data_log_diff = data_log.diff().dropna()

X = pd.DataFrame(training_data1 + training_data2 + training_data3 + training_data4 + training_data5 + training_data6 + training_data7 + training_data8 + training_data9 + training_data10)
y = labels1 + labels2 + labels3 + labels4 + labels5 + labels6 + labels7 + labels8 + labels9 + labels10

# all_training_data = []
# all_labels = []
# for training_data, labels in all_results:
#     all_training_data.extend(training_data)
#     all_labels.extend(labels)

# # Create the final DataFrame
# X = pd.DataFrame(all_training_data)

# # If you need the labels separately:
# y = pd.Series(all_labels)
# Convert results into training data and labels

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



print(f"{Fore.CYAN}BayesSearch Parameter tuning in progress{Style.RESET_ALL}")

# Initialize classifier
rf_classifier = RandomForestClassifier(random_state=42)
gb_classifier = GradientBoostingClassifier(random_state=42)

rf_param_space = {
    'n_estimators': Integer(100, 1000),
    'max_depth': Integer(10, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Categorical(['sqrt', 'log2']),
    'class_weight': Categorical(['balanced', None])
}

# Define the parameter space for Gradient Boosting
gb_param_space = {
    'n_estimators': Integer(100, 500),
    'learning_rate': Real(0.01, 0.2),
    'max_depth': Integer(3, 10),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 4)
}

rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

class_weights = class_weight.compute_sample_weight('balanced', y_train)

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

rf_bayes_search = BayesSearchCV(
    estimator=rf_classifier,
    search_spaces=rf_param_space,
    n_iter=50,  # Number of iterations for Bayesian optimization
    cv=TimeSeriesSplit(n_splits=3),  # Use time-series cross-validation
    scoring='accuracy',  # Metric to optimize
    n_jobs=-1,  # Use all available CPU cores
    random_state=42
)

# Set up BayesSearchCV for Gradient Boosting
gb_bayes_search = BayesSearchCV(
    estimator=gb_classifier,
    search_spaces=gb_param_space,
    n_iter=50,  # Number of iterations for Bayesian optimization
    cv=TimeSeriesSplit(n_splits=3),  # Use time-series cross-validation
    scoring='accuracy',  # Metric to optimize
    n_jobs=-1,  # Use all available CPU cores
    random_state=42
)

# Fit BayesSearchCV for Random Forest
print("Optimizing Random Forest...")
rf_bayes_search.fit(X_train, y_train)
print("Best parameters for Random Forest:", rf_bayes_search.best_params_)
print("Best cross-validation score for Random Forest:", rf_bayes_search.best_score_)

# Fit BayesSearchCV for Gradient Boosting
print("Optimizing Gradient Boosting...")
gb_bayes_search.fit(X_train, y_train)
print("Best parameters for Gradient Boosting:", gb_bayes_search.best_params_)
print("Best cross-validation score for Gradient Boosting:", gb_bayes_search.best_score_)

# Get the best models
best_rf = rf_bayes_search.best_estimator_
best_gb = gb_bayes_search.best_estimator_


# Create the VotingClassifier with the best models
classifier = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('gb', best_gb)
    ],
    voting='soft',  # Use soft voting for probability-based predictions
    n_jobs=-1
) 

# Train the ensemble
print(f"{Fore.CYAN}Training Voting Classifier")
classifier.fit(X_train, y_train)

# Evaluate model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=1)
cross_val_scores = cross_val_score(classifier, X, y, cv=5)

tscv = TimeSeriesSplit(n_splits=5)
cross_val_scores2 = cross_val_score(classifier, X, y, cv=tscv)

print("Accuracy on test set:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
print("Cross Validation Score: ",cross_val_scores)
print("Cross Validation Score 2: ",cross_val_scores2)

end_total = time.time()
formatted_total_time = str(timedelta(seconds=end_total-start_total))
print("Preprocess time: ", formatted_prepro_time)
print("Feature Generation time: ", formatted_feat_gen_time)
print("Feature extraction time: ", formatted_feat_extr_time)
print("Total Execution Time: ", formatted_total_time)