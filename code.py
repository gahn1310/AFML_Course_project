import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
from google.colab import drive
drive.mount('/content/drive')


!ls "/content/drive/MyDrive/AF_dataset/AF/af-classification-from-a-short-single-lead-ecg-recording-the-physionet-computing-in-cardiology-challenge-2017-1.0.0"

import os
import zipfile

# ✅ Use the correct base path (based on what you just showed)
BASE_PATH = "/content/drive/MyDrive/AF_dataset/AF/af-classification-from-a-short-single-lead-ecg-recording-the-physionet-computing-in-cardiology-challenge-2017-1.0.0"

zip_path = os.path.join(BASE_PATH, "training2017.zip")
extract_path = os.path.join(BASE_PATH, "training2017")

# ✅ Extract only if not already extracted
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(BASE_PATH)
    print("✅ Extracted training2017.zip successfully!")
else:
    print("✅ training2017 folder already exists.")

!ls "/content/drive/MyDrive/AF_dataset/AF/af-classification-from-a-short-single-lead-ecg-recording-the-physionet-computing-in-cardiology-challenge-2017-1.0.0/training2017" | head

# Install necessary libraries for ECG signal processing
!pip install wfdb scipy scikit-learn -q

print("✅ Libraries installed successfully!")
print("Importing libraries...")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("✅ All imports successful!")
# Load the REFERENCE.csv file to get labels for all recordings
reference_path = os.path.join(extract_path, "REFERENCE.csv")
labels_df = pd.read_csv(reference_path, header=None, names=['filename', 'label'])

print(f"✅ Total recordings: {len(labels_df)}")
print(f"\n📊 Class Distribution:")
print(labels_df['label'].value_counts())
print(f"\n🔍 First 5 entries:")
print(labels_df.head())

# Map labels to full names for clarity
label_map = {'N': 'Normal', 'A': 'AF', 'O': 'Other', '~': 'Noisy'}
labels_df['label_name'] = labels_df['label'].map(label_map)

print(f"\n✅ Labels loaded and mapped successfully!")
# Load one sample ECG to understand the data structure
sample_file = labels_df.iloc[0]['filename']  # A00001 (Normal)
sample_path = os.path.join(extract_path, sample_file)

# Read the ECG signal using wfdb
record = wfdb.rdrecord(sample_path)

print(f"📁 Sample File: {sample_file}")
print(f"🏷️  Label: {labels_df.iloc[0]['label']} ({labels_df.iloc[0]['label_name']})")
print(f"\n📊 Signal Info:")
print(f"  - Sampling Frequency: {record.fs} Hz")
print(f"  - Duration: {len(record.p_signal) / record.fs:.2f} seconds")
print(f"  - Signal Shape: {record.p_signal.shape}")
print(f"  - Number of samples: {len(record.p_signal)}")

# Plot the ECG signal
plt.figure(figsize=(15, 4))
plt.plot(record.p_signal, linewidth=0.5)
plt.title(f'ECG Signal - {sample_file} (Label: {labels_df.iloc[0]["label_name"]})')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude (mV)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✅ Sample ECG loaded and visualized!")
# Function to detect R-peaks using bandpass filtering and peak detection
def detect_r_peaks(ecg_signal, fs=300):
    """
    Detect R-peaks in ECG signal using bandpass filter and peak finding
    Args:
        ecg_signal: 1D numpy array of ECG signal
        fs: sampling frequency (default 300 Hz)
    Returns:
        r_peaks: indices of detected R-peaks
    """
    # Bandpass filter (5-15 Hz) to isolate QRS complexes
    nyquist = fs / 2
    low = 5 / nyquist
    high = 15 / nyquist
    b, a = butter(3, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, ecg_signal)

    # Square the signal to emphasize peaks
    squared_signal = filtered_signal ** 2

    # Find peaks with minimum distance between peaks (200ms = 0.2s * 300Hz = 60 samples)
    min_distance = int(0.2 * fs)
    r_peaks, _ = find_peaks(squared_signal, distance=min_distance, height=np.mean(squared_signal))

    return r_peaks

# Test R-peak detection on the sample ECG
sample_signal = record.p_signal.flatten()
r_peaks = detect_r_peaks(sample_signal, fs=record.fs)

print(f"✅ Detected {len(r_peaks)} R-peaks")
print(f"📊 Average Heart Rate: {len(r_peaks) / (len(sample_signal) / record.fs) * 60:.2f} BPM")

# Visualize R-peaks on ECG
plt.figure(figsize=(15, 4))
plt.plot(sample_signal, linewidth=0.5, label='ECG Signal')
plt.plot(r_peaks, sample_signal[r_peaks], 'ro', markersize=8, label='R-peaks')
plt.title(f'R-Peak Detection - {sample_file}')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✅ R-peak detection successful!")
def create_windows_from_rr_intervals(ecg_signal, r_peaks, fs=300, max_threshold_ms=500):
    """
    Create windows using midpoints of R-R intervals as start/end points
    Exactly as described in the paper (Figure 1)

    Args:
        ecg_signal: 1D ECG signal
        r_peaks: indices of R-peaks
        fs: sampling frequency
        max_threshold_ms: max threshold in milliseconds (default 500ms = 0.5s)
    Returns:
        windows: list of ECG windows
        window_indices: list of (start, end) index pairs
    """
    windows = []
    window_indices = []
    max_samples = int((max_threshold_ms / 1000) * fs)  # Convert ms to samples

    for i in range(len(r_peaks) - 1):
        # Calculate midpoint between current and next R-peak
        rr_interval = r_peaks[i+1] - r_peaks[i]
        midpoint = r_peaks[i] + (rr_interval // 2)

        # Starting point: midpoint of previous R-R interval (or threshold before R-peak)
        if i == 0:
            start = max(0, r_peaks[i] - max_samples)
        else:
            prev_midpoint = r_peaks[i-1] + ((r_peaks[i] - r_peaks[i-1]) // 2)
            start = prev_midpoint

        # Ending point: current midpoint
        end = midpoint

        # Apply threshold if distance too large
        if end - start > 2 * max_samples:
            start = max(start, r_peaks[i] - max_samples)
            end = min(end, r_peaks[i] + max_samples)

        # Extract window
        window = ecg_signal[start:end]
        if len(window) > 0:
            windows.append(window)
            window_indices.append((start, end))

    return windows, window_indices

# Test window creation on sample signal
windows, window_indices = create_windows_from_rr_intervals(sample_signal, r_peaks, fs=record.fs)

print(f"✅ Created {len(windows)} windows from {len(r_peaks)} R-peaks")
print(f"📊 Window length statistics:")
print(f"   - Min: {min([len(w) for w in windows])} samples")
print(f"   - Max: {max([len(w) for w in windows])} samples")
print(f"   - Mean: {np.mean([len(w) for w in windows]):.2f} samples")

# Visualize the first 3 windows
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
for idx in range(min(3, len(windows))):
    axes[idx].plot(windows[idx], linewidth=1)
    axes[idx].set_title(f'Window {idx+1} (Length: {len(windows[idx])} samples)')
    axes[idx].set_xlabel('Sample')
    axes[idx].set_ylabel('Amplitude (mV)')
    axes[idx].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✅ Window creation successful!")
def calculate_heart_rate(r_peaks, fs=300):
    """
    Calculate instantaneous heart rate from R-R intervals
    Returns heart rate for each R-R interval in BPM
    """
    if len(r_peaks) < 2:
        return np.array([])

    # Calculate R-R intervals in samples
    rr_intervals = np.diff(r_peaks)

    # Convert to seconds
    rr_intervals_sec = rr_intervals / fs

    # Calculate heart rate in BPM (beats per minute)
    heart_rate = 60 / rr_intervals_sec

    return heart_rate

# Test heart rate calculation on sample
heart_rate = calculate_heart_rate(r_peaks, fs=record.fs)

print(f"✅ Calculated heart rate for {len(heart_rate)} intervals")
print(f"📊 Heart Rate Statistics:")
print(f"   - Min: {heart_rate.min():.2f} BPM")
print(f"   - Max: {heart_rate.max():.2f} BPM")
print(f"   - Mean: {heart_rate.mean():.2f} BPM")
print(f"   - Std: {heart_rate.std():.2f} BPM")

# Visualize heart rate over time
plt.figure(figsize=(15, 4))
plt.plot(heart_rate, marker='o', linestyle='-', markersize=4, linewidth=1)
plt.axhline(y=heart_rate.mean(), color='r', linestyle='--', label=f'Mean: {heart_rate.mean():.2f} BPM')
plt.title('Instantaneous Heart Rate Over Time')
plt.xlabel('R-R Interval Index')
plt.ylabel('Heart Rate (BPM)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("\n✅ Heart rate calculation successful!")
def normalize_heart_rate(heart_rate):
    """
    Normalize heart rate to [0, 1] range as mentioned in paper
    """
    if len(heart_rate) == 0:
        return heart_rate

    # Normalize to [0, 1]
    hr_min = heart_rate.min()
    hr_max = heart_rate.max()

    if hr_max - hr_min > 0:
        normalized_hr = (heart_rate - hr_min) / (hr_max - hr_min)
    else:
        normalized_hr = np.zeros_like(heart_rate)

    return normalized_hr

# Test normalization
normalized_hr = normalize_heart_rate(heart_rate)

print(f"✅ Heart rate normalized to [0, 1] range")
print(f"📊 Normalized HR Statistics:")
print(f"   - Min: {normalized_hr.min():.4f}")
print(f"   - Max: {normalized_hr.max():.4f}")
print(f"   - Mean: {normalized_hr.mean():.4f}")

# Visualize before and after normalization
fig, axes = plt.subplots(1, 2, figsize=(15, 4))

axes[0].plot(heart_rate, marker='o', markersize=3)
axes[0].set_title('Original Heart Rate')
axes[0].set_xlabel('R-R Interval Index')
axes[0].set_ylabel('Heart Rate (BPM)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(normalized_hr, marker='o', markersize=3, color='green')
axes[1].set_title('Normalized Heart Rate [0, 1]')
axes[1].set_xlabel('R-R Interval Index')
axes[1].set_ylabel('Normalized Value')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Heart rate normalization successful!")
def preprocess_single_record(filename, base_path, fs=300):
    """
    Complete preprocessing pipeline for a single ECG recording
    Returns: (windows, normalized_hr, label)
    """
    try:
        # Load ECG signal
        record_path = os.path.join(base_path, filename)
        record = wfdb.rdrecord(record_path)
        ecg_signal = record.p_signal.flatten()

        # Detect R-peaks
        r_peaks = detect_r_peaks(ecg_signal, fs=fs)

        if len(r_peaks) < 2:
            return None, None, None

        # Create windows from R-R intervals
        windows, _ = create_windows_from_rr_intervals(ecg_signal, r_peaks, fs=fs)

        # Calculate and normalize heart rate
        heart_rate = calculate_heart_rate(r_peaks, fs=fs)
        normalized_hr = normalize_heart_rate(heart_rate)

        return windows, normalized_hr, filename

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None, None, None

# Test the complete pipeline on a sample
test_filename = labels_df.iloc[0]['filename']
test_windows, test_hr, test_name = preprocess_single_record(test_filename, extract_path)

print(f"✅ Complete preprocessing pipeline test:")
print(f"   - Filename: {test_name}")
print(f"   - Number of windows: {len(test_windows)}")
print(f"   - Number of HR values: {len(test_hr)}")
print(f"   - Window lengths: min={min([len(w) for w in test_windows])}, max={max([len(w) for w in test_windows])}")
print(f"   - HR range: [{test_hr.min():.4f}, {test_hr.max():.4f}]")

print("\n✅ Single record preprocessing pipeline ready!")
from tqdm import tqdm  # For progress bar

def preprocess_all_records(labels_df, base_path, fs=300):
    """
    Preprocess ALL recordings in the dataset
    Returns organized data for model training
    """
    all_data = []
    failed_records = []

    print("🔄 Starting batch preprocessing of all recordings...")
    print(f"Total records to process: {len(labels_df)}\n")

    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing"):
        filename = row['filename']
        label = row['label']

        windows, normalized_hr, _ = preprocess_single_record(filename, base_path, fs)

        if windows is not None and normalized_hr is not None:
            all_data.append({
                'filename': filename,
                'windows': windows,
                'heart_rate': normalized_hr,
                'label': label,
                'num_windows': len(windows)
            })
        else:
            failed_records.append(filename)

    print(f"\n✅ Preprocessing complete!")
    print(f"   - Successfully processed: {len(all_data)} records")
    print(f"   - Failed: {len(failed_records)} records")

    return all_data, failed_records

# WARNING: This will take 10-20 minutes depending on your system
# Process all 8,528 recordings
preprocessed_data, failed = preprocess_all_records(labels_df, extract_path)

print(f"\n📊 Preprocessing Summary:")
print(f"   - Total successful: {len(preprocessed_data)}")
print(f"   - Total failed: {len(failed)}")
if len(failed) > 0:
    print(f"   - Failed files: {failed[:5]}...")  # Show first 5
import pickle
import os

# Define save path in your Google Drive
save_dir = "/content/drive/MyDrive/AF_dataset/AF/af-classification-from-a-short-single-lead-ecg-recording-the-physionet-computing-in-cardiology-challenge-2017-1.0.0"
save_path = os.path.join(save_dir, "preprocessed_data.pkl")

# Save preprocessed data
print("💾 Saving preprocessed data to Google Drive...")
with open(save_path, 'wb') as f:
    pickle.dump({
        'preprocessed_data': preprocessed_data,
        'labels_df': labels_df,
        'total_records': len(preprocessed_data),
        'failed_records': failed
    }, f)

file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
print(f"✅ Preprocessed data saved successfully!")
print(f"   - Location: {save_path}")
print(f"   - File size: {file_size_mb:.2f} MB")
print(f"   - Total records: {len(preprocessed_data)}")
print("\n💡 You can now reload this data anytime without reprocessing!")
# Analyze the preprocessed data distribution
import pandas as pd

# Create summary statistics
summary_stats = {
    'label': [],
    'count': [],
    'total_windows': [],
    'avg_windows_per_record': []
}

for label in ['N', 'A', 'O', '~']:
    label_data = [d for d in preprocessed_data if d['label'] == label]
    count = len(label_data)
    total_windows = sum([d['num_windows'] for d in label_data])
    avg_windows = total_windows / count if count > 0 else 0

    summary_stats['label'].append(label)
    summary_stats['count'].append(count)
    summary_stats['total_windows'].append(total_windows)
    summary_stats['avg_windows_per_record'].append(round(avg_windows, 2))

stats_df = pd.DataFrame(summary_stats)
stats_df['label_name'] = stats_df['label'].map({'N': 'Normal', 'A': 'AF', 'O': 'Other', '~': 'Noisy'})

print("📊 Preprocessed Data Summary:\n")
print(stats_df.to_string(index=False))

# Visualize class distribution
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Records per class
axes[0].bar(stats_df['label_name'], stats_df['count'], color=['#2ecc71', '#e74c3c', '#3498db', '#95a5a6'])
axes[0].set_title('Number of Records per Class', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Records')
axes[0].grid(axis='y', alpha=0.3)

# Total windows per class
axes[1].bar(stats_df['label_name'], stats_df['total_windows'], color=['#2ecc71', '#e74c3c', '#3498db', '#95a5a6'])
axes[1].set_title('Total Windows per Class', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Number of Windows')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Data analysis complete!")
from tensorflow.keras.preprocessing.sequence import pad_sequences

def prepare_model_data(preprocessed_data, max_window_length=None, max_num_windows=None):
    """
    Pad all windows and heart rates to fixed lengths for model input
    """
    # Find maximum dimensions if not specified
    if max_window_length is None:
        max_window_length = max([max([len(w) for w in d['windows']]) for d in preprocessed_data])

    if max_num_windows is None:
        max_num_windows = max([len(d['windows']) for d in preprocessed_data])

    X_ecg = []
    X_hr = []
    y_labels = []
    filenames = []

    print(f"📏 Padding parameters:")
    print(f"   - Max window length: {max_window_length} samples")
    print(f"   - Max number of windows per record: {max_num_windows}")

    for record in preprocessed_data:
        # Pad windows (ECG data)
        padded_windows = pad_sequences(record['windows'],
                                       maxlen=max_window_length,
                                       dtype='float32',
                                       padding='post',
                                       truncating='post')

        # Pad number of windows to max_num_windows
        if len(padded_windows) < max_num_windows:
            padding = np.zeros((max_num_windows - len(padded_windows), max_window_length))
            padded_windows = np.vstack([padded_windows, padding])
        else:
            padded_windows = padded_windows[:max_num_windows]

        # Pad heart rate
        padded_hr = np.pad(record['heart_rate'],
                          (0, max_num_windows - len(record['heart_rate'])),
                          mode='constant',
                          constant_values=0)[:max_num_windows]

        X_ecg.append(padded_windows)
        X_hr.append(padded_hr)
        y_labels.append(record['label'])
        filenames.append(record['filename'])

    X_ecg = np.array(X_ecg)
    X_hr = np.array(X_hr)
    y_labels = np.array(y_labels)

    return X_ecg, X_hr, y_labels, filenames, max_window_length, max_num_windows

# Prepare data
X_ecg, X_hr, y_labels, filenames, max_win_len, max_num_win = prepare_model_data(preprocessed_data)

print(f"\n✅ Data prepared for model!")
print(f"📊 Final shapes:")
print(f"   - X_ecg shape: {X_ecg.shape} (records, windows, samples)")
print(f"   - X_hr shape: {X_hr.shape} (records, windows)")
print(f"   - y_labels shape: {y_labels.shape}")
print(f"\n💾 Memory usage: {(X_ecg.nbytes + X_hr.nbytes) / (1024**2):.2f} MB")
# Install Logical Neural Networks library
!pip install lnn -q

print("✅ LNN library installed!")

# Import required libraries for model building
from tensorflow import keras
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dropout, LSTM, Dense,
    Concatenate, Masking, Flatten, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform, he_normal
from sklearn.preprocessing import LabelEncoder

# Import LNN components
try:
    from lnn import LNN, Predicate, And, Or, Not, Implies
    print("✅ LNN components imported successfully!")
except ImportError:
    print("⚠️ LNN import failed - will use alternative approach")

print("\n✅ All model libraries ready!")
print(f"📦 TensorFlow version: {keras.__version__}")
import tensorflow as tf
from tensorflow.keras.layers import Layer

class LogicalNeuralLayer(Layer):
    """
    Custom Logical Neural Network layer for reasoning over temporal features
    Implements logical operations (AND, OR, NOT) on LSTM outputs
    """
    def __init__(self, units=64, **kwargs):
        super(LogicalNeuralLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Logical operation weights
        self.w_and = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='w_and'
        )
        self.w_or = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='w_or'
        )
        self.w_not = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='w_not'
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        super(LogicalNeuralLayer, self).build(input_shape)

    def call(self, inputs):
        # Logical AND operation (conjunction)
        and_output = tf.nn.sigmoid(tf.matmul(inputs, self.w_and))

        # Logical OR operation (disjunction)
        or_output = tf.nn.sigmoid(tf.matmul(inputs, self.w_or))

        # Logical NOT operation (negation)
        not_output = 1.0 - tf.nn.sigmoid(tf.matmul(inputs, self.w_not))

        # Combine logical operations
        logical_output = (and_output + or_output + not_output) / 3.0

        # Add bias and activation
        output = tf.nn.tanh(logical_output + self.bias)

        return output

    def get_config(self):
        config = super(LogicalNeuralLayer, self).get_config()
        config.update({"units": self.units})
        return config

# Test the custom LNN layer
print("✅ Custom Logical Neural Network layer created!")
print("🧠 Features:")
print("   - Implements AND, OR, NOT logical operations")
print("   - Trainable logical reasoning weights")
print("   - Integrates with Keras/TensorFlow seamlessly")
def build_crnn_lnn_model_fixed(max_num_windows, max_window_length):
    """
    Fixed CRNN + LNN model with proper shape handling
    """

    # ========== INPUT LAYERS ==========
    input_ecg = Input(shape=(max_num_windows, max_window_length), name='ecg_input')
    input_hr = Input(shape=(max_num_windows,), name='hr_input')

    # ========== ECG CNN BRANCH ==========
    print("🔨 Building ECG CNN branch...")
    # Flatten windows into single sequence
    x_ecg = Reshape((max_num_windows * max_window_length, 1))(input_ecg)

    # Conv Block 1
    x_ecg = Conv1D(32, 3, activation='relu', padding='same',
                   kernel_initializer=glorot_uniform())(x_ecg)
    x_ecg = MaxPooling1D(2)(x_ecg)
    x_ecg = Dropout(0.05)(x_ecg)

    # Conv Block 2
    x_ecg = Conv1D(64, 3, activation='relu', padding='same',
                   kernel_initializer=glorot_uniform())(x_ecg)
    x_ecg = MaxPooling1D(2)(x_ecg)
    x_ecg = Dropout(0.1)(x_ecg)

    # Conv Block 3
    x_ecg = Conv1D(128, 3, activation='relu', padding='same',
                   kernel_initializer=glorot_uniform())(x_ecg)
    x_ecg = MaxPooling1D(2)(x_ecg)
    x_ecg = Dropout(0.15)(x_ecg)

    print(f"   ECG branch output shape: {x_ecg.shape}")

    # ========== HEART RATE CNN BRANCH ==========
    print("🔨 Building Heart Rate CNN branch...")
    x_hr = Reshape((max_num_windows, 1))(input_hr)

    # Match the pooling to get same sequence length as ECG
    # ECG: 223*300 = 66900 -> /2/2/2 = 8362
    # HR: 223 -> need to upsample first to match

    # Upsample HR to match ECG length before pooling
    x_hr = layers.UpSampling1D(size=300)(x_hr)  # 223 -> 66900

    # Conv Block 1
    x_hr = Conv1D(32, 3, activation='relu', padding='same',
                  kernel_initializer=glorot_uniform())(x_hr)
    x_hr = MaxPooling1D(2)(x_hr)
    x_hr = Dropout(0.05)(x_hr)

    # Conv Block 2
    x_hr = Conv1D(64, 3, activation='relu', padding='same',
                  kernel_initializer=glorot_uniform())(x_hr)
    x_hr = MaxPooling1D(2)(x_hr)
    x_hr = Dropout(0.1)(x_hr)

    # Conv Block 3
    x_hr = Conv1D(128, 3, activation='relu', padding='same',
                  kernel_initializer=glorot_uniform())(x_hr)
    x_hr = MaxPooling1D(2)(x_hr)
    x_hr = Dropout(0.15)(x_hr)

    print(f"   HR branch output shape: {x_hr.shape}")

    # ========== MERGE BRANCHES ==========
    print("🔨 Merging branches...")
    merged = Concatenate(axis=-1)([x_ecg, x_hr])
    merged = Masking(mask_value=0.0)(merged)

    # ========== LSTM LAYERS ==========
    print("🔨 Adding LSTM layers...")
    x = LSTM(64, return_sequences=True, activation='tanh')(merged)
    x = LSTM(64, return_sequences=False, activation='tanh')(x)

    # ========== LNN LAYER ==========
    print("🔨 Adding Logical Neural Network layer...")
    x = LogicalNeuralLayer(units=64, name='logical_reasoning')(x)

    # ========== OUTPUT ==========
    output = Dense(2, activation='softmax', kernel_initializer=he_normal())(x)

    model = Model(inputs=[input_ecg, input_hr], outputs=output, name='CRNN_LNN_Model')
    return model

# Build the fixed model
print("🏗️ Building Fixed CRNN + LNN Model...\n")
model = build_crnn_lnn_model_fixed(max_num_win, max_win_len)

print("\n✅ Model built successfully!")
model.summary()
# ============================================================
# CELL: PREPARE FINAL DATA FOR HIERARCHICAL TRAINING
# ============================================================

print("="*70)
print("PREPARING FINAL DATA FOR TRAINING")
print("="*70)

# Check what variables we actually have from preprocessing
print("\n🔍 Checking available data variables...")

# Assuming your preprocessing created these variables:
# - ecg_windows: list of ECG windows for each recording
# - hr_sequences: list of heart rate sequences for each recording
# - y_labels: encoded labels (0=N, 1=AF, 2=O, 3=~)

# If you have different variable names, you need to adjust this section
# Let's create properly formatted arrays

# Check if we have the window data
try:
    print(f"✓ ecg_windows shape: {np.array(ecg_windows, dtype=object).shape}")
    print(f"✓ hr_sequences shape: {np.array(hr_sequences, dtype=object).shape}")
    print(f"✓ y_labels shape: {np.array(y_labels).shape}")

    # Pad ECG windows
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Find max dimensions
    max_num_win = max(len(windows) for windows in ecg_windows)
    max_win_len = max(max(len(win) for win in windows) if len(windows) > 0 else 0
                      for windows in ecg_windows)

    print(f"\n📊 Max number of windows: {max_num_win}")
    print(f"📊 Max window length: {max_win_len}")

    # Pad ECG sequences
    X_padded = []
    for windows in ecg_windows:
        # Pad each window to max_win_len
        padded_windows = pad_sequences(windows, maxlen=max_win_len,
                                       padding='post', truncating='post', dtype='float32')
        # Pad number of windows to max_num_win
        if len(padded_windows) < max_num_win:
            padding = np.zeros((max_num_win - len(padded_windows), max_win_len))
            padded_windows = np.vstack([padded_windows, padding])
        X_padded.append(padded_windows)

    X_padded = np.array(X_padded)

    # Pad HR sequences
    X_hr_padded = pad_sequences(hr_sequences, maxlen=max_num_win,
                                padding='post', truncating='post', dtype='float32')

    # Reshape for model input
    X_hr_padded = X_hr_padded.reshape(-1, max_num_win, 1)

    # Encode labels
    y_encoded = np.array(y_labels)

    print(f"\n✅ X_padded shape: {X_padded.shape}")
    print(f"✅ X_hr_padded shape: {X_hr_padded.shape}")
    print(f"✅ y_encoded shape: {y_encoded.shape}")

    print(f"\n📈 Class distribution:")
    unique, counts = np.unique(y_encoded, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"   Class {cls}: {cnt} samples")

except NameError as e:
    print(f"\n❌ ERROR: Variable not found - {e}")
    print("\n⚠️  You need to check your preprocessing cells for the actual variable names.")
    print("   Common variable names to look for:")
    print("   - ecg_windows, ecg_data, X_ecg, processed_ecg")
    print("   - hr_sequences, hr_data, X_hr, heart_rates")
    print("   - y_labels, labels, y, targets")
    print("\nPlease scroll up and find the cell that creates the windowed data.")
# ============================================================
# DEBUG: FIND ALL DATA VARIABLES
# ============================================================

print("="*70)
print("SEARCHING FOR DATA VARIABLES")
print("="*70)

# Get all variables in current scope
all_vars = dir()

print("\n🔍 Variables containing 'ecg' or 'signal':")
ecg_vars = [v for v in all_vars if ('ecg' in v.lower() or 'signal' in v.lower())
            and not v.startswith('_')]
for v in ecg_vars:
    try:
        var_type = type(eval(v)).__name__
        print(f"   {v} ({var_type})")
    except:
        pass

print("\n🔍 Variables containing 'hr' or 'heart' or 'rate':")
hr_vars = [v for v in all_vars if ('hr' in v.lower() or 'heart' in v.lower() or 'rate' in v.lower())
           and not v.startswith('_')]
for v in hr_vars:
    try:
        var_type = type(eval(v)).__name__
        print(f"   {v} ({var_type})")
    except:
        pass

print("\n🔍 Variables containing 'window':")
window_vars = [v for v in all_vars if 'window' in v.lower() and not v.startswith('_')]
for v in window_vars:
    try:
        var_type = type(eval(v)).__name__
        print(f"   {v} ({var_type})")
    except:
        pass

print("\n🔍 Variables containing 'label' or 'y':")
label_vars = [v for v in all_vars if (('label' in v.lower() or v == 'y' or v.startswith('y_'))
               and not v.startswith('_'))]
for v in label_vars:
    try:
        var_type = type(eval(v)).__name__
        print(f"   {v} ({var_type})")
    except:
        pass

print("\n🔍 Large list variables (potential data containers):")
for v in all_vars:
    if not v.startswith('_'):
        try:
            obj = eval(v)
            if isinstance(obj, list) and len(obj) > 100:
                print(f"   {v} (list with {len(obj)} items)")
        except:
            pass

print("\n🔍 NumPy array variables:")
for v in all_vars:
    if not v.startswith('_'):
        try:
            obj = eval(v)
            if isinstance(obj, np.ndarray) and obj.size > 100:
                print(f"   {v} (array with shape {obj.shape})")
        except:
            pass

print("\n" + "="*70)
print("⚠️  COPY THE OUTPUT ABOVE AND SHARE IT")
print("="*70)
max_num_win = X_ecg_train.shape[1]
max_win_len = X_ecg_train.shape[2]

print("max_num_win =", max_num_win)
print("max_win_len =", max_win_len)

model = build_crnn_lnn_model_fixed(max_num_win, max_win_len)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("🔥 Model compiled!")

