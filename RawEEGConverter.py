import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


# Function to create a bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


# Function to compute average power for each frequency band
def compute_average_power(eeg_data, fs):
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'high beta': (30, 40),
        'gamma': (40, 100)
    }

    results = {band: {} for band in bands}

    for channel in eeg_data.columns[1:]:  # Skip the time column
        channel_data = eeg_data[channel].values

        for band, (lowcut, highcut) in bands.items():
            # Filter the data
            filtered_data = bandpass_filter(channel_data, lowcut, highcut, fs)
            # Compute power spectral density using FFT
            fft_result = np.fft.fft(filtered_data)
            power_spectrum = np.abs(fft_result) ** 2

            # Calculate frequencies
            freqs = np.fft.fftfreq(len(filtered_data), d=1 / fs)

            # Identify frequency indices for the band
            band_indices = np.where((freqs >= lowcut) & (freqs <= highcut))
            # Calculate average power for the band
            average_power = np.mean(power_spectrum[band_indices])

            results[band][channel] = average_power

    return results

data = {
    'Time (s)': list(range(100)),  # More data points
    'FP1 (µV)': np.random.normal(11, 0.5, 100),  # Simulated EEG data
    'FP2 (µV)': np.random.normal(11, 0.5, 100),
    'F3 (µV)': np.random.normal(5, 0.3, 100)
}


eeg_data = pd.DataFrame(data)

def graph(data):
    eeg_data = pd.DataFrame(data)

    # Create a nice line graph for the EEG data in the data dictionary
    plt.figure(figsize=(10, 6))

    # Plot the EEG data for each channel
    for channel in eeg_data.columns[1:]:
        plt.plot(eeg_data['Time (s)'], eeg_data[channel], label=channel)

    # Add labels and title
    plt.title("EEG Signal Over Time", fontsize=16)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Amplitude (µV)", fontsize=12)

    # Add legend
    plt.legend(loc="upper right")

    # Display the grid
    plt.grid(True)
    plt.savefig('RAWEEG.png')

    # Show the plot
    plt.show()


# Sampling frequency
fs = 256  # Hz

# Compute average power for each frequency band
average_power_results = compute_average_power(eeg_data, fs)


new_df = pd.DataFrame()
# Display results
for band, channels in average_power_results.items():
    print(f"\n{band} Power (µV²):")
    for channel, power in channels.items():
        print(f"  {channel}: {power:.4f} µV²")

average_power_results = compute_average_power(eeg_data, fs)

# Create a DataFrame with the results
results_df = pd.DataFrame([average_power_results])

# Display results
print("\nResults DataFrame:")
print(results_df)
graph(data)