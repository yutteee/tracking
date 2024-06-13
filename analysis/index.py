import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def process_file(file_path):
    data = pd.read_csv(file_path)
    t_list = data.iloc[:, 0].values
    x_list = data.iloc[:, 1].values
    y_list = data.iloc[:, 2].values

    print(t_list)
    print(x_list)

    # グラフ描画
    plt.plot(t_list, x_list)
    plt.show()

    # Beats per minute
    ten_sec = 10 * 1000  # [ms]
    t_list_10sec = t_list[t_list <= ten_sec]
    x_list_10sec = x_list[:len(t_list_10sec)]
    x_smoothed = moving_average(x_list_10sec, 10)
    threshold = 5
    peaks, _ = find_peaks(x_smoothed, height=threshold)
    beats_per_minute = len(peaks) * 6
    print('Beats per minute:', beats_per_minute)

def main():
    OUTPUT_DIR = '../data_output'
    csv_files = [file for file in os.listdir(OUTPUT_DIR) if file.endswith('.csv')]

    for csv_file in csv_files:
        process_file(os.path.join(OUTPUT_DIR, csv_file))

if __name__ == '__main__':
    main()