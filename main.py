import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import soundfile as sf


def load_wav_file(filename):
    return librosa.load(filename, sr=None)

def get_mfcc(y, sr, n_mfcc):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

def kalman_filter_for_features(observed, clean_features, A, B, H, Q, R, initial_state, initial_covariance):
    # state_estimate и covariance начинают с начальных значений
    state_estimate = initial_state
    covariance = initial_covariance

    # Список для сохранения оценок состояния
    estimates = []

    for idx, observation in enumerate(observed):
        # Шаг предсказания
        state_predict = A @ state_estimate  # @ обозначает умножение матрицы
        covariance_predict = A @ covariance @ A.T + Q

        # Вычисление коэффициента Калмана
        kalman_gain = covariance_predict @ H.T @ np.linalg.inv(H @ covariance_predict @ H.T + R)

        # Шаг обновления (коррекции)
        state_estimate = state_predict + kalman_gain @ (observation - clean_features[idx] - H @ state_predict)
        covariance = (np.identity(covariance.shape[0]) - kalman_gain @ H) @ covariance_predict

        # Сохранение оценки состояния
        estimates.append(state_estimate)

    return np.array(estimates)
    
def main(filename):
    num_coefficients = 128
    
    y, sr = load_wav_file(filename)
    mfcc_clean_features = get_mfcc(y, sr, num_coefficients).T
    
    noise = np.random.normal(0, 0.05, y.shape)  # 0.05 - это стандартное отклонение
    y = y + noise

    # 2. A: Матрица перехода состояния
    A = np.eye(num_coefficients)

    # 3. B: Матрица управления
    B = np.zeros((num_coefficients, num_coefficients))

    # 4. H: Матрица наблюдения
    H = np.eye(num_coefficients)

    Q = np.eye(num_coefficients) * 0.08
    R = np.eye(num_coefficients) * 0.5
    initial_state = np.zeros(
        num_coefficients)  # Начальное предположение может быть нулевым вектором, если нет лучшего предположения
    initial_covariance = np.eye(
        num_coefficients)  # Начальная ковариация может быть единичной матрицей, предполагая максимальную неопределенность

    mfcc_features = get_mfcc(y, sr, num_coefficients).T  # Транспонирование матрицы, чтобы каждое наблюдение было строкой
    filtered_mfcc = kalman_filter_for_features(mfcc_features, mfcc_clean_features, A, B, H, Q, R, initial_state, initial_covariance)

    # Восстановление аудио из MFCC
    melspec = librosa.feature.inverse.mfcc_to_mel(filtered_mfcc.T, n_mels=num_coefficients)
    y_mfcc_reconstructed = librosa.feature.inverse.mel_to_audio(melspec, sr=sr)
    
    # Определение размера самого большого массива
    max_size = max(y_mfcc_reconstructed.size, y.size)

    # Дополнение каждого массива нулями до размера max_size
    y_mfcc_reconstructed = np.pad(y_mfcc_reconstructed, (0, max_size - y_mfcc_reconstructed.size))
    y = np.pad(y, (0, max_size - y.size))

    combined_signal_pca = y - y_mfcc_reconstructed

    sf.write('rec.wav', y_mfcc_reconstructed, sr)
    sf.write('rec_pca.wav', combined_signal_pca, sr)
    sf.write('orig.wav', y, sr)
    
    # Визуализация
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.title("Original Signal")
    librosa.display.waveshow(y, sr=sr)
    plt.subplot(3, 1, 2)
    plt.title("Reconstructed from MFCC - Kalman")
    librosa.display.waveshow(y_mfcc_reconstructed, sr=sr)
    plt.subplot(3, 1, 3)
    plt.title("Reconstructed from PCA")
    librosa.display.waveshow(combined_signal_pca, sr=sr)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main('c:\\Users\Professional\Downloads\common_voice_ru_18947271.wav')