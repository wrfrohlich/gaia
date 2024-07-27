import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr, spearmanr

# 1. Carregar os dados
wearable_data = pd.read_csv('wearable_data.csv')
camera_data = pd.read_csv('camera_data.csv')

# 2. Sincronizar dados (se necessário)
# Aqui, assumimos que os dados já estão sincronizados.

# 3. Filtragem de dados
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

fs = 100  # Frequência de amostragem
cutoff = 3.0  # Freqüência de corte
wearable_data['accel_x_filtered'] = butter_lowpass_filter(wearable_data['accel_x'], cutoff, fs)
camera_data['joint_angle_filtered'] = butter_lowpass_filter(camera_data['joint_angle'], cutoff, fs)

# 4. Extração de características
# Por exemplo, calcular a magnitude da aceleração
wearable_data['accel_magnitude'] = np.sqrt(wearable_data['accel_x_filtered']**2 + wearable_data['accel_y']**2 + wearable_data['accel_z']**2)

# 5. Normalização
wearable_data['accel_magnitude_normalized'] = (wearable_data['accel_magnitude'] - wearable_data['accel_magnitude'].mean()) / wearable_data['accel_magnitude'].std()
camera_data['joint_angle_normalized'] = (camera_data['joint_angle_filtered'] - camera_data['joint_angle_filtered'].mean()) / camera_data['joint_angle_filtered'].std()

# 6. Análise de correlação
correlation_pearson = pearsonr(wearable_data['accel_magnitude_normalized'], camera_data['joint_angle_normalized'])
correlation_spearman = spearmanr(wearable_data['accel_magnitude_normalized'], camera_data['joint_angle_normalized'])

print('Pearson correlation:', correlation_pearson)
print('Spearman correlation:', correlation_spearman)

# 7. Visualização
sns.scatterplot(x=wearable_data['accel_magnitude_normalized'], y=camera_data['joint_angle_normalized'])
plt.xlabel('Normalized Acceleration Magnitude')
plt.ylabel('Normalized Joint Angle')
plt.title('Correlation between Wearable Data and Camera Data')
plt.show()
