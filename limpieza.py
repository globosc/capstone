import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from numpy.linalg import norm
import joblib
import sys

def add_relative_time(df):
    """Añade tiempo relativo a la sesión."""
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    start_time = df['Timestamp'].min()
    df['timestamp_sec'] = (df['Timestamp'] - start_time).dt.total_seconds()
    return df

def calculate_smoothness(positions):
    """Calcula la suavidad del movimiento."""
    velocity = np.diff(positions, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)
    jerk_magnitude = norm(jerk, axis=1)
    padding = np.zeros(3)
    return np.concatenate([padding, jerk_magnitude])

def calculate_movement_economy(positions):
    """Calcula la economía del movimiento."""
    distances = norm(np.diff(positions, axis=0), axis=1)
    cumulative_distance = np.concatenate([[0], np.cumsum(distances)])
    direct_distances = norm(positions - positions[0], axis=1)
    economy = np.where(cumulative_distance > 0,
                      direct_distances / (cumulative_distance + 1e-10),
                      1.0)
    return economy

def calculate_angular_velocity(quaternions):
    """Calcula la velocidad angular."""
    q_diff = np.diff(quaternions, axis=0)
    angular_velocity = norm(q_diff, axis=1)
    return np.concatenate([[0], angular_velocity])

def add_features(df):
    """Añade características calculadas al DataFrame."""
    positions = df[['PositionX', 'PositionY', 'PositionZ']].values
    quaternions = df[['RotationW', 'RotationX', 'RotationY', 'RotationZ']].values
    
    df['movement_smoothness'] = calculate_smoothness(positions)
    df['movement_economy'] = calculate_movement_economy(positions)
    df['angular_velocity'] = calculate_angular_velocity(quaternions)
    
    return df

def interpolate_timeseries(df, target_length=1800):
    """Interpola la serie temporal a una longitud específica."""
    df = df.copy()
    df['time'] = (pd.to_datetime(df['Timestamp']) - pd.to_datetime(df['Timestamp'].iloc[0])).dt.total_seconds()
    common_time = np.linspace(0, df['time'].iloc[-1], target_length)
    
    interpolated_df = pd.DataFrame({'time': common_time})
    skip_cols = ['Timestamp', 'time', 'session_id', 'Label']
    
    for col in df.columns:
        if col not in skip_cols:
            interpolated_df[col] = interp1d(df['time'], df[col], 
                                          kind='cubic', 
                                          fill_value='extrapolate')(common_time)
    
    return interpolated_df

def prepare_data_for_hmm(df):
    """Prepara los datos para el modelo HMM."""
    hmm_columns = ['PositionX', 'PositionY', 'PositionZ', 
                   'RotationX', 'RotationY', 'RotationZ', 'RotationW',
                   'movement_smoothness', 'movement_economy', 'angular_velocity']
    return df[hmm_columns].copy()

def process_new_session(file_path, hmm_model_path):
    """Procesa una nueva sesión desde un archivo CSV."""
    # Cargar datos
    df = pd.read_csv(file_path)
    
    # Aplicar preprocesamiento
    df = add_relative_time(df)
    df = add_features(df)
    df = interpolate_timeseries(df)
    df = prepare_data_for_hmm(df)
    
    # Cargar modelo HMM y predecir estados
    hmm_model = joblib.load(hmm_model_path)
    df['hidden_state'] = hmm_model.predict(df.values)
    
    return df

def main():
    if len(sys.argv) != 3:
        print("Uso: python3 limpieza.py <archivo_sesion.csv> <modelo_hmm.joblib>")
        sys.exit(1)
    
    session_file = sys.argv[1]
    hmm_model_file = sys.argv[2]
    
    try:
        processed_df = process_new_session(session_file, hmm_model_file)
        output_file = f"processed_{session_file}"
        processed_df.to_csv(output_file, index=False)
        print(f"Sesión procesada guardada en: {output_file}")
    except Exception as e:
        print(f"Error procesando la sesión: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
