import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler
from joblib import load

def calculate_features(df):
    """
    Adds movement_smoothness, movement_economy, and angular_velocity features.
    """
    def calculate_smoothness(positions):
        velocity = np.diff(positions, axis=0)
        acceleration = np.diff(velocity, axis=0)
        jerk = np.diff(acceleration, axis=0)
        jerk_magnitude = norm(jerk, axis=1)
        smoothness = np.concatenate([np.zeros(3), jerk_magnitude])
        return smoothness

    def calculate_movement_economy(positions):
        distances = norm(np.diff(positions, axis=0), axis=1)
        cumulative_distance = np.concatenate([[0], np.cumsum(distances)])
        direct_distances = norm(positions - positions[0], axis=1)
        economy = np.where(cumulative_distance > 0, direct_distances / (cumulative_distance + 1e-10), 1.0)
        return economy

    def calculate_angular_velocity(quaternions):
        q_diff = np.diff(quaternions, axis=0)
        angular_velocity = norm(q_diff, axis=1)
        angular_velocity = np.concatenate([[0], angular_velocity])
        return angular_velocity

    positions = df[['PositionX', 'PositionY', 'PositionZ']].values
    quaternions = df[['RotationW', 'RotationX', 'RotationY', 'RotationZ']].values

    df['movement_smoothness'] = calculate_smoothness(positions)
    df['movement_economy'] = calculate_movement_economy(positions)
    df['angular_velocity'] = calculate_angular_velocity(quaternions)

    return df

def preprocess_with_existing_hmm(file_path, target_length, hmm_model_path):
    # Leer archivo
    data = pd.read_csv(file_path)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['session_id'] = 0  # Asignar ID de sesión único si es solo un archivo

    # Agregar características
    data = calculate_features(data)

    # Interpolación
    def interpolate_timeseries(df, target_length):
        session_data = df.copy()
        session_data['time'] = (pd.to_datetime(session_data['Timestamp']) - pd.to_datetime(session_data['Timestamp'].iloc[0])).dt.total_seconds()
        common_time = np.linspace(0, session_data['time'].iloc[-1], target_length)
        interpolated_df = pd.DataFrame({'time': common_time})
        for col in session_data.columns:
            if col not in ['Timestamp', 'time', 'session_id']:
                interpolated_df[col] = interp1d(session_data['time'], session_data[col], kind='linear', fill_value='extrapolate')(common_time)
        interpolated_df['session_id'] = 0
        return interpolated_df

    data = interpolate_timeseries(data, target_length)

    # Cargar el modelo HMM existente
    hmm_model = load(hmm_model_path)
    features = ['PositionX', 'PositionY', 'PositionZ', 'RotationX', 'RotationY', 'RotationZ', 
                'RotationW', 'movement_smoothness', 'movement_economy', 'angular_velocity']
    scaler = RobustScaler()
    X = scaler.fit_transform(data[features].values)
    states = hmm_model.predict(X)
    data['hidden_state'] = states

    return data

if __name__ == "__main__":
    import argparse

    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Preprocesar datos utilizando un HMM ya entrenado.")
    parser.add_argument("file_path", help="Ruta del archivo CSV a procesar")
    parser.add_argument("--target_length", type=int, default=1800, help="Longitud objetivo para interpolar")
    parser.add_argument("--hmm_model_path", help="Ruta del modelo HMM entrenado (joblib)")

    args = parser.parse_args()

    # Procesar el archivo
    processed_data = preprocess_with_existing_hmm(args.file_path, args.target_length, args.hmm_model_path)

    # Guardar resultado
    output_path = args.file_path.replace(".csv", "_processed.csv")
    processed_data.to_csv(output_path, index=False)
    print(f"Archivo procesado guardado en: {output_path}")

