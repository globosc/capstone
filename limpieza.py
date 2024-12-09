#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from joblib import load
import argparse
import sys

def calculate_smoothness(positions):
    velocity = np.diff(positions, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    padding = np.zeros(3)
    return np.concatenate([padding, jerk_magnitude])

def calculate_movement_economy(positions):
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative_distance = np.concatenate([[0], np.cumsum(distances)])
    direct_distances = np.linalg.norm(positions - positions[0], axis=1)
    economy = np.where(cumulative_distance > 0,
                      direct_distances / (cumulative_distance + 1e-10),
                      1.0)
    return economy

def calculate_angular_velocity(quaternions):
    q_diff = np.diff(quaternions, axis=0)
    angular_velocity = np.linalg.norm(q_diff, axis=1)
    return np.concatenate([[0], angular_velocity])

def add_features(df):
    positions = df[['PositionX', 'PositionY', 'PositionZ']].values
    quaternions = df[['RotationW', 'RotationX', 'RotationY', 'RotationZ']].values

    df['movement_smoothness'] = calculate_smoothness(positions)
    df['movement_economy'] = calculate_movement_economy(positions)
    df['angular_velocity'] = calculate_angular_velocity(quaternions)

    return df

def interpolate_timeseries(df, target_length=1800):
    df = df.copy()
    # Convertir timestamp a segundos desde el inicio
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    start_time = df['Timestamp'].min()
    df['time'] = (df['Timestamp'] - start_time).dt.total_seconds()
    
    common_time = np.linspace(0, df['time'].iloc[-1], target_length)
    interpolated_df = pd.DataFrame({'time': common_time})
    skip_cols = ['time', 'Timestamp']
    
    for col in df.columns:
        if col not in skip_cols:
            interpolated_df[col] = interp1d(
                df['time'], 
                df[col], 
                kind='cubic', 
                fill_value='extrapolate'
            )(common_time)
    
    # Reconstruir timestamps interpolados
    interpolated_df['Timestamp'] = pd.to_datetime(start_time + pd.to_timedelta(common_time, unit='s'))
    
    return interpolated_df

def prepare_data_for_hmm(df):
    hmm_columns = [
        'PositionX', 'PositionY', 'PositionZ',
        'RotationX', 'RotationY', 'RotationZ', 'RotationW',
        'movement_smoothness', 'movement_economy', 'angular_velocity'
    ]
    return df[hmm_columns].copy()

def process_new_data(input_file, model_file):
    try:
        # Cargar datos y modelo
        print("Cargando datos...")
        df = pd.read_csv(input_file)
        hmm_model = load(model_file)
        
        # Agregar características
        print("Calculando características...")
        df = add_features(df)
        
        # Interpolar series de tiempo
        print("Interpolando series temporales...")
        df_interpolated = interpolate_timeseries(df)
        
        # Preparar datos para HMM
        print("Preparando datos para HMM...")
        df_hmm = prepare_data_for_hmm(df_interpolated)
        
        # Obtener secuencia de Viterbi
        print("Obteniendo secuencias de Viterbi...")
        features = df_hmm.values
        hidden_states = hmm_model.predict(features)
        
        # Agregar estados ocultos al DataFrame original
        df_interpolated['hidden_state'] = hidden_states
        
        # Guardar resultados
        output_file = 'processed_' + input_file
        df_interpolated.to_csv(output_file, index=False)
        print(f"Datos procesados guardados en {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Procesar nuevos datos con HMM')
    parser.add_argument('input_file', help='Archivo CSV con datos de entrada')
    parser.add_argument('model_file', help='Archivo .joblib con modelo HMM')
    args = parser.parse_args()
    
    process_new_data(args.input_file, args.model_file)
