import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Rango de expertos para feedback
EXPERT_RANGES = {
    'movement_smoothness': (0.6, 0.9),
    'movement_economy': (0.6, 0.8),
    'angular_velocity': (0.2, 0.4)
}

# Cargar clasificador previamente entrenado
def load_classifier(model_path='best_rf_model.joblib'):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        print("Error: No se encontró el modelo clasificador. Asegúrate de entrenarlo y guardarlo como 'rf_model.pkl'.")
        sys.exit(1)

# Generar feedback para una sesión
def generate_feedback(session_metrics, expert_ranges):
    feedback = []
    
    # Suavidad de movimientos
    smoothness = session_metrics['avg_movement_smoothness']
    if smoothness < expert_ranges['movement_smoothness'][0]:
        feedback.append("🌊 Suavidad: Tus movimientos son bruscos. Intenta mantener una velocidad más constante.")
    elif smoothness > expert_ranges['movement_smoothness'][1]:
        feedback.append("🌊 Suavidad: Movimientos demasiado lentos. Busca un ritmo más dinámico.")
    else:
        feedback.append("🌊 Suavidad: ¡Excelente fluidez!")

    # Economía de movimiento
    economy = session_metrics['avg_movement_economy']
    if economy < expert_ranges['movement_economy'][0]:
        feedback.append("🎯 Economía: Movimientos largos. Planifica rutas más cortas.")
    elif economy > expert_ranges['movement_economy'][1]:
        feedback.append("🎯 Economía: Movimientos demasiado conservadores. Usa el espacio más efectivamente.")
    else:
        feedback.append("🎯 Economía: ¡Buena planificación de movimientos!")
    
    # Velocidad angular
    angular = session_metrics['avg_angular_velocity']
    if angular < expert_ranges['angular_velocity'][0]:
        feedback.append("🎮 Rotación: Movimientos angulares limitados. Explora mayor flexibilidad.")
    elif angular > expert_ranges['angular_velocity'][1]:
        feedback.append("🎮 Rotación: Rotaciones erráticas. Controla mejor la orientación.")
    else:
        feedback.append("🎮 Rotación: ¡Buen control angular!")

    return feedback

# Evaluar sesión
def evaluate_session(file_path, model):
    # Cargar datos
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        sys.exit(1)

    # Calcular métricas
    session_metrics = {
        'avg_movement_smoothness': data['movement_smoothness'].mean(),
        'avg_movement_economy': data['movement_economy'].mean(),
        'avg_angular_velocity': data['angular_velocity'].mean()
    }

    # Predecir clasificación con el modelo
    X = data['hidden_state'].values.reshape(-1, 1)
    probabilities = model.predict_proba(X)[:, 1]
    session_prob = probabilities.mean()
    classification = "Experto" if session_prob > 0.4 else "Novato"

    # Generar feedback
    feedback = generate_feedback(session_metrics, EXPERT_RANGES)

    # Mostrar resultados
    print("\nEvaluación de la Sesión:")
    print("------------------------")
    print(f"Clasificación: {classification}")
    print(f"Probabilidad promedio de ser experto: {session_prob:.3f}")
    print("\nRetroalimentación:")
    for line in feedback:
        print(f"- {line}")

# Entrada principal
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python3 evaluacion.py <ruta_del_archivo>")
        sys.exit(1)

    # Cargar el clasificador
    model = load_classifier()

    # Evaluar la sesión proporcionada
    evaluate_session(sys.argv[1], model)

