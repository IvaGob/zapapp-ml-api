import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
import lightgbm as lgb
from sklearn.metrics import f1_score
from datetime import datetime
import logging

# Налаштовуємо логер
logger = logging.getLogger(__name__)
# 1. Знаходимо шлях до поточного файлу (training.py)
CURRENT_FILE_PATH = os.path.abspath(__file__)
# 2. Знаходимо шлях до папки 'endpoints'
ENDPOINTS_DIR = os.path.dirname(CURRENT_FILE_PATH)
# 3. Знаходимо шлях до папки 'app'
APP_DIR = os.path.dirname(ENDPOINTS_DIR)
# 4. Знаходимо шлях до КОРЕНЯ ПРОЕКТУ ('root')
PROJECT_ROOT = os.path.dirname(APP_DIR)

# --- Налаштування шляхів (тепер прив'язані до КОРЕНЯ) ---
DATABASE_FILE = os.path.join(PROJECT_ROOT, 'data', 'charge_database_alt.sqlite')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, 'production_model.joblib')

# --- Перевірка шляху до БД ---
logger.info(f"[training.py] Шукаю базу даних за шляхом: {DATABASE_FILE}")
if not os.path.exists(DATABASE_FILE):
    logger.error(f"[training.py] !!! ФАЙЛ БАЗИ ДАНИХ НЕ ЗНАЙДЕНО за шляхом: {DATABASE_FILE} !!!")

# --- 1. Логіка Завантаження Даних ---
def load_data(db_file: str) -> pd.DataFrame:
    """Завантажує та об'єднує дані з трьох таблиць."""
    logger.info("Завантаження даних з БД...")
    try:
        conn = sqlite3.connect(db_file)
        df_sessions = pd.read_sql_query("SELECT * FROM ChargingSessions_ScaledNoAnomaly", conn)
        df_users = pd.read_sql_query("SELECT * FROM Users_Encoded", conn)
        df_vehicles = pd.read_sql_query("SELECT * FROM Vehicles_ScaledNoAnomaly", conn)
        conn.close()
    except Exception as e:
        logger.error(f"Помилка завантаження даних: {e}")
        raise Exception(f"Помилка завантаження даних: {e}")

    df = pd.merge(df_sessions, df_users, on='user_id', how='left')
    df = pd.merge(df, df_vehicles.drop(columns=['user_id'], errors='ignore'), on='vehicle_id', how='left')
    
    df['charging_start_time'] = pd.to_datetime(df['charging_start_time'])
    df = df.sort_values(by='charging_start_time').reset_index(drop=True)
    return df

# --- 2. Логіка Підготовки Даних ---
# === ВИПРАВЛЕННЯ: Змінено (pd.DataFrame, pd.Series) на tuple[pd.DataFrame, pd.Series] ===
def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Готує X (ознаки) та y (ціль) з DataFrame."""
    logger.info("Підготовка ознак (X) та цілі (y)...")
    try:
        conditions = [
            df['charger_type_Level 1'] == 1,
            df['charger_type_Level 2'] == 1,
            df['charger_type_DC Fast Charger'] == 1
        ]
        choices = ['Level 1', 'Level 2', 'DC Fast Charger']
        df['Target_ChargerType'] = np.select(conditions, choices, default='Other')
        df = df[df['Target_ChargerType'] != 'Other']
    except KeyError as e:
        logger.error(f"Відсутня ключова колонка для створення 'Target_ChargerType': {e}")
        raise Exception(f"Необхідна колонка відсутня: {e}")

    y = df['Target_ChargerType']
    
    columns_to_drop = [
        'session_id', 'user_id', 'vehicle_id', 'charging_station_id', 
        'charging_start_time', 'charging_end_time', 'timestamp', 
        'charging_station_location', 'time_of_day', 'day_of_week', 'vehicle_model',
        'energy_consumed_kwh', 'charging_duration_hours', 'charging_rate_kw', 
        'charging_cost_usd', 'state_of_charge_end',
        'charger_type_DC Fast Charger', 'charger_type_Level 1', 'charger_type_Level 2',
        'Target_ChargerType'
    ]
    
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    X = df.drop(columns=existing_columns_to_drop)
    
    if X.isnull().sum().sum() > 0:
        logger.warning("Знайдено пропуски (NaN) в X. Заповнюю нулями...")
        X = X.fillna(0)
        
    return X, y

# --- 3. Логіка Логування ---
def log_predictions_to_db(db_file: str, y_true: pd.Series, y_pred: np.ndarray, source: str):
    """Записує прогнози у таблицю 'predictions'."""
    logger.info(f"Логування {len(y_pred)} прогнозів у БД з джерелом '{source}'...")
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        now_timestamp = datetime.now().isoformat()
        data_to_insert = [
            (None, now_timestamp, source, pred, true)
            for pred, true in zip(y_pred, y_true)
        ]
        cursor.executemany(
            """
            INSERT INTO predictions (input_id, prediction_timestamp, source, predicted_class, actual_class)
            VALUES (?, ?, ?, ?, ?)
            """,
            data_to_insert
        )
        conn.commit()
    except Exception as e:
        # === ЗМІНА: змушуємо помилку бути голосною ===
        error_message = f"!!! КРИТИЧНА ПОМИЛКА ЛОГУВАННЯ: {e} !!!"
        print(error_message) # Друкуємо в консоль
        logger.error(error_message)

        # Додаємо повний трейсбек, щоб побачити точну причину
        import traceback
        logger.error(traceback.format_exc()) 
        # === Кінець зміни ===
    finally:
        if conn:
            conn.close()

# --- 4. Головна Функція Тренування ---
def run_training_pipeline() -> dict:
    """
    Запускає повний процес тренування моделі та повертає результати.
    """
    logger.info("Запуск повного пайплайну тренування...")
    
    all_data = load_data(DATABASE_FILE)
    X, y = prepare_data(all_data)
    
    if X.empty:
        raise Exception("Не знайдено даних для тренування.")

    split_index = int(len(X) * 0.9)
    if split_index < 1 or (len(X) - split_index) < 1:
        raise Exception("Недостатньо даних для поділу.")

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    logger.info(f"Дані розділено: {len(X_train)} train, {len(X_test)} new_input (test).")
    
    logger.info("Тренування моделі LightGBM...")
    best_params = {
        'subsample': 1.0, 'reg_lambda': 0.0, 'reg_alpha': 1.0,
        'num_leaves': 40, 'n_estimators': 200, 'max_depth': -1,
        'learning_rate': 0.05, 'colsample_bytree': 0.8
    }
    model = lgb.LGBMClassifier(**best_params, random_state=42, n_jobs=-1)
    
    model.fit(X_train, y_train)
    
    logger.info(f"Збереження моделі у {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    
    y_pred_train = model.predict(X_train)
    log_predictions_to_db(DATABASE_FILE, y_train, y_pred_train, source="train")
    
    y_pred_test = model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    logger.info(f"Фінальна F1-score на new_input (test) даних: {test_f1:.4f}")
    
    return {
        "status": "success",
        "message": "Модель успішно натренована та збережена.",
        "model_path": MODEL_PATH,
        "test_f1_score": test_f1,
        "trained_samples": len(X_train),
        "test_samples": len(X_test)
    }