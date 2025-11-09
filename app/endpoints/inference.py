import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
import json
import logging
from pydantic import BaseModel
from datetime import datetime

# --- Налаштування шляхів та логера ---
logger = logging.getLogger(__name__)
# === ВИПРАВЛЕННЯ: Визначаємо абсолютні шляхи ===
CURRENT_FILE_PATH = os.path.abspath(__file__)
ENDPOINTS_DIR = os.path.dirname(CURRENT_FILE_PATH)
APP_DIR = os.path.dirname(ENDPOINTS_DIR)
PROJECT_ROOT = os.path.dirname(APP_DIR)

# --- Налаштування шляхів (тепер прив'язані до КОРЕНЯ) ---
DATABASE_FILE = os.path.join(PROJECT_ROOT, 'data', 'charge_database_alt.sqlite')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'production_model.joblib')

# --- Глобальна змінна для кешування моделі ---
# Це гарантує, що модель завантажується з диска лише 1 раз при старті
_model = None

def load_model():
    """Завантажує модель з диска у кеш, якщо вона ще не завантажена."""
    global _model
    if _model is None:
        logger.info(f"Завантаження моделі з {MODEL_PATH}...")
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Файл моделі не знайдено за шляхом: {MODEL_PATH}")
            raise FileNotFoundError(f"Модель не знайдено. Спочатку запустіть /train-model.")
        _model = joblib.load(MODEL_PATH)
        logger.info("Модель успішно завантажено.")
    return _model

# --- 2. Pydantic Модель (Валідація Вхідних Даних) ---
# Ця модель описує, який JSON ми очікуємо.
# Назви полів ПОВИННІ збігатися з тими, на яких тренувалася модель.
class InferenceInput(BaseModel):
    # З ChargingSessions_ScaledNoAnomaly
    state_of_charge_start: float
    distance_driven_km: float
    temperature_c: float
    
    # З Vehicles_ScaledNoAnomaly
    battery_capacity_kwh: float
    vehicle_age_years: float
    
    # З Users_Encoded
    user_type_Commuter: float
    user_type_Long_Distance_Traveler: float # Pydantic не любить пробіли
    user_type_Occasional: float
    
    class Config:
        # Дозволяє Pydantic працювати з назвою "user_type_Long-Distance Traveler"
        alias_generator = lambda string: string.replace('_', ' ')
        populate_by_name = True

# --- 3. Логіка Логування ---

def log_to_inference_inputs(data: InferenceInput) -> int:
    """Логує вхідні параметри в БД та повертає ID запису."""
    logger.info("Логування вхідних даних у 'inference_inputs'...")
    data_json = data.model_dump_json(by_alias=True) # Зберігаємо як JSON
    now = datetime.now().isoformat()
    
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO inference_inputs (request_timestamp, input_data_json) VALUES (?, ?)",
            (now, data_json)
        )
        conn.commit()
        input_id = cursor.lastrowid # Отримуємо ID
        conn.close()
        logger.info(f"Вхідні дані збережено, input_id: {input_id}")
        return input_id
    except Exception as e:
        logger.error(f"Помилка логування в 'inference_inputs': {e}")
        raise Exception(f"Помилка логування в 'inference_inputs': {e}")

def log_to_predictions(input_id: int, prediction: str):
    """Логує результат прогнозу в БД."""
    logger.info(f"Логування прогнозу для input_id {input_id}...")
    now = datetime.now().isoformat()
    
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        # 'actual_class' = NULL, оскільки ми не знаємо правди
        cursor.execute(
            """
            INSERT INTO predictions (input_id, prediction_timestamp, source, predicted_class, actual_class)
            VALUES (?, ?, 'inference', ?, NULL)
            """,
            (input_id, now, prediction)
        )
        conn.commit()
        conn.close()
        logger.info("Прогноз успішно залоговано.")
    except Exception as e:
        logger.error(f"Помилка логування в 'predictions': {e}")
        # Не "ламаємо" запит, якщо лог не вдався
        
# --- 4. Головна Функція Передбачення ---

def run_prediction_pipeline(data: InferenceInput) -> str:
    """
    Виконує повний пайплайн для одного запиту:
    1. Логує вхідні дані.
    2. Завантажує модель.
    3. Робить прогноз.
    4. Логує вихідні дані.
    5. Повертає результат.
    """
    
    # 1. Логуємо вхідні дані та отримуємо ID
    try:
        input_id = log_to_inference_inputs(data)
    except Exception as e:
        # Якщо логування впало, не продовжуємо
        raise Exception(f"Не вдалося залогувати вхідні дані: {e}")
    
    try:
        # 2. Завантажуємо модель (з кешу)
        model = load_model()
        
        # 3. Готуємо дані для моделі
        # (Перетворюємо Pydantic-об'єкт на DataFrame з 1 рядком)
        # Важливо: імена в model_dump() повинні відповідати іменам ознак
        input_df = pd.DataFrame([data.model_dump(by_alias=True)])
        
        # 4. Робимо прогноз
        # model.predict() повертає numpy array, беремо [0] (перший елемент)
        prediction = model.predict(input_df)[0]
        
        # 5. Логуємо результат
        log_to_predictions(input_id, str(prediction))
        
        # 6. Повертаємо результат
        return str(prediction)
        
    except Exception as e:
        logger.error(f"Помилка під час прогнозування: {e}")
        raise Exception(f"Помилка під час прогнозування: {e}")