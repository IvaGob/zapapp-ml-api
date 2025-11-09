import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# === ВИПРАВЛЕННЯ: Кажемо Python зайти в папку 'app' ===
from endpoints import training 
from endpoints import inference 

# --- Налаштування шляхів ---
LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok=True)

# --- Налаштування логування ---
# ... (ваш код логування залишається без змін) ...
log_file = os.path.join(LOGS_DIR, 'api_logs.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file), 
        logging.StreamHandler()      
    ]
)
logger = logging.getLogger(__name__)

# --- Ініціалізація FastAPI ---
app = FastAPI(
    title="ZapApp Model API",
    description="API для тренування та передбачення."
)

# === Завантаження моделі при старті ===
@app.on_event("startup")
def startup_event():
    """Завантажує модель у кеш при старті сервера."""
    logger.info("Сервер запускається...")
    try:
        # Тепер inference - це модуль, завантажений з 'app'
        inference.load_model()
        logger.info("Модель успішно завантажена в кеш.")
    except FileNotFoundError:
        logger.warning("Файл моделі 'production_model.joblib' не знайдено.")
        logger.warning("Запустіть /train-model, щоб створити її.")
    except Exception as e:
        logger.error(f"Критична помилка при завантаженні моделі: {e}")

# --- 1. Моделі Pydantic для Ендпоінтів ---
class TrainResponse(BaseModel):
    status: str
    message: str
    model_path: str
    test_f1_score: float
    trained_samples: int
    test_samples: int

class PredictionResponse(BaseModel):
    predicted_class: str

# --- 2. Ендпоінт для Тренування ---
@app.post("/train-model", response_model=TrainResponse)
def trigger_training():
    """
    Запускає повний процес тренування моделі.
    """
    logger.info("Отримано POST-запит на /train-model...")
    try:
        # Викликаємо функцію з імпортованого модуля
        result = training.run_training_pipeline()
        inference.load_model() 
        return result
    except Exception as e:
        logger.error(f"Під час тренування сталася помилка: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# === 3. ЕНДПОІНТ для Передбачення ===
@app.post("/predict", response_model=PredictionResponse)
def get_prediction(data: inference.InferenceInput):
    """
    Робить прогноз для одного запиту.
    """
    logger.info("Отримано POST-запит на /predict...")
    try:
        # Викликаємо функцію з імпортованого модуля
        prediction = inference.run_prediction_pipeline(data)
        
        return {"predicted_class": prediction}
        
    except FileNotFoundError as e:
        logger.error(f"Помилка прогнозування: Модель не навчена. {e}")
        raise HTTPException(status_code=503, detail="Модель не навчена. Будь ласка, запустіть /train-model.")
    except Exception as e:
        logger.error(f"Під час прогнозування сталася помилка: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- 4. Запуск Сервера (для локальної розробки) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Запуск FastAPI сервера на http://127.0.0.1:8000")
    # Команда запуску залишається 'main:app', оскільки файл - 'main.py'
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)