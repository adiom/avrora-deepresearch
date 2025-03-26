import logging # Добавляем импорт
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import google.generativeai as genai
from research_logic import perform_deep_research
from typing import List, Optional, Dict, Any
import json
from utils.file_utils import generate_result_files, setup_file_logging, save_research_result

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, # Уровень логирования (INFO, DEBUG, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
# ---------------------------

load_dotenv() # Загрузка переменных из .env
logger.info("Загрузка переменных окружения...")

# Конфигурация Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("Переменная окружения GOOGLE_API_KEY не найдена!")
    raise ValueError("Необходимо установить переменную окружения GOOGLE_API_KEY")

logger.info("Конфигурация Google Generative AI...")
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Google Generative AI успешно сконфигурирован.")
except Exception as e:
    logger.exception("Ошибка конфигурации Google Generative AI!")
    raise

app = FastAPI(
    title="Deep Research API",
    description="API для выполнения глубокого итеративного исследования с использованием Gemini."
)
logger.info("FastAPI приложение инициализировано.")

# Модель запроса Pydantic
class ResearchRequest(BaseModel):
    query: str = Field(..., description="Основной запрос для исследования")
    depth: int = Field(default=2, ge=1, le=5, description="Глубина исследования (количество итераций)")
    breadth: int = Field(default=3, ge=1, le=5, description="Ширина исследования (количество направлений/запросов на шаге)")
    existing_learnings: Optional[List[str]] = Field(None, description="Опциональный список существующих знаний")

# Модель ответа Pydantic
class ResearchResponse(BaseModel):
    result_file: str = Field(..., description="Путь к файлу с результатами исследования")
    error: Optional[str] = Field(None, description="Описание ошибки, если она возникла")

@app.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    """
    Запускает процесс глубокого исследования.
    """
    # Генерируем имена файлов для результатов
    result_file, log_file = generate_result_files()
    
    # Настраиваем логирование в файл
    setup_file_logging(log_file)
    
    logger.info(f"Получен запрос на исследование: query='{request.query}', depth={request.depth}, breadth={request.breadth}")
    try:
        report, sources, conclusions, directions = await perform_deep_research(
            initial_query=request.query,
            depth=request.depth,
            breadth=request.breadth,
            existing_learnings=request.existing_learnings
        )
        
        # Формируем итоговый отчет
        final_report = await generate_final_report(request.query, sources, {
            "report": report,
            "conclusions": conclusions,
            "directions": directions
        })
        
        if not final_report:
            raise ValueError("Не удалось сгенерировать итоговый отчет")
        
        # Добавляем заголовок к отчету
        header = """исследование создано с помощью canfly-avrora-deepresearch и содержит в себе информацию из открытых источников. 2005-2025 (С) canfly | культура твоего сознания

---
"""
        final_report = header + final_report
        
        # Сохраняем результаты в файл
        save_research_result(result_file, final_report)
            
        logger.info(f"Исследование для запроса '{request.query}' успешно завершено.")
        return ResearchResponse(result_file=result_file)
    except Exception as e:
        error_msg = f"Произошла ошибка во время исследования для запроса '{request.query}': {str(e)}"
        logger.exception(error_msg)
        return ResearchResponse(result_file="", error=error_msg)

@app.get("/")
async def root():
    logger.debug("Запрос к корневому эндпоинту /")
    return {"message": "Deep Research API запущен"}

async def generate_final_report(query: str, sources: List[str], research_data: Dict[str, Any]) -> str:
    """Генерирует итоговый отчет на основе собранных данных."""
    try:
        # Создаем промпт для генерации отчета
        prompt = f"""На основе следующей информации создайте подробный отчет:

Запрос: {query}

Источники:
{chr(10).join(f'- {source}' for source in sources)}

Основные выводы:
{chr(10).join(f'- {conclusion}' for conclusion in research_data["conclusions"])}

Направления для дальнейшего исследования:
{chr(10).join(f'- {direction}' for direction in research_data["directions"])}

Создайте структурированный отчет, который включает:
1. Краткое введение
2. Основные выводы
3. Подробный анализ
4. Заключение

Отчет должен быть информативным и хорошо структурированным."""

        # Инициализируем модель Gemini
        model = genai.GenerativeModel('gemini-2.0-pro-exp')
        
        # Генерируем отчет с помощью Gemini
        response = await model.generate_content_async(prompt)
        
        if not response or not response.text:
            raise ValueError("Не удалось получить ответ от модели")
            
        return response.text
    except Exception as e:
        logger.error(f"Ошибка при генерации итогового отчета: {e}")
        return f"Произошла ошибка при генерации отчета: {str(e)}"

# Если этот файл запускается напрямую (для отладки, хотя обычно используется uvicorn)
if __name__ == "__main__":
    # Для запуска: uvicorn main:app --reload --host 0.0.0.0 --port 8000
    logger.warning("Запуск через 'python main.py' не рекомендуется для продакшена. Используйте uvicorn.")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)