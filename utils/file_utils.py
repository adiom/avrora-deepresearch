import os
import hashlib
from datetime import datetime
import logging
from typing import Tuple

def generate_result_files() -> Tuple[str, str]:
    """
    Генерирует имена файлов для результатов исследования.
    
    Returns:
        Tuple[str, str]: (путь к файлу результатов, путь к файлу логов)
    """
    # Создаем директорию results если её нет
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Генерируем timestamp и hash
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    hash_input = f"{timestamp}-{os.urandom(8).hex()}"
    file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    # Формируем имена файлов
    result_file = f"results/result-{timestamp}-{file_hash}.md"
    log_file = f"results/result-{timestamp}-{file_hash}.log"
    
    return result_file, log_file

def setup_file_logging(log_file: str) -> None:
    """
    Настраивает логирование в файл.
    
    Args:
        log_file: путь к файлу логов
    """
    # Создаем форматтер для логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Создаем handler для файла
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Получаем корневой логгер и добавляем handler
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    # Устанавливаем уровень логирования
    root_logger.setLevel(logging.INFO)

def save_research_result(result_file: str, content: str) -> None:
    """
    Сохраняет результаты исследования в файл.
    
    Args:
        result_file: путь к файлу результатов
        content: содержимое для сохранения
    """
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(content) 