import asyncio
from typing import List, Tuple, Dict
from utils import search, scraper, llm # Импортируем наши модули

async def perform_deep_research(
    initial_query: str,
    depth: int,
    breadth: int,
    existing_learnings: List[str] = None
) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Выполняет итеративное глубокое исследование.

    Возвращает:
        Tuple[str, List[str], List[str], List[str]]: Кортеж (Итоговый отчет в Markdown, Список источников, Список выводов, Список направлений)
    """
    if existing_learnings is None:
        existing_learnings = []

    all_learnings = list(existing_learnings)
    visited_urls = set()
    all_sources = []
    current_query_context = initial_query
    final_directions = []

    for i in range(depth):
        print(f"--- Глубина {i+1}/{depth} ---")

        # 1. Генерация поисковых запросов с помощью Gemini
        print("Генерация поисковых запросов...")
        search_queries = await llm.generate_search_queries(
            current_query_context,
            all_learnings,
            breadth
        )
        print(f"Сгенерировано запросов: {search_queries}")

        # 2. Поиск ссылок для каждого запроса
        urls_to_scrape = []
        search_tasks = [search.find_urls(query, num_results=2) for query in search_queries]
        results = await asyncio.gather(*search_tasks)
        for urls in results:
            for url in urls:
                if url not in visited_urls:
                    urls_to_scrape.append(url)
                    visited_urls.add(url)

        if not urls_to_scrape:
            print("Не найдено новых URL для исследования на этом шаге.")
            continue

        print(f"Найдено {len(urls_to_scrape)} уникальных URL для скрапинга.")

        # 3. Асинхронный скрапинг контента со страниц
        print("Скрапинг контента...")
        scrape_tasks = [scraper.scrape_content(url) for url in urls_to_scrape]
        scraped_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

        scraped_data : List[Dict[str, str]] = []
        successful_urls = []
        for result in scraped_results:
            if isinstance(result, dict) and result.get('text'):
                scraped_data.append(result)
                successful_urls.append(result['url'])
            elif isinstance(result, Exception):
                print(f"Ошибка скрапинга: {result}")

        if not scraped_data:
            print("Не удалось извлечь контент ни с одной страницы на этом шаге.")
            continue

        print(f"Успешно извлечен контент с {len(scraped_data)} страниц.")
        all_sources.extend(successful_urls)

        # 4. Обработка контента и генерация новых направлений
        print("Анализ контента и генерация выводов...")
        combined_text = "\n\n---\n\n".join([item['text'] for item in scraped_data])

        summary_and_directions = await llm.summarize_and_find_directions(
            current_query_context,
            all_learnings,
            combined_text
        )

        new_learnings = summary_and_directions.get("learnings", [])
        next_directions = summary_and_directions.get("directions", [])

        print(f"Новые выводы: {new_learnings}")
        print(f"Следующие направления: {next_directions}")

        all_learnings.extend(new_learnings)
        final_directions = next_directions  # Сохраняем последние направления

    # 6. Формирование итогового отчета
    print("Формирование итогового отчета...")
    final_report = await llm.generate_final_report(initial_query, all_learnings)

    return final_report, list(set(all_sources)), all_learnings, final_directions