import re
import json

headers = [
            "2. Диагностика", 
            "2.1 Жалобы и анамнез", 
            "2.2 Физикальное обследование", 
            "2.4 Инструментальные диагностические исследования", 
            "2.5 Иные диагностические исследования", 
            "3. Лечение",
            "3.1",
            "3.2",
            "3.3",
            "4. Реабилитация", 
            "5. Профилактика",
            "6. Дополнительная информация, влияющая на течение и исход заболевания",
            "7. Дополнительная информация",
            "7.1",
            "7.2",
            "7.3",
            "Приложение Г"
            ]

def split_document(document):
    header_pattern = re.compile(r'|'.join([fr'\b{header}\b' for header in headers]), re.IGNORECASE)
    
    # Находим позиции всех заголовков
    matches = list(header_pattern.finditer(document))
    
    # Добавляем фиктивный матч в конец документа
    fake_match = type('FakeMatch', (object,), {'start': lambda self: len(document), 'group': lambda self: 'END'})
    matches.append(fake_match())

    # Разделяем документ на части
    sections = {}
    for i in range(len(matches) - 1):
        start = matches[i].start()
        end = matches[i + 1].start()
        header = matches[i].group()
        sections[header] = document[start:end].strip()
    
    return sections