from pypdf import PdfReader

MAX_PAGES = 30
MAX_CHARS = 30000

def extract_pdf(stream):
    try:
        reader = PdfReader(stream)
        if reader.is_encrypted:
            raise ValueError('PDF защищён паролем. Загрузите документ без защиты.')
        if len(reader.pages) > MAX_PAGES:
            raise ValueError(f'В первой версии поддерживаются документы до {MAX_PAGES} страниц.')
        result = []
        total = 0
        for i, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or '').strip()
            if not text:
                continue
            if total + len(text) > MAX_CHARS:
                raise ValueError('Документ содержит слишком много текста для версии 1.1. Попробуйте меньший PDF.')
            result.append((i, text))
            total += len(text)
        return result
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('Не удалось прочитать PDF. Убедитесь, что файл не повреждён.') from exc
