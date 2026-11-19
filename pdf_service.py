from pypdf import PdfReader

MAX_PAGES = 300
MAX_CHARS = 600_000
CHUNK_SIZE = 1200
OVERLAP = 180

def extract_pdf(stream):
    try:
        reader = PdfReader(stream)
        if reader.is_encrypted:
            raise ValueError('PDF защищён паролем.')
        if len(reader.pages) > MAX_PAGES:
            raise ValueError(f'Максимум {MAX_PAGES} страниц.')
        result, total = [], 0
        for number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or '').strip()
            total += len(text)
            if total > MAX_CHARS:
                raise ValueError('Слишком много текста (лимит 600 000 символов).')
            if text:
                result.append((number, text))
        return result
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('Не удалось прочитать PDF.') from exc

def chunk_pages(pages):
    chunks = []
    for number, text in pages:
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunks.append((number, text[start:end]))
            if end == len(text):
                break
            start = end - OVERLAP
    return chunks
