import os
from flask import Flask, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from pdf_service import extract_pdf
from ai_service import answer_question

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    file = request.files.get('pdf')
    question = (request.form.get('question') or '').strip()
    if not file or not file.filename.lower().endswith('.pdf'):
        return render_template('index.html', error='Выберите PDF-файл.'), 400
    if not question or len(question) > 1000:
        return render_template('index.html', error='Введите вопрос (до 1000 символов).'), 400
    try:
        pages = extract_pdf(file.stream)
        if not pages:
            return render_template('index.html', error='Текст не найден. Для сканов нужен OCR.'), 400
        answer = answer_question(pages, question)
        return render_template('index.html', answer=answer, question=question, filename=file.filename)
    except ValueError as exc:
        return render_template('index.html', error=str(exc)), 400
    except Exception:
        app.logger.exception('Document processing failed')
        return render_template('index.html', error='Ошибка обработки PDF или AI. Проверьте логи Render.'), 500

@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return render_template('index.html', error='Максимальный размер загрузки — 15 МБ.'), 413

if __name__ == '__main__':
    app.run(debug=False)
