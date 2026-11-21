from flask import Flask, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from pdf_service import extract_pdf
from ai_service import answer_question
import mistune
import bleach
from markupsafe import Markup

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024
markdown = mistune.create_markdown(escape=True, plugins=['table'])

def format_answer(answer):
    return Markup(bleach.clean(markdown(answer), tags={'p','br','strong','em','ul','ol','li','h1','h2','h3','h4','h5','h6','hr','blockquote','code','pre','table','thead','tbody','tr','th','td'}, attributes={'ol':['start']}, strip=True))

UI = {
'en': dict(intro='Upload a PDF and ask a question about its contents.', pdf='PDF (up to 15 MB, 300 pages)', question='Your question', placeholder='What is this document about?', ask='Ask AI', answer='AI answer', note='Demo project. Do not upload confidential documents. Extracted PDF text is sent to the OpenAI API. Scanned PDFs are not supported yet.', file_error='Choose a PDF file.', question_error='Enter a question (up to 1000 characters).', empty='No text found. Scanned documents require OCR.', invalid='Unable to read PDF. Check encryption, page limit (300), and text size.', failed='PDF or AI processing failed. Check Render logs.', large='Maximum upload size is 15 MB.', language='Language'),
'he': dict(intro='העלו מסמך PDF ושאלו שאלה על תוכנו.', pdf='PDF (עד 15 מ״ב, 300 עמודים)', question='השאלה שלכם', placeholder='על מה המסמך הזה?', ask='שאלו את AI', answer='תשובת AI', note='פרויקט הדגמה. אין להעלות מסמכים חסויים. טקסט שחולץ מהמסמך נשלח ל-OpenAI API. מסמכים סרוקים עדיין אינם נתמכים.', file_error='בחרו קובץ PDF.', question_error='הזינו שאלה (עד 1000 תווים).', empty='לא נמצא טקסט. מסמכים סרוקים דורשים OCR.', invalid='לא ניתן לקרוא את המסמך. בדקו הצפנה, מגבלת עמודים (300) וגודל הטקסט.', failed='שגיאה בעיבוד המסמך או AI. בדקו את יומני Render.', large='גודל הקובץ המרבי הוא 15 מ״ב.', language='שפה'),
'ru': dict(intro='Загрузите PDF и задайте вопрос по его содержимому.', pdf='PDF (до 15 МБ, 300 страниц)', question='Ваш вопрос', placeholder='О чём этот документ?', ask='Спросить AI', answer='Ответ AI', note='Демонстрационный проект. Не загружайте конфиденциальные документы. Извлечённый текст PDF отправляется в OpenAI API. Сканированные PDF пока не поддерживаются.', file_error='Выберите PDF-файл.', question_error='Введите вопрос (до 1000 символов).', empty='Текст не найден. Для сканов нужен OCR.', invalid='Не удалось прочитать PDF. Проверьте защиту, лимит страниц (300) и объём текста.', failed='Ошибка обработки PDF или AI. Проверьте логи Render.', large='Максимальный размер загрузки — 15 МБ.', language='Язык')
}

def language():
    lang = request.args.get('lang', 'en')
    return lang if lang in UI else 'en'

def page(**kwargs):
    lang = language()
    return render_template('index.html', lang=lang, t=UI[lang], **kwargs)

@app.route('/', methods=['GET', 'POST'])
def index():
    lang = language()
    t = UI[lang]
    if request.method == 'GET':
        return page()
    file = request.files.get('pdf')
    question = (request.form.get('question') or '').strip()
    if not file or not file.filename.lower().endswith('.pdf'):
        return page(error=t['file_error'], question=question), 400
    if not question or len(question) > 1000:
        return page(error=t['question_error'], question=question), 400
    try:
        pages = extract_pdf(file.stream)
    except ValueError:
        return page(error=t['invalid'], question=question), 400
    except Exception:
        app.logger.exception('PDF extraction failed')
        return page(error=t['failed'], question=question), 500
    if not pages:
        return page(error=t['empty'], question=question), 400
    try:
        answer = answer_question(pages, question, lang)
        return page(answer=answer, answer_html=format_answer(answer), question=question, filename=file.filename)
    except Exception:
        app.logger.exception('Document processing failed')
        return page(error=t['failed'], question=question), 500

@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return page(error=UI[language()]['large']), 413

if __name__ == '__main__':
    app.run(debug=False)
