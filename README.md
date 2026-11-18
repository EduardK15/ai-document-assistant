# AI Document Assistant v1.1

Flask app for asking questions about small PDF documents using OpenAI Responses API.

## Features
- Upload PDF up to 5 MB, 30 pages, 30,000 extracted characters
- Extract text with pure-Python `pypdf` (no PyMuPDF)
- Ask a question in Russian and get a document-grounded response with page references
- No permanent file storage (uploads processed in memory)

## Render deployment
1. Upload all files and folders from this directory to the root of your GitHub repository.
2. Render > New > Web Service > Connect GitHub repository.
3. Runtime: Python; Build command: `pip install -r requirements.txt`.
4. Start command: `gunicorn app:app`.
5. Add secret environment variable `OPENAI_API_KEY` with your existing key.
6. Deploy and open the Render URL.

Do not upload `.env` or API keys to GitHub. API usage is billed separately from ChatGPT subscriptions.

## Notes
- This is a first prototype, not yet RAG: it sends extracted text (up to 30k chars) to the model.
- Scanned/image-only PDFs require OCR and are not supported.
- Public deployment has no login or usage quota. Anyone with the URL can generate API charges; add authentication and rate limiting before sharing widely.
- Do not upload sensitive documents; PDF text is transmitted to the OpenAI API.
- Do not use for high-stakes decisions without checking original pages.
