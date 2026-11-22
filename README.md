
# AI Document Assistant

An AI-powered web application for analyzing PDF documents, extracting information, and answering questions with page references.

## Live Demo

https://ai-document-assistant-qlno.onrender.com/

## Key Features

- PDF processing: up to 300 pages and 15 MB.
- Document-grounded question answering.
- Hybrid semantic and keyword search.
- Retrieval-Augmented Generation (RAG).
- Query decomposition for complex questions.
- Query routing for different types of requests.
- Map-Reduce analysis for large documents.
- AI answer verification.
- Page references for source verification.
- Document summaries and structured tables.
- English, Hebrew, and Russian language support.
- Responsive interface for desktop, tablet, and mobile devices.

## Technology Stack

- Python
- Flask
- OpenAI Responses API
- OpenAI Embeddings API
- pypdf
- HTML, CSS, JavaScript
- Gunicorn
- GitHub
- Render

## AI Architecture

1. The user uploads a PDF and submits a question.
2. The application validates the file and extracts text.
3. Query routing selects the appropriate analysis strategy.
4. Specific questions are processed using hybrid RAG.
5. General document analysis uses Map-Reduce.
6. Relevant information is processed by the AI model.
7. The application generates a document-grounded answer.
8. The result is displayed with page references.

## Testing and Evaluation

The application has been tested with:

- Technical manuals.
- Legal contracts.
- Documents containing tables.
- Large documents with 200 pages.
- Questions in English, Hebrew, and Russian.
- Mobile and tablet interfaces.

### Large Document Test

A synthetic 200-page document containing eight sections was used to evaluate document overview capabilities.

| Metric | Initial RAG | After Map-Reduce |
|---|---:|---:|
| Sections identified | 2/8 | 8/8 |
| Section coverage | 25% | 100% |

These results apply to the specific test document and do not represent overall model accuracy.

## Project Structure

- app.py — Flask application and request handling.
- ai_service.py — AI integration and document analysis.
- pdf_service.py — PDF text extraction and chunking.
- templates/ — HTML interface.
- static/ — CSS styles.
- requirements.txt — Python dependencies.

## Local Installation

1. Clone the repository.
2. Install dependencies:

   pip install -r requirements.txt

3. Configure the OPENAI_API_KEY environment variable.

4. Run the application:

   python app.py

5. Open http://127.0.0.1:5000 in your browser.

## Deployment

The application is deployed on Render using Gunicorn and GitHub integration.

API keys are configured through environment variables and must never be committed to the repository.

## Current Limitations

- Scanned and image-only PDFs require OCR, which is not currently supported.
- AI-generated answers may contain inaccuracies.
- Processing time depends on document size and question complexity.
- Public deployment requires authentication and rate limiting before production use.
- Extracted document text is transmitted to the OpenAI API.
- Sensitive or confidential documents should not be uploaded.

## Future Improvements

- OCR support.
- Document caching.
- Multi-document analysis.
- User authentication.
- API usage monitoring and rate limiting.
- Expanded automated evaluation.

## About the Project

This portfolio project demonstrates practical experience in AI integration, API communication, RAG architecture, prompt engineering, document processing, testing, and cloud deployment.

## Author

Eduard Kritman

AI Integration Portfolio Project
