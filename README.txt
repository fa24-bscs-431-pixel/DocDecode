# Official Report Reader

Official letters, bills and legal notices are hard to understand, and missed
deadlines are expensive. Official Report Reader turns a photo or PDF of any
official document into plain words, a deadline list and clear next steps.

## Features
- **Plain-language summary** of what the document is, who sent it and what it wants
- **Deadline detection** with a tracker and calendar (.ics) reminders
- **Legal guidance** tailored to your country, with possible scenarios and the
  typical process (general information, not legal advice)
- **Multilingual output**, including Urdu, Hindi and Arabic
- **Multimodal input**: JPG, PNG, WEBP and PDF
- **Follow-up questions** about the document

## Built with
- Gemma 4 through the Gemini API
- Python and Streamlit

## Getting started
1. Clone the repo and install dependencies:
   `pip install -r requirements.txt`
2. Create a `.env` file in the project folder:
   `GEMINI_API_KEY=your-key-here`
   (get a free key at https://aistudio.google.com/apikey)
3. Run the app:
   `python -m streamlit run app.py`

## Privacy and limitations
- Uploaded documents are sent to Google's Gemini API for processing.
- Output can contain mistakes, especially with blurry images. Always check
  dates and amounts against the original.
- This tool provides general information only and is not a substitute for
  a qualified lawyer.

## Why an open model?
Gemma is open-weights, so the same approach can later run on self-hosted
infrastructure, keeping sensitive documents on your own servers.

## License
MIT