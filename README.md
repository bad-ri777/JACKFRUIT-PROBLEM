# JACKFRUIT-PROBLEM
A simple OCR-powered tool that digitizes handwritten shop records and turns them into searchable data. Our project helps small shop owners view summaries, generate charts, and manage their daily numbers with ease.

### Handwritten Particulars Recognition
* **`main.py`**: Interfaces with the Gemini API to convert handwritten item descriptions into clean text. It processes the cropped and stiched “Particulars” cells and returns structured strings aligned with each row of the Excel output.

### Streamlit Output Application
* **`mains.py`**: A lightweight Streamlit interface that loads the generated final Excel sheet and displays the processed bill data in a user-friendly format. Provides summaries, computed totals, and organized tables for easy review.
