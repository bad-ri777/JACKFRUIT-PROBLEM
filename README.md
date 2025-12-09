# JACKFRUIT-PROBLEM
A simple OCR-powered tool that digitizes handwritten shop records and turns them into searchable data. Our project helps small shop owners view summaries, generate charts, and manage their daily numbers with ease.

### Excel Generation Module
* **`add_columns_to_excel.py`**: Takes the integer values produced by the digit extraction module and updates the existing Excel sheet containing the Gemini-generated Particulars, combining both into a single, fully aligned spreadsheet.
* **`excel_to_csv.py`**: Converts the Excel sheet generated in the previous step and converts it to a CSV file for further processing.
* **`excel_to_csv.py`**: Converts the modified CSV file to an Excel sheet - adds inventory management and adds formatting.
