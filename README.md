# Jackfruit: ML-Powered Invoice Digitizer

## The Problem
Many small retail shops continue to maintain handwritten records because the method is simple, familiar, and requires no technical expertise. However, paper bills are difficult to store, search, and analyze over time. Manually re-typing them into a digital format is time-consuming and often avoided. 

Our solution is a machine learning pipeline that converts handwritten shop bills into structured digital data. This enables long-term storage and automated sales analytics while allowing shopkeepers to keep their existing paper-based workflow.

---

## Features
* **Digit Extraction (CNN):** A custom-trained Convolutional Neural Network predicts the handwritten numbers in the quantity and rate columns.
* **Text Extraction (Tesseract OCR):** The "Particulars" (item names) column is passed to Tesseract for offline handwriting recognition.
* **Smart Autocorrect:** Extracted item names are cross-referenced against a predefined dictionary of grocery items using difflib to automatically fix minor spelling mistakes or OCR reading errors.
* **Human-in-the-Loop Correction:** Generates an intermediate Excel file so users can easily review and correct AI extractions before final processing.
* **Data Formatting & Analysis:** The compiled data is aggregated, formatted, and visualized using a Streamlit dashboard.

---

## Model Training & Custom Dataset
Because standard datasets lack certain formatting required for our specific invoice grids, our team manually curated a custom dataset. 
* We wrote, scanned, and segmented approximately 1,900 unique handwritten samples on custom grids. 
* The Convolutional Neural Network (CNN) was built and trained using TensorFlow/Keras.
* We augmented our custom data with the standard MNIST dataset to improve generalization, ultimately achieving over 99% testing accuracy on digit classification.

---

## Engineering Challenges & Decisions
Building a real-world pipeline required navigating several roadblocks and making strict engineering trade-offs:

* **The Camera vs. iPad Pivot (Segmentation Noise):** We originally designed the system for mobile phone photos. However, manual scanning and mobile photography introduced severe shadow noise, lighting inconsistencies, and table border artifacts that ruined OpenCV's segmentation. We pivoted to standardized iPad inputs to isolate the ML model's performance from environmental noise.
* **The Decimal Point Disaster (Catastrophic Forgetting):** Standard MNIST lacks decimal points, so we collected over 500 custom decimal samples. Unfortunately, incorrect OpenCV cropping often left table borders in the images, which the model falsely learned as decimals. Furthermore, trying to append this new class to our pre-trained digit model caused catastrophic forgetting. Due to semester time constraints, we ruthlessly prioritized the core pipeline and excluded the decimal class.
* **Unstable OCR Outputs:** Using Tesseract for handwriting occasionally resulted in garbled or misread grocery items. This required the implementation of a deterministic fallback dictionary to ensure consistent, accurate outputs.

---

## How to Run the Project

**1. Setup and Installation:**
Clone or download the repository to your local machine:

**2. Configure Tesseract:**
Ensure Tesseract-OCR is installed on your machine. You must update the `TESSERACT_CMD_PATH` variable inside `main_program.py` to match your local installation path.

**3. Prepare the Input Image**
* **Use Existing:** You can test the program immediately using the provided `hard_copy_photo.jpg` and `soft_copy_template.jpg`.
* **Use Custom:** If you want to use your own invoice, fill out the template, save it as an image, and place it in the directory. You may need to edit the base template coordinates and change the `num_items` row count variable in the code to match your new invoice's dimensions.

**4. Run the Extraction Pipeline:**
Run the main program to slice the image, run the OCR models, and generate the initial data:
```bash
python main_program.py
```

**5. Manual Review:**
Open the newly generated `initial.xlsx` file. Because no handwriting-recognition model is perfectly accurate, check the extracted serial numbers, particulars, rates, and quantities for any errors. Fix any incorrect values manually and save the file.

**6. Generate Final Output and Dashboard:**
Once the data is verified, run the Streamlit application to aggregate duplicates, calculate totals, and view the final analytical dashboard:
```bash
streamlit run output.py
```

---

## The Team
* Badrinath S Kini
* Akshat P Shanbog
* Aadya Udupa K S
* Bhargava G
