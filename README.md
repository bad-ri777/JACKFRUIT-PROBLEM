# JACKFRUIT-PROBLEM
A simple OCR-powered tool that digitizes handwritten shop records and turns them into searchable data. Our project helps small shop owners view summaries, generate charts, and manage their daily numbers with ease.

### Digit Extraction Module
* **`digit_extractor.py`**: This module contains the core OCR pipeline. It accepts a folder of cropped form cells, preprocesses the images to match MNIST standards, and aggregates the recognized digits into structured NumPy arrays (Serial No, Rate, Quantity) for further processing.
* **`mnist_high_acc_model.keras`**: A pre-trained Convolutional Neural Network (CNN) optimized for digit classification. This model serves as the inference engine used by the extractor to identify numerical data with high accuracy.
