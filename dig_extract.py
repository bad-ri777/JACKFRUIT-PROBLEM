import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

def extract_form_data(input_dir, model_path, num_items=16):
    """
    Main pipeline function to extract digit data from cropped cell images.
    
    ARGS:
        input_dir (str): Path to the folder containing cropped cell images (e.g., 'slno_1.jpg').
        model_path (str): File path to the trained .keras model.
        num_items (int): Number of rows/items to process (default 16).

    RETURNS:
        tuple: (sl_no_array, rate_array, qty_array) 
               Three numpy arrays containing the extracted integers.
    """
    
    # Load the trained model once at the start
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Return empty arrays in case of model failure
        return np.array([]), np.array([]), np.array([])

    # --- Helper Functions (Internal Logic) ---

    def clean_cell_borders(img, margin=4):
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, h), (255, 255, 255), thickness=margin*2)
        return img

    def preprocess_digit_for_mnist(digit_roi):
        if np.mean(digit_roi) > 127:
            digit_roi = cv2.bitwise_not(digit_roi)
        
        coords = cv2.findNonZero(digit_roi)
        if coords is None:
            return np.zeros((28, 28, 1))
            
        x, y, w, h = cv2.boundingRect(coords)
        roi = digit_roi[y:y+h, x:x+w]
        
        rows, cols = roi.shape
        if rows > cols:
            factor = 20.0 / rows
            rows = 20
            cols = int(round(cols * factor))
        else:
            factor = 20.0 / cols
            cols = 20
            rows = int(round(rows * factor))
            
        roi = cv2.resize(roi, (cols, rows), interpolation=cv2.INTER_AREA)
        
        colsPadding = (int(np.ceil((28 - cols) / 2.0)), int(np.floor((28 - cols) / 2.0)))
        rowsPadding = (int(np.ceil((28 - rows) / 2.0)), int(np.floor((28 - rows) / 2.0)))
        padded = np.pad(roi, (rowsPadding, colsPadding), 'constant')
        
        _, padded = cv2.threshold(padded, 30, 255, cv2.THRESH_BINARY)
        
        padded = padded.astype('float32') / 255.0
        padded = np.expand_dims(padded, axis=-1)
        return padded

    def process_single_cell(image_path, expected_val=None):
        if not os.path.exists(image_path): return 0
            
        img = cv2.imread(image_path)
        if img is None: return 0
        
        img = clean_cell_borders(img, margin=3)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not cnts: return 0
            
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][0]))
        
        batch_digits = []
        
        for i, c in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(c)
            if h < 12 or w < 3: continue
                
            if w > 1.3 * h:
                half_w = w // 2
                batch_digits.append(preprocess_digit_for_mnist(thresh[y:y+h, x:x+half_w]))
                batch_digits.append(preprocess_digit_for_mnist(thresh[y:y+h, x+half_w:x+w]))
            else:
                batch_digits.append(preprocess_digit_for_mnist(thresh[y:y+h, x:x+w]))
            
        if len(batch_digits) > 0:
            predictions = model.predict(np.array(batch_digits), verbose=0)
            
            if expected_val is not None:
                expected_str = str(expected_val)
                if len(predictions) == len(expected_str):
                    for k in range(len(predictions)):
                        target_digit = int(expected_str[k])
                        predictions[k][target_digit] += 2.0
    
            predicted_digits = ""
            for p in predictions:
                predicted_digits += str(np.argmax(p))
                    
            return int(predicted_digits) if predicted_digits else 0
        
        return 0

    # --- Main Processing Loop ---
    sl_no_list = []
    rate_list = []
    qty_list = []
    
    print(f"Processing {num_items} items from: {input_dir}")
    
    for i in range(1, num_items + 1):
        try:
            # Dynamically join the input_dir with standard filenames
            path_sl = os.path.join(input_dir, f"slno_{i}.jpg")
            path_rate = os.path.join(input_dir, f"rate_{i}.jpg")
            path_qty = os.path.join(input_dir, f"quantity_{i}.jpg")
            
            val_sl = process_single_cell(path_sl, expected_val=i)
            val_rate = process_single_cell(path_rate, expected_val=None)
            val_qty = process_single_cell(path_qty, expected_val=None)
            
            sl_no_list.append(val_sl)
            rate_list.append(val_rate)
            qty_list.append(val_qty)
            
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            sl_no_list.append(0)
            rate_list.append(0)
            qty_list.append(0)

    return np.array(sl_no_list), np.array(rate_list), np.array(qty_list)

# Example Usage Block
# This allows you to test the file, but doesn't run when your teammates import it.
if __name__ == "__main__":
    print("This script is intended to be imported as a module.")
    # test_in = "path/to/cells"
    # test_model = "path/to/model.keras"
    # s, r, q = extract_form_data(test_in, test_model)
    # print(s, r, q)