import cv2 as cv
import numpy as np
import os

# detect red corners
def get_red_corners(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 

    # Red color ranges
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask = cv.inRange(hsv, lower_red1, upper_red1) + cv.inRange(hsv, lower_red2, upper_red2)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    corners = []
    for cnt in contours:
        M = cv.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            corners.append([cx, cy])

    if len(corners) != 4:
        raise ValueError(f"Expected 4 red corners, found {len(corners)}")
    
    return np.array(corners, dtype=np.float32)

# detect four corners in the hard copy
def get_paper_corners(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    edges = cv.Canny(blur, 50, 150)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in hard copy")
    cnt = max(contours, key=cv.contourArea)
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) != 4:
        raise ValueError(f"Expected 4 corners in hard copy, found {len(approx)}")
    
    return np.array([p[0] for p in approx], dtype=np.float32)

# top left, bottom right, top right and bottom left
def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect

# to extract the table from the hard copy 
def crop_image_edges_cv(img, top=0, bottom=0, left=0, right=0):
    h, w = img.shape[:2]

    # Compute valid crop boundaries
    y1 = top
    y2 = h - bottom
    x1 = left
    x2 = w - right

    # Safety clamp to prevent invalid cropping
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))

    return img[y1:y2, x1:x2]

# to extract cells from the table

def save_table_cells_to_folder(table_img, col_widths, col_names,
                               folder, row_height, thickness):
    """
    table_img  : input table image (numpy array)
    col_widths : list of column widths in pixels
    col_names  : list of column names
    folder     : folder to save images
    row_height : fixed height of each table row (default 57px)
    thickness  : border pixels to remove from all 4 sides (default 5px)
    """
    os.makedirs(folder, exist_ok=True)

    height, width = table_img.shape[:2]
    n_rows = height // row_height  # compute rows using known row height

    for r in range(n_rows):
        # Row cropping
        y1 = r * row_height
        y2 = y1 + row_height
        row_img = table_img[y1:y2, :]

        x_start = 0
        for c, w in enumerate(col_widths):
            x1 = x_start
            x2 = x_start + w
            cell_img = row_img[:, x1:x2]

            # ---- Remove border thickness ----
            t = thickness
            h, w_cell = cell_img.shape[:2]

            cell_img = cell_img[
                t : h - t,
                t : w_cell - t
            ]

            # Save
            filename = os.path.join(folder, f"{col_names[c]}_{r+1}.jpg")
            cv.imwrite(filename, cell_img)

            x_start += w

# to stitch the cells of the particulars column into one
def extract_column(img, filename, distance_from_left, column_width):
    h, w = img.shape[:2]

    x1 = distance_from_left
    x2 = distance_from_left + column_width

    # Ensure safe bounds
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))

    cv.imwrite(filename, img[:, x1:x2])

