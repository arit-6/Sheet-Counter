import cv2
import numpy as np
import streamlit as st
from PIL import Image

def count_sheets(image):
    # Convert the image to a numpy array
    image = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply contrast adjustment
    alpha = 1.65  # Contrast control
    beta = 0     # Brightness control
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)

    # Apply edge detection
    global edges
    edges = cv2.Canny(blurred , 55, 145)
    

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours that are likely to be sheets
    sheet_contours = []
    for contour in contours:
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        # Check if the contour is a rectangle (has 4 sides) and is of significant size
        if len(approx) == 4 and cv2.contourArea(contour) > 0:
            sheet_contours.append(contour)

    # Count the number of sheet contours
    num_sheets = len(sheet_contours)

    return num_sheets

def main():
    st.title("Sheet Counter")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV image
        image = Image.open(uploaded_file)

        # Perform sheet counting
        num_sheets = count_sheets(image)
        
        st.markdown(f"<h2 style='text-align: center; color: black;'>Number of sheets: {num_sheets}</h2>", unsafe_allow_html=True)
        # st.image(edges, caption='Edges', use_column_width=True)
if __name__ == "__main__":
    main()
