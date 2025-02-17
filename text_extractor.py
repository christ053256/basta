import cv2
import pytesseract
import customtkinter as ctk
from tkinter import filedialog
import os
import fitz  # PyMuPDF for PDF handling
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'

# Initialize the app
ctk.set_appearance_mode('dark')
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Image & PDF Text Recognition")
app.iconbitmap('amadeus_5tl_1.ico')
# Flag to prevent resize event handling during initialization
initializing = True


# Toggle fullscreen function
def toggle_fullscreen():
    if app.attributes('-fullscreen'):
        app.attributes('-fullscreen', False)
        fullscreen_button.configure(text="Enter Fullscreen")
    else:
        app.attributes('-fullscreen', True)
        fullscreen_button.configure(text="Exit Fullscreen")


file_path = None


def get_file():
    global file_path
    current_file_path = filedialog.askopenfilename(filetypes=[("Supported Files", "*.png;*.jpg;*.jpeg;*.bmp;*.pdf")])
    if current_file_path:
        file_path = current_file_path
        label.configure(text=f"Selected: {os.path.basename(file_path)}")


def extract_text_from_image(img):
    # Process image for text extraction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    extracted_text = ""

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = img[y:y + h, x:x + w]
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(cropped_gray)
        extracted_text += text + "\n"

    return extracted_text


def extract_text():
    global file_path
    if not file_path:
        label.configure(text="Please select a file first.", text_color="white")
        return

    extracted_text = ""

    # Check if file is PDF or image
    if str(file_path).lower().endswith('.pdf'):
        try:
            pdf_document = fitz.open(file_path)
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)

                # Get page as an image
                pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))

                # Convert to a format OpenCV can process
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

                if pix.n == 4:  # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

                # Extract text from the page image
                page_text = extract_text_from_image(img)
                extracted_text += f"--- Page {page_num + 1} ---\n{page_text}\n"

            pdf_document.close()
        except Exception as e:
            label.configure(text=f"Error processing PDF: {str(e)}", text_color="red")
            return
    else:
        try:
            # For regular images
            img = cv2.imread(str(file_path))
            if img is None:
                label.configure(text="Error: Could not open image.", text_color="red")
                return

            extracted_text = extract_text_from_image(img)
        except Exception as e:
            label.configure(text=f"Error processing image: {str(e)}", text_color="red")
            return

    # Display extracted text in the textbox
    output.delete("1.0", "end")  # Clear previous text
    output.insert("1.0", extracted_text)

    label.configure(text="Successfully extracted the text!", text_color="green")


# color palettes
def extract_color_palette(number_of_clusters=5):
    global file_path
    """
    Extracts the color palette from an image using KMeans clustering.

    Args:
        img_path (str): Path to the image file.
        number_of_clusters (int): The number of colors to extract in the palette.  Defaults to 5.

    Returns:
        list: A list of hex color values representing the extracted color palette.
              Returns None if there's an error loading or processing the image.
    """
    try:
        # Load the image
        image = cv2.imread(str(file_path))
        if image is None:
            print(f"Error: Could not load image from {file_path}")
            return None

        # Convert the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Reshape the image into a list of RGB pixels
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        # Perform KMeans clustering
        clt = KMeans(n_clusters=number_of_clusters, n_init='auto')  # Set n_init explicitly
        clt.fit(image)

        # Get the cluster centroids (colors)
        centroids = clt.cluster_centers_

        # Convert the RGB centroids to hex color values
        hex_colors = [rgb_to_hex(color) for color in centroids]

        return hex_colors

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def rgb_to_hex(rgb):
    """
    Converts an RGB tuple to a hex color code.

    Args:
        rgb (tuple): A tuple representing an RGB color (e.g., (255, 0, 0)).

    Returns:
        str: The hex color code (e.g., "#FF0000").
    """
    rgb = rgb.astype(int)  # Ensure RGB values are integers
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def centroid_histogram(clt):
    """
    Calculates the histogram of pixel cluster assignments.

    Args:
        clt (sklearn.cluster.KMeans): The fitted KMeans clustering object.

    Returns:
        numpy.ndarray: The histogram of cluster assignments.
    """
    # grab the number of different clusters and create a histogram
    # based on the pixel assignments
    num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=num_labels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    """
    Generates a color bar representing the color palette.

    Args:
        hist (numpy.ndarray): The histogram of cluster assignments.
        centroids (numpy.ndarray): The cluster centroids (RGB values).

    Returns:
        numpy.ndarray: A color bar representing the color palette.
    """
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    start_on_x = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        end_on_x = start_on_x + (percent * 300)
        cv2.rectangle(bar, (int(start_on_x), 0), (int(end_on_x), 50),
                      color.astype("uint8").tolist(), -1)
        start_on_x = end_on_x

    # return the bar chart
    return bar


def display_hex_colors():
    hex_colors = extract_color_palette()
    if hex_colors:
        # Display the hex color values in the output textbox
        output.delete("1.0", ctk.END)  # Clear previous text
        output.insert("1.0", "Color Palette (Hex):\n")
        for color in hex_colors:
            output.insert(ctk.END, color + "\n")
        label.configure(text="Successfully extracted the color palette (hex)!", text_color="green")
    else:
        label.configure(text="Error extracting color palette.", text_color="red")


# Function to handle window resizing - with protection against infinite loops
def on_resize(event):
    global initializing

    # Skip resize handling during initialization
    if initializing:
        return

    # Get current window dimensions
    current_width = app.winfo_width()
    current_height = app.winfo_height()

    # Only update if dimensions are reasonable (to prevent loop)
    if current_width > 100 and current_height > 100:
        # Update textbox size when window is resized
        width = current_width - 100  # Keep some margin
        height = current_height - 200  # Keep space for other widgets
        output.configure(width=width, height=height)


# Design - main frame with padding that expands to fill
frame = ctk.CTkFrame(master=app)
frame.pack(pady=20, padx=20, fill="both", expand=True)

# Top control frame for buttons
control_frame = ctk.CTkFrame(master=frame)
control_frame.pack(pady=10, fill="x")

label = ctk.CTkLabel(master=control_frame, text="Select an image or PDF to extract text.", font=("Arial", 14))
label.pack(side="left", padx=10)

fullscreen_button = ctk.CTkButton(master=control_frame, text="Enter Fullscreen", command=toggle_fullscreen)
fullscreen_button.pack(side="right", padx=10)

# Button frame
button_frame = ctk.CTkFrame(master=frame)
button_frame.pack(pady=10, fill="x")

selectButton = ctk.CTkButton(master=button_frame, text="Select File", command=get_file)
selectButton.pack(side="left", padx=10)

extractButton = ctk.CTkButton(master=button_frame, text="Extract Text", command=extract_text)
extractButton.pack(side="left", padx=10)

extractColorsButton = ctk.CTkButton(master=button_frame, text="Extract Colors from Image", command=display_hex_colors)
extractColorsButton.pack(side="left", padx=10)

# Text output that fills the remaining space
output = ctk.CTkTextbox(master=frame)
output.pack(pady=10, padx=10, fill="both", expand=True)

# Move the noteLabel here, above the exitButton
noteLabel = ctk.CTkLabel(master=frame, text="Please click the exit button to close the program.", font=("Arial", 14))
noteLabel.pack(pady=0)

# Exit button should be placed after the noteLabel
exitButton = ctk.CTkButton(master=frame, text="Exit", command=app.quit)
exitButton.pack(pady=0, padx=10)

# Create a key binding to exit fullscreen with Escape key
app.bind("<Escape>", lambda event: app.attributes('-fullscreen', False))

app.mainloop()
