
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

try:
    from tkinter import *
    from tkinter import filedialog
    from PIL import Image, ImageTk
    GUI_AVAILABLE = True
except ModuleNotFoundError:
    print("[WARNING] tkinter is not available. Running in CLI test mode.")
    GUI_AVAILABLE = False

def load_image(path):
    if not os.path.exists(path):
        raise ValueError("File does not exist")
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Invalid image file")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def rotate_image(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def flip_image(img):
    return cv2.flip(img, 1)

def blur_image(img):
    return cv2.GaussianBlur(img, (11, 11), 0)

def detect_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)

def histogram_equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def run_cli():
    print("Running safe CLI test mode...")
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    img = to_grayscale(img)
    img = rotate_image(img)
    img = flip_image(img)
    img = blur_image(img)
    img = detect_edges(img)
    img = histogram_equalization(img)

    print("CLI processing completed successfully.")

if GUI_AVAILABLE:

    original_img = None
    current_img = None

    def open_image():
        global original_img, current_img
        path = filedialog.askopenfilename()
        if path:
            img = load_image(path)
            original_img = img.copy()
            current_img = img.copy()
            show_image(current_img)
            status_var.set(f"Loaded: {os.path.basename(path)}")

    def show_image(img):
        img = cv2.resize(img, (800, 800))
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        image_label.imgtk = imgtk
        image_label.config(image=imgtk)

    def apply(func):
        global current_img
        if current_img is not None:
            current_img = func(current_img)
            show_image(current_img)
            status_var.set("Effect applied")        

    def apply_blur():
        global current_img
        if current_img is not None:
            k = blur_slider.get()
            if k % 2 == 0:
                k += 1
            current_img = cv2.GaussianBlur(current_img, (k, k), 0)
            show_image(current_img)
            status_var.set(f"Blur applied: {k}")

    def reset():
        global current_img
        if original_img is not None:
            current_img = original_img.copy()
            show_image(current_img)
            status_var.set("Reset to original")

    def save_image():
        if current_img is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if path:
            cv2.imwrite(path, cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR))
            status_var.set("Image saved")
    def show_histogram():
        if current_img is None:
            return

        plt.figure("Histogram")

        colors = ('r', 'g', 'b')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([current_img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)

        plt.title("RGB Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")

        plt.show()   
    
    root = Tk()
    root.title("✨ Advanced Image Processing Tool")
    root.geometry("1000x800")
    root.configure(bg="#1e1e1e")

    image_frame = Frame(root, bg="#1e1e1e")
    image_frame.pack(side=LEFT, padx=10, pady=10)
    image_label = Label(image_frame, bg="black")
    image_label.pack()

    control_frame = Frame(root, bg="#2c2c2c")
    control_frame.pack(side=RIGHT, fill=Y, padx=10, pady=10)

    def styled_button(text, cmd):
        return Button(
            control_frame, 
            text=text,
            command=cmd,
            bg="#3c3f41",
            fg="white",
            activebackground="#56595b",
            width=18,
            pady=5
        )

    Label(control_frame, text="Controls", bg="#2c2c2c",
          fg="white", font=("Arial", 14, "bold")).pack(pady=10)

    styled_button("Open Image", open_image).pack(pady=3)
    styled_button("Grayscale", lambda: apply(to_grayscale)).pack(pady=3)
    styled_button("Rotate", lambda: apply(rotate_image)).pack(pady=3)
    styled_button("Flip", lambda: apply(flip_image)).pack(pady=3)
    styled_button("Edges", lambda: apply(detect_edges)).pack(pady=3)
    styled_button("Histogram", lambda: apply(histogram_equalization)).pack(pady=3)
    styled_button("Show Histogram", show_histogram).pack(pady=3)

   
    Label(control_frame, text="Blur Strength",
          bg="#2c2c2c", fg="white").pack(pady=(10, 0))

    blur_slider = Scale(control_frame, from_=1, to=25,
                        orient=HORIZONTAL, bg="#2c2c2c",
                        fg="white")
    blur_slider.set(10)
    blur_slider.pack()

    styled_button("Apply Blur", apply_blur).pack(pady=5)

    styled_button("Reset", reset).pack(pady=3)
    styled_button("Save", save_image).pack(pady=3)

    status_var = StringVar()
    status_var.set("Ready")

    status_bar = Label(root, textvariable=status_var,
                       bg="#111", fg="white", anchor=W)
    status_bar.pack(side=BOTTOM, fill=X)

    root.mainloop()

else:
    run_cli()

def _test():
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    assert to_grayscale(img).shape == img.shape
    assert rotate_image(img).shape == img.shape
    assert flip_image(img).shape == img.shape
    assert blur_image(img).shape == img.shape
    assert detect_edges(img).shape == img.shape
    assert histogram_equalization(img).shape == img.shape
    assert to_grayscale(img).dtype == np.uint8
    edges = detect_edges(img)
    assert np.sum(edges) == 0
    print("All tests passed successfully!")

if __name__ == "__main__":
    _test()