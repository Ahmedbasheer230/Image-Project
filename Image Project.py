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
    print("[WARNING] tkinter not available")
    GUI_AVAILABLE = False


# ================= IMAGE PROCESSING =================

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Invalid image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def rotate_90(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def flip_image(img):
    return cv2.flip(img, 1)

def blur_image(img, k):
    return cv2.GaussianBlur(img, (k, k), 0)

def detect_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)

def histogram_equalization(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


# ================= NEW FEATURES =================

def zoom_nearest(img, scale):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)

def zoom_bilinear(img, scale):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

def rotate_custom(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))


# ================= CLI =================

def run_cli():
    print("CLI test...")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img = to_grayscale(img)
    img = rotate_90(img)
    img = flip_image(img)
    img = blur_image(img, 11)
    img = detect_edges(img)
    img = histogram_equalization(img)
    print("Done")


# ================= GUI =================

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
            show_image(img)

    def show_image(img):
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(im)

        canvas.image = imgtk
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=NW, image=imgtk)

        canvas.config(scrollregion=canvas.bbox(ALL))

    def apply(func):
        global current_img
        if current_img is not None:
            current_img = func(current_img)
            show_image(current_img)

    def apply_blur():
        k = blur_slider.get()
        if k % 2 == 0:
            k += 1
        apply(lambda img: blur_image(img, k))

    def reset():
        global current_img
        if original_img is not None:
            current_img = original_img.copy()
            show_image(current_img)

    def save_image():
        if current_img is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if path:
            cv2.imwrite(path, cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR))

    def compare_view():
        if current_img is None:
            return

        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.imshow(current_img)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1,2,2)
        for i, col in enumerate(['r','g','b']):
            hist = cv2.calcHist([current_img],[i],None,[256],[0,256])
            plt.plot(hist, color=col)

        plt.title("Histogram")
        plt.tight_layout()
        plt.show()


    # ================= UI =================

    root = Tk()
    root.title("Advanced Image Tool")
    root.geometry("1100x750")
    root.configure(bg="#1e1e1e")

    # Image Frame + Scroll
    image_frame = Frame(root, bg="#1e1e1e")
    image_frame.pack(side=LEFT, padx=10, pady=10)

    canvas = Canvas(image_frame, bg="black", width=650, height=650)
    canvas.pack(side=LEFT)

    scroll_y = Scrollbar(image_frame, orient="vertical", command=canvas.yview)
    scroll_y.pack(side=RIGHT, fill=Y)

    scroll_x = Scrollbar(root, orient="horizontal", command=canvas.xview)
    scroll_x.pack(side=BOTTOM, fill=X)

    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    # Controls
    control_frame = Frame(root, bg="#2c2c2c")
    control_frame.pack(side=RIGHT, fill=Y, padx=10, pady=10)

    def btn(text, cmd):
        return Button(control_frame, text=text, command=cmd,
                      bg="#3c3f41", fg="white", width=18)

    Label(control_frame, text="Controls",
          bg="#2c2c2c", fg="white").pack(pady=10)

    btn("Open", open_image).pack(pady=2)
    btn("Grayscale", lambda: apply(to_grayscale)).pack(pady=2)
    btn("Rotate 90", lambda: apply(rotate_90)).pack(pady=2)
    btn("Flip", lambda: apply(flip_image)).pack(pady=2)
    btn("Edges", lambda: apply(detect_edges)).pack(pady=2)
    btn("Equalize", lambda: apply(histogram_equalization)).pack(pady=2)

    # Blur
    Label(control_frame, text="Blur", bg="#2c2c2c", fg="white").pack()
    blur_slider = Scale(control_frame, from_=1, to=25,
                        orient=HORIZONTAL, bg="#2c2c2c", fg="white")
    blur_slider.set(11)
    blur_slider.pack()
    btn("Apply Blur", apply_blur).pack(pady=2)

    # Zoom
    Label(control_frame, text="Zoom", bg="#2c2c2c", fg="white").pack()
    zoom_slider = Scale(control_frame, from_=1, to=3,
                        resolution=0.1, orient=HORIZONTAL,
                        bg="#2c2c2c", fg="white")
    zoom_slider.set(1)
    zoom_slider.pack()

    btn("Zoom Nearest",
        lambda: apply(lambda img: zoom_nearest(img, zoom_slider.get()))).pack(pady=2)

    btn("Zoom Bilinear",
        lambda: apply(lambda img: zoom_bilinear(img, zoom_slider.get()))).pack(pady=2)

    # Rotation
    Label(control_frame, text="Angle", bg="#2c2c2c", fg="white").pack()
    rotate_slider = Scale(control_frame, from_=-180, to=180,
                          orient=HORIZONTAL,
                          bg="#2c2c2c", fg="white")
    rotate_slider.set(0)
    rotate_slider.pack()

    btn("Rotate Angle",
        lambda: apply(lambda img: rotate_custom(img, rotate_slider.get()))).pack(pady=2)

    btn("Compare", compare_view).pack(pady=2)
    btn("Reset", reset).pack(pady=2)
    btn("Save", save_image).pack(pady=2)

    root.mainloop()

else:
    run_cli()


# ================= TEST =================

def _test():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert to_grayscale(img).shape == img.shape
    assert rotate_90(img).shape == img.shape
    assert flip_image(img).shape == img.shape
    assert detect_edges(img).shape == img.shape
    print("All tests passed!")

if __name__ == "__main__":
    _test()