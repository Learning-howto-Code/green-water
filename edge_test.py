import argparse
import os
import sys
import cv2
import numpy as np

# ---------------------------
# Enhanced edge detection
# ---------------------------

def ensure_odd(k):
    k = int(k)
    return k if k % 2 == 1 and k > 0 else max(1, k + 1)

def compute_edges(img, th1, th2, blur_ksize, dilate_iter=2, sobel_thresh=None, strong_factor=1.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = ensure_odd(blur_ksize)
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # Stronger Canny
    low = int(max(0, th1 * strong_factor))
    high = int(max(low + 1, th2 * strong_factor))
    canny = cv2.Canny(gray, low, high)

    # Sobel magnitude
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if sobel_thresh is None:
        _, mag_bin = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mag_bin = cv2.threshold(mag_u8, int(sobel_thresh), 255, cv2.THRESH_BINARY)

    strong_edges = cv2.bitwise_and(canny, mag_bin)

    # Dilate edges
    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        strong_edges = cv2.dilate(strong_edges, kernel, iterations=dilate_iter)

    return strong_edges

# ---------------------------
# Display helpers
# ---------------------------

def overlay_edges_color(image, edges, color=(0, 0, 255), alpha=0.8):
    if image.dtype != np.uint8:
        base = (image * 255).astype(np.uint8)
    else:
        base = image.copy()

    colored = np.zeros_like(base)
    colored[edges > 0] = color

    overlay = base.copy()
    cv2.addWeighted(colored, alpha, overlay, 1.0 - alpha, 0, overlay)
    return overlay

def make_side_by_side(orig, edges_overlay, max_width=None):
    h1 = orig.shape[0]
    h2 = edges_overlay.shape[0]

    if h1 != h2:
        scale = h1 / float(h2)
        new_w = int(edges_overlay.shape[1] * scale)
        edges_overlay = cv2.resize(edges_overlay, (new_w, h1), interpolation=cv2.INTER_AREA)

    combined = np.hstack((orig, edges_overlay))

    if max_width:
        h, w = combined.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            new_size = (int(w * scale), int(h * scale))
            combined = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)

    return combined

# ---------------------------
# CLI & main program
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Show image and enhanced edges side-by-side.")
    p.add_argument("image", help="Path to input image file")
    p.add_argument("--th1", type=int, default=50, help="Canny threshold1")
    p.add_argument("--th2", type=int, default=150, help="Canny threshold2")
    p.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel size (odd)")
    p.add_argument("--max-width", type=int, default=1200, help="Max combined window width")
    return p.parse_args()

def load_image(path):
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to load image: {path}", file=sys.stderr)
        sys.exit(1)
    return img

def main():
    args = parse_args()
    img = load_image(args.image)

    edges = compute_edges(img, args.th1, args.th2, args.blur)

    # Create colored overlay
    edges_overlay = overlay_edges_color(img, edges)

    # Combine for display
    combined = make_side_by_side(img, edges_overlay, max_width=args.max_width)

    window_name = "Original | Enhanced Edges – press s to save"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, combined)

    key = cv2.waitKey(0) & 0xFF
    if key == ord("s"):
        base = os.path.splitext(os.path.basename(args.image))[0]
        cv2.imwrite(f"{base}_edges.png", edges)
        cv2.imwrite(f"{base}_combined.png", combined)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
