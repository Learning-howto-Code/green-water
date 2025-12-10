import argparse
import os
import sys
import cv2
import numpy as np
# Enhanced edge detection and overlay helpers
def compute_edges(img, th1, th2, blur_ksize, dilate_iter=2, sobel_thresh=None, strong_factor=1.5):
    """
    Compute stronger, more "major" edges by combining Canny with Sobel magnitude
    and then dilating the result to make edges more visible.

    - th1, th2: base Canny thresholds (will be scaled by strong_factor)
    - blur_ksize: gaussian blur kernel size (odd)
    - dilate_iter: how many times to dilate (thickness)
    - sobel_thresh: optional explicit threshold for Sobel magnitude; if None uses Otsu
    - strong_factor: scales up Canny thresholds to favor major edges
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = ensure_odd(blur_ksize)
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # stronger Canny to prefer major edges
    low = int(max(0, th1 * strong_factor))
    high = int(max(low + 1, th2 * strong_factor))
    canny = cv2.Canny(gray, low, high)

    # Sobel gradient magnitude to measure "strength" of edges
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if sobel_thresh is None:
        _, mag_bin = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mag_bin = cv2.threshold(mag_u8, int(sobel_thresh), 255, cv2.THRESH_BINARY)

    # keep only canny edges that are also strong in Sobel magnitude
    strong_edges = cv2.bitwise_and(canny, mag_bin)

    # optionally dilate to make edges more visible
    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        strong_edges = cv2.dilate(strong_edges, kernel, iterations=dilate_iter)

    return strong_edges


def overlay_edges_color(image, edges, color=(0, 0, 255), alpha=0.8):
    """
    Return a BGR image with colored edges blended onto the original.

    - color: BGR tuple for edge color (default red)
    - alpha: blend weight for the colored edges (0..1). Larger means edges are more dominant.
    """
    if image.dtype != np.uint8:
        base = (image * 255).astype(np.uint8)
    else:
        base = image.copy()

    # create a colored layer where edges are present
    colored = np.zeros_like(base)
    mask = edges > 0
    colored[mask] = color

    # blend the colored edges onto the base image
    overlay = base.copy()
    cv2.addWeighted(colored, alpha, overlay, 1.0 - alpha, 0, overlay)
    return overlay


def make_side_by_side(orig, edges_img, max_width=None):
    """
    Stack the plain color image (left) and the color-overlayed image (right) side-by-side.
    edges_img here is expected to be a BGR image already (overlay result). This function
    will resize the combined image if max_width is provided.
    """
    # ensure same height
    h1 = orig.shape[0]
    h2 = edges_img.shape[0]
    if h1 != h2:
        # resize second to match first height
        scale = h1 / float(h2)
        new_w = int(edges_img.shape[1] * scale)
        edges_img = cv2.resize(edges_img, (new_w, h1), interpolation=cv2.INTER_AREA)

    combined = np.hstack((orig, edges_img))

    if max_width:
        h, w = combined.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            new_size = (int(w * scale), int(h * scale))
            combined = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)

    return combined
#!/usr/bin/env python3
"""
edge_test.py

Usage:
    python edge_test.py /path/to/image.jpg

Displays the original image and an edge-detected version side-by-side.
Press any key to exit. Press "s" to save the edges to disk (edges.png).
"""




def parse_args():
    p = argparse.ArgumentParser(description="Show image and Canny edges side-by-side.")
    p.add_argument("image", help="Path to input image file")
    p.add_argument("--th1", type=int, default=50, help="Canny threshold1 (low)")
    p.add_argument("--th2", type=int, default=150, help="Canny threshold2 (high)")
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


def ensure_odd(k):
    k = int(k)
    return k if k % 2 == 1 and k > 0 else max(1, k + 1)


def compute_edges(img, th1, th2, blur_ksize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = ensure_odd(blur_ksize)
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(gray, th1, th2)
    return edges


def make_side_by_side(orig, edges, max_width=None):
    # convert edges to BGR so stacking works
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((orig, edges_bgr))
    if max_width:
        h, w = combined.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            new_size = (int(w * scale), int(h * scale))
            combined = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)
    return combined


def main():
    args = parse_args()
    img = load_image(args.image)
    edges = compute_edges(img, args.th1, args.th2, args.blur)
    combined = make_side_by_side(img, edges, max_width=args.max_width)

    window_name = "Original (left) | Edges (right) - press s to save, any other key to exit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, combined)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("s"):
        # save edges and combined
        base = os.path.splitext(os.path.basename(args.image))[0]
        edges_path = f"{base}_edges.png"
        combined_path = f"{base}_combined.png"
        cv2.imwrite(edges_path, edges)
        cv2.imwrite(combined_path, combined)
        print(f"Saved: {edges_path}, {combined_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()