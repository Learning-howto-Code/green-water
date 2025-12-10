import argparse
import os
import sys
import cv2
import numpy as np

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