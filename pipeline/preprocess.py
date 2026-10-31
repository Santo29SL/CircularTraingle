import cv2
import numpy as np

def extract_silhouette(image_path: str) -> np.ndarray:
    """Reads a sketch and returns a filled binary silhouette."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load {image_path}")
        
    # Invert if the image is dark-on-light (white background)
    if np.mean(img) > 127:
        img = 255 - img
        
    # Create binary mask and fill internal holes to create a solid object
    _, binary = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    
    return mask