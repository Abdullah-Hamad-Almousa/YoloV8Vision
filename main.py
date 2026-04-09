import cv2
import torch
from PIL import Image
from ultralytics import YOLO
import json
import os
import gc

# Configuration
IMAGE_PATH = "my_image.jpg"
OUTPUT_IMAGE_PATH = "output_with_boxes.jpg"
OUTPUT_JSON_PATH = "detected_elements.json"

# UI element types
UI_ELEMENT_TYPES = [
    'BackgroundImage', 'Bottom_Navigation', 'Card', 'CheckBox', 'Checkbox',
    'CheckedTextView', 'Drawer', 'EditText', 'Icon', 'Image', 'Map', 'Modal',
    'Multi_Tab', 'PageIndicator', 'Remember', 'Spinner', 'Switch', 'Text',
    'TextButton', 'Toolbar', 'UpperTaskBar'
]

def load_model():
    """Load OmniParser YOLO model for CPU"""
    print("Loading OmniParser YOLO model on CPU...")
    weights_dir = "weights"
    yolo_path = f"{weights_dir}/icon_detect/icon_detect/model.pt"

    if not os.path.exists(yolo_path):
        print("Error: YOLO model not found!")
        return None

    yolo_model = YOLO(yolo_path)
    yolo_model.to("cpu")
    yolo_model.eval()
    print("Model loaded on CPU!")
    return yolo_model

def classify_element(bbox, img_width, img_height):
    """
    Classify UI element type based on bounding box characteristics.
    Uses position, size, and aspect ratio heuristics.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / height if height > 0 else 1
    rel_width = width / img_width
    rel_height = height / img_height

    # Normalize Y position (0 = top, 1 = bottom)
    y_center_norm = y1 / img_height
    x_center_norm = (x1 + x2) / 2 / img_width

    # Classification rules based on position and size
    # Bottom navigation bar: wide and at bottom
    if rel_width > 0.3 and y_center_norm > 0.75:
        return 'Bottom_Navigation'

    # Toolbar: at top, wide
    if y_center_norm < 0.15 and rel_width > 0.5:
        return 'Toolbar'

    # Upper task bar: at very top, medium width
    if y_center_norm < 0.08 and rel_width > 0.3 and rel_width < 0.7:
        return 'UpperTaskBar'

    # Card: medium-large area, not at edges
    if 0.05 < rel_width < 0.6 and 0.05 < rel_height < 0.4 and 0.2 < y_center_norm < 0.8:
        if 0.5 < aspect_ratio < 3:
            return 'Card'

    # Page indicator: small, at bottom
    if rel_width < 0.1 and rel_height < 0.05 and y_center_norm > 0.8:
        return 'PageIndicator'

    # Switch: small square or toggle shape at right side
    if 0.05 < rel_width < 0.15 and 0.02 < rel_height < 0.06:
        if x_center_norm > 0.7:
            return 'Switch'

    # Checkbox: small square
    if 0.02 < rel_width < 0.08 and 0.02 < rel_height < 0.08:
        if 0.8 < aspect_ratio < 1.2:
            return 'CheckBox'

    # EditText: wide and short, centered or left-aligned
    if rel_width > 0.2 and 0.02 < rel_height < 0.08:
        if 2 < aspect_ratio < 15:
            return 'EditText'

    # TextButton: medium width, medium height
    if 0.1 < rel_width < 0.4 and 0.03 < rel_height < 0.12:
        if 1 < aspect_ratio < 8:
            return 'TextButton'

    # Spinner/Dropdown: medium width, at right side
    if 0.1 < rel_width < 0.25 and 0.03 < rel_height < 0.08:
        if x_center_norm > 0.6:
            return 'Spinner'

    # Text: small to medium width, thin
    if rel_height < 0.06 and 0.05 < rel_width < 0.5:
        return 'Text'

    # Modal/Dialog: centered, medium-large
    if 0.3 < rel_width < 0.8 and 0.2 < rel_height < 0.6:
        if 0.3 < x_center_norm < 0.7 and 0.3 < y_center_norm < 0.7:
            return 'Modal'

    # Drawer: on left or right edge
    if x_center_norm < 0.15 or x_center_norm > 0.85:
        if 0.2 < rel_height < 0.7:
            return 'Drawer'

    # Tab: at top or middle, wide but thin
    if y_center_norm < 0.2 and 0.1 < rel_width < 0.5 and rel_height < 0.1:
        return 'Multi_Tab'

    # Map: large area
    if rel_width > 0.4 and rel_height > 0.3:
        return 'Map'

    # Default: Icon or Image based on size
    if rel_width > 0.15 and rel_height > 0.15:
        return 'Image'

    return 'Icon'

def detect_elements(image_path, yolo_model):
    """Detect UI elements in the image"""
    print(f"Processing image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_array = cv2.imread(image_path)
    img_height, img_width = image_array.shape[:2]
    print(f"Image size: {img_width}x{img_height}")

    print("Running YOLO detection...")
    results = yolo_model(image_path, verbose=True, device="cpu")

    elements = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            print("No elements detected")
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            yolo_class = yolo_model.names.get(cls_id, 'icon')

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Classify element type
            element_type = classify_element([x1, y1, x2, y2], img_width, img_height)

            # Calculate center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            elements.append({
                "type": element_type,
                "bbox": {
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2
                },
                "center_x": center_x,
                "center_y": center_y,
                "confidence": round(conf, 4)
            })

    return image_array, elements, img_width, img_height

def draw_boxes(image_array, elements):
    """Draw bounding boxes with X_min, Y_min, X_max, Y_max labels"""
    output = image_array.copy()

    colors = {
        'BackgroundImage': (128, 128, 128),
        'Bottom_Navigation': (255, 165, 0),
        'Card': (0, 255, 255),
        'CheckBox': (0, 255, 0),
        'Checkbox': (0, 200, 0),
        'CheckedTextView': (255, 255, 0),
        'Drawer': (255, 0, 255),
        'EditText': (0, 0, 255),
        'Icon': (255, 0, 0),
        'Image': (0, 128, 255),
        'Map': (128, 0, 255),
        'Modal': (255, 128, 0),
        'Multi_Tab': (128, 255, 0),
        'PageIndicator': (255, 128, 128),
        'Remember': (128, 128, 255),
        'Spinner': (0, 255, 128),
        'Switch': (128, 0, 0),
        'Text': (0, 128, 0),
        'TextButton': (0, 0, 128),
        'Toolbar': (255, 0, 128),
        'UpperTaskBar': (128, 255, 128),
    }

    for elem in elements:
        elem_type = elem["type"]
        bbox = elem["bbox"]
        x1, y1, x2, y2 = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
        color = colors.get(elem_type, (255, 255, 255))

        # Draw rectangle
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # Create label with type and all coordinates
        label = f"{elem_type}"
        coord_label = f"X:{x1},{x2} Y:{y1},{y2}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        # Get text sizes
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        (coord_w, coord_h), _ = cv2.getTextSize(coord_label, font, font_scale, thickness)

        max_w = max(label_w, coord_w)
        box_x1 = x1
        box_y1 = y1 - label_h - coord_h - 15

        # Draw background
        cv2.rectangle(output, (box_x1, box_y1), (box_x1 + max_w + 10, y1), color, -1)

        # Draw text (type)
        cv2.putText(output, label, (box_x1 + 5, y1 - coord_h - 8), font, font_scale + 0.1, (255, 255, 255), thickness)

        # Draw coordinates
        cv2.putText(output, coord_label, (box_x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)

        # Draw center point
        cv2.circle(output, (elem['center_x'], elem['center_y']), 4, color, -1)
        cv2.putText(output, f"C:({elem['center_x']},{elem['center_y']})", (elem['center_x'] + 5, elem['center_y'] - 5), font, 0.35, color, 1)

    return output

def save_results(elements, image_path):
    """Save detected elements to JSON"""
    summary = {}
    for elem in elements:
        elem_type = elem["type"]
        summary[elem_type] = summary.get(elem_type, 0) + 1

    output_data = {
        "image": image_path,
        "total_elements": len(elements),
        "elements": elements,
        "summary": summary
    }

    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {OUTPUT_JSON_PATH}")
    return output_data

def print_results(results):
    """Print detection results to console"""
    print("\n" + "="*90)
    print("UI ELEMENT DETECTION RESULTS (OmniParserV2)")
    print("="*90)
    print(f"Image: {results['image']}")
    print(f"Total elements detected: {results['total_elements']}")

    print("\nElement Summary:")
    for elem_type, count in results['summary'].items():
        print(f"  {elem_type}: {count}")

    print("\nDetailed Positions:")
    print("-"*90)
    print(f"{'Type':<18} {'X_min':<8} {'Y_min':<8} {'X_max':<8} {'Y_max':<8} {'Center_X':<10} {'Center_Y':<10} {'Conf':<8}")
    print("-"*90)
    for elem in results['elements']:
        bbox = elem['bbox']
        print(f"{elem['type']:<18} {bbox['x_min']:<8} {bbox['y_min']:<8} {bbox['x_max']:<8} {bbox['y_max']:<8} {elem['center_x']:<10} {elem['center_y']:<10} {elem['confidence']:<8.4f}")
    print("="*90)

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image '{IMAGE_PATH}' not found!")
        return

    yolo_model = load_model()
    if yolo_model is None:
        return

    image_array, elements, img_width, img_height = detect_elements(IMAGE_PATH, yolo_model)

    output_image = draw_boxes(image_array, elements)
    cv2.imwrite(OUTPUT_IMAGE_PATH, output_image)
    print(f"Output image saved to {OUTPUT_IMAGE_PATH}")

    results = save_results(elements, IMAGE_PATH)
    print_results(results)

if __name__ == "__main__":
    main()
