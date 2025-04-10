import numpy as np
import supervision as sv


def draw_image(image_rgb, masks, xyxy, probs, labels):
    box_annotator = sv.BoxCornerAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    # Create class_id for each unique label
    unique_labels = list(set(labels))
    class_id_map = {label: idx for idx, label in enumerate(unique_labels)}
    class_id = [class_id_map[label] for label in labels]

    # Add class_id to the Detections object
    detections = sv.Detections(
        xyxy=xyxy,
        mask=masks.astype(bool),
        confidence=probs,
        class_id=np.array(class_id),
    )
    annotated_image = box_annotator.annotate(scene=image_rgb.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
    return annotated_image

def keep_mask_regions_only(image_rgb, masks):
    # Create a blank image with the same dimensions as the input image, filled with the color (127, 127, 127)
    blank_image = np.full_like(image_rgb, fill_value=127, dtype=image_rgb.dtype)

    # Loop over each mask and copy the masked regions from the original image to the blank image
    for mask in masks:
        mask = mask.astype(bool)
        blank_image[mask] = image_rgb[mask]

    return blank_image

def create_rgba_masked_image(image_rgb, masks):
    # Initialize an RGBA image with alpha channel set to 0 (fully transparent)
    h, w, _ = image_rgb.shape
    rgba_image = np.zeros((h, w, 4), dtype=image_rgb.dtype)

    # Loop over each mask and fill the corresponding areas in the RGBA image
    for mask in masks:
        mask = mask.astype(bool)
        rgba_image[mask, :3] = image_rgb[mask]  # Copy RGB colors
        rgba_image[mask, 3] = 255  # Set alpha to 255 (fully opaque) in the masked regions

    return rgba_image
