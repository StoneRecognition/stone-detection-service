from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import cv2


CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "weight/groundingdino_swint_ogc.pth"
DEVICE = "cuda"
IMAGE_PATH = "data/raw/1.jpg"
TEXT_PROMPT = "stone. rock. pebble."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
FP16_INFERENCE = False  # Disabled to avoid type mismatch

image_source, image = load_image(IMAGE_PATH)
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

# Force model to float32 and move to device
model = model.float().to(DEVICE)

if FP16_INFERENCE:
    image = image.half()
    model = model.half()

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=DEVICE,
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("outputs/grounding_dino/annotated_image.jpg", annotated_frame)
print(f"Detected {len(boxes)} objects. Output saved to outputs/grounding_dino/annotated_image.jpg")