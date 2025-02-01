# Unused
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import models, transforms

from constants import SYNTHETIC_IMAGES_DIR


model = models.segmentation.deeplabv3_resnet101(
    weights="DeepLabV3_ResNet101_Weights.DEFAULT"
)
model.eval()


def preprocess_image(image_path):
    input_image = Image.open(image_path)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform(input_image).unsqueeze(0)
    return input_tensor, input_image


def preprocess_mask(output, input_image):
    output = output["out"][0]
    output_predictions = output.argmax(0)

    person_class = 15
    mask = (output_predictions == person_class).cpu().numpy().astype(np.uint8) * 255

    mask = cv2.resize(mask, (input_image.width, input_image.height), interpolation=cv2.INTER_NEAREST)
    unique_values = np.unique(mask)
    print(
        f"Mask unique values: {unique_values}"
    )  # Should contain 0 and 255 if the person class is detected

    return mask


def blur_background(image_path, mask, blur_strength=35):
    img = cv2.imread(image_path)

    blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)
    mask_inv = cv2.bitwise_not(mask)

    background = cv2.bitwise_and(blurred, blurred, mask=mask_inv)
    foreground = cv2.bitwise_and(img, img, mask=mask)
    combined = cv2.add(foreground, background)

    return combined


def main():
    image_path = f"{SYNTHETIC_IMAGES_DIR}/seed0003.png"
    input_tensor, input_image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)

    mask = preprocess_mask(output, input_image)
    blurred_image = blur_background(image_path, mask)
    print(blurred_image)
    status = cv2.imwrite("./seed0003.png", blurred_image)
    print(status)


if __name__ == "__main__":
    main()
