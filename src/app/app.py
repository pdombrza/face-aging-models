from enum import Enum
from pathlib import Path

import gradio as gr
from PIL import Image
import torch
import torchvision.transforms as transforms
from models.FRAN.fran import Generator


class Model(Enum):
    FRAN = "FRAN"
    CycleGAN = "CycleGAN"
    Diffusion = "Diffusion"

class ModelManager:
    cache: dict[Model, torch.nn.Module] = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def setup_model(checkpoint_path: Path | str = Path("models/fran/fran_final2.ckpt"), model_type: Model = Model.FRAN) -> torch.nn.Module:
        model = torch.load(checkpoint_path)
        if model_type == Model.FRAN:
            weights = {k.replace("generator.", "", 1): v for k, v in model["state_dict"].items() if k.startswith("generator.")}
            generator_model = Generator()
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented.")
        generator_model.load_state_dict(weights)
        generator_model.to(ModelManager.device)
        return generator_model

    @staticmethod
    def get_model(model_type: Model, checkpoint_path: Path | str = Path("models/fran/checkpoints/fran_epoch=04.ckpt")) -> torch.nn.Module:
        if model_type not in ModelManager.cache:
            ModelManager.cache[model_type] = ModelManager.setup_model(checkpoint_path, model_type)
        return ModelManager.cache[model_type]


def images_to_gif(images):
    path = Path("examples/temp.gif")
    images[0].save(
        path, format="GIF", save_all=True, append_images=images[1:], loop=0, duration=500
    )
    return path



def generate_output(input_image, input_age, input_gender, model_type):
    model = ModelManager.get_model(Model(model_type))
    model.eval()
    if model_type == "FRAN":
        target_ages = [13, 18, 23, 33, 38, 48, 53, 58, 63, 68, 73, 78]
        transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
        input_image_tensor = transform(input_image)
        input_age_embedding = torch.full((1, input_image_tensor.shape[1], input_image_tensor.shape[2]), input_age / 100)
        out_image_array = []
        for i, target_age in enumerate(target_ages):
            target_age_embedding = torch.full((1, input_image_tensor.shape[1], input_image_tensor.shape[2]), target_age / 100)
            input_tensor = torch.cat((input_image_tensor, input_age_embedding, target_age_embedding), dim=0)
            with torch.no_grad():
                output = model(input_tensor.unsqueeze(0).to(ModelManager.device))
            new_image_out = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
            min_val, max_val = new_image_out.min(), new_image_out.max()
            new_image_normalized  = 255 * (new_image_out - min_val) / (max_val - min_val)
            out_image_array.append(Image.fromarray(new_image_normalized.astype("uint8")))
        gif = images_to_gif(out_image_array)
        return gif
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented.")


def run_app():
    input_image = gr.Image(label="Input Image", type="pil")
    input_age = gr.Number(label="Input Age", minimum=18, maximum=83, value=18)
    input_gender = gr.Radio(["Male", "Female"], label="Specify your gender")
    input_model = gr.Dropdown([model.value for model in Model], label="Select model", value=Model.FRAN.value)
    output_gif = gr.Image(label="Aging process", streaming=True)
    interface = gr.Interface(
        fn=generate_output,
        inputs=[input_image, input_age, input_gender, input_model],
        outputs=output_gif,
        title="Age progression timelapse demo",
        description="Upload an image, specify your age and gender and choose your model",
    )
    interface.launch()


def main():
    run_app()


if __name__ == "__main__":
    main()

