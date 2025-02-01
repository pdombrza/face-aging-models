from enum import Enum
from pathlib import Path

import gradio as gr
from PIL import Image
import torch
import torchvision.transforms as transforms
from src.models.FRAN.fran import Generator
from src.models.CycleGAN.train_cycle_gan import CycleGAN
from src.models.diffusion.diffusion import DiffusionModel


class Model(Enum):
    FRAN = "FRAN"
    CycleGAN = "CycleGAN"
    Diffusion = "Diffusion"


class ModelManager:
    cache: dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup_model(
        model_type: Model = Model.FRAN,
        checkpoint_path: Path | str = Path("models/fran/fran_synthetic_512.pth"),
    ) -> torch.nn.Module:
        if model_type == Model.FRAN:
            model = torch.load(checkpoint_path)
            weights = {
                k.replace("generator.", "", 1): v
                for k, v in model["state_dict"].items()
                if k.startswith("generator.")
            }
            generator_model = Generator()
            generator_model.load_state_dict(weights)
        elif model_type == Model.CycleGAN:
            generator_model = CycleGAN.load_from_checkpoint(checkpoint_path)
        elif model_type == Model.Diffusion:
            generator_model = DiffusionModel.load_from_checkpoint(checkpoint_path)
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented.")
        generator_model.to(ModelManager.device)
        return generator_model

    @staticmethod
    def get_model(
        model_type: str | Model,
        checkpoint_path: Path | str = Path("models/fran/fran_synthetic_512.pth"),
    ) -> torch.nn.Module:
        if model_type not in ModelManager.cache:
            ModelManager.cache[model_type] = ModelManager.setup_model(
                model_type, checkpoint_path
            )
        return ModelManager.cache[model_type]


def images_to_gif(images, fps=45):
    path = Path("examples/temp.gif")
    images[0].save(
        path, format="GIF", save_all=True, append_images=images[1:], loop=0, fps=fps
    )
    return path


def prep_image_to_pil(image):
    new_image_out = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
    min_val, max_val = new_image_out.min(), new_image_out.max()
    new_image_normalized = 255 * (new_image_out - min_val) / (max_val - min_val)
    return Image.fromarray(new_image_normalized.astype("uint8"))


def generate_output(input_image, input_age, input_gender, model_type):
    if model_type == "FRAN":
        model = ModelManager.get_model(Model(model_type))
        model.eval()
        target_ages = [13, 18, 23, 33, 38, 48, 53, 58, 63, 68, 73, 78]
        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        input_image_tensor = transform(input_image)
        input_age_embedding = torch.full(
            (1, input_image_tensor.shape[1], input_image_tensor.shape[2]),
            input_age / 100,
        )
        out_image_array = []
        for i, target_age in enumerate(target_ages):
            target_age_embedding = torch.full(
                (1, input_image_tensor.shape[1], input_image_tensor.shape[2]),
                target_age / 100,
            )
            input_tensor = torch.cat(
                (input_image_tensor, input_age_embedding, target_age_embedding), dim=0
            )
            with torch.no_grad():
                output = model(input_tensor.unsqueeze(0).to(ModelManager.device))
            out_image_array.append(prep_image_to_pil(output))
        gif = images_to_gif(out_image_array, fps=24)
        return gif
    elif model_type == "CycleGAN":
        if input_gender.lower() == "male":
            cyclegan1 = ModelManager.cache.setdefault(
                "cyclegan1_male",
                ModelManager.setup_model(
                    Model(model_type),
                    Path("models/cycle_gan/cycle_gan_cacd_male1_244.pth"),
                ),
            )
            cyclegan2 = ModelManager.cache.setdefault(
                "cyclegan2_male",
                ModelManager.setup_model(
                    Model(model_type),
                    Path("models/cycle_gan/cycle_gan_cacd_male2_244.pth"),
                ),
            )
            cyclegan3 = ModelManager.cache.setdefault(
                "cyclegan3_male",
                ModelManager.setup_model(
                    Model(model_type),
                    Path("models/cycle_gan/cycle_gan_cacd_male3_244.pth"),
                ),
            )
        else:
            cyclegan1 = ModelManager.cache.setdefault(
                "cyclegan1_female",
                ModelManager.setup_model(
                    Model(model_type),
                    Path("models/cycle_gan/cycle_gan_cacd_female1_244.pth"),
                ),
            )
            cyclegan2 = ModelManager.cache.setdefault(
                "cyclegan2_female",
                ModelManager.setup_model(
                    Model(model_type),
                    Path("models/cycle_gan/cycle_gan_cacd_female2_244.pth"),
                ),
            )
            cyclegan3 = ModelManager.cache.setdefault(
                "cyclegan3_female",
                ModelManager.setup_model(
                    Model(model_type),
                    Path("models/cycle_gan/cycle_gan_cacd_female3_244.pth"),
                ),
            )
        cyclegan1.eval()
        cyclegan2.eval()
        cyclegan3.eval()
        transform = transforms.Compose(
            [
                transforms.Resize((244, 244)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        input_image_tensor = transform(input_image)
        out_image_intermediate = []
        out_image_array = []
        with torch.no_grad():
            if input_age < 35:
                out_image_intermediate.append(input_image_tensor)
                out_image_intermediate.append(
                    cyclegan2(input_image_tensor.unsqueeze(0).to(ModelManager.device))
                )
                out_image_intermediate.append(
                    cyclegan1(input_image_tensor.unsqueeze(0).to(ModelManager.device))
                )
            elif input_age < 50:
                out_image_intermediate.append(
                    cyclegan2(
                        input_image_tensor.unsqueeze(0).to(ModelManager.device),
                        reverse=True,
                    )
                )
                out_image_intermediate.append(input_image_tensor)
                out_image_intermediate.append(cyclegan1(out_image_intermediate[0]))
            else:
                out_image_intermediate.append(
                    cyclegan1(
                        input_image_tensor.unsqueeze(0).to(ModelManager.device),
                        reverse=True,
                    )
                )
                out_image_intermediate.append(cyclegan2(out_image_intermediate[0]))
                out_image_intermediate.append(input_image_tensor)
        for output in out_image_intermediate:
            out_image_array.append(prep_image_to_pil(output))
        gif = images_to_gif(out_image_array, fps=1)
        return gif
    elif model_type == "Diffusion":
        model = ModelManager.get_model(
            Model(model_type), "models/diffusion/diffusion_cyclegan_cacd_180.pth"
        )
        model.eval()
        transform = transforms.Compose(
            [
                transforms.Resize((180, 180)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        input_image_tensor = transform(input_image)
        with torch.no_grad():
            output = model(input_image_tensor.unsqueeze(0).to(ModelManager.device))
        output = prep_image_to_pil(output)
        return output
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented.")


def run_app():
    input_image = gr.Image(label="Input Image", type="pil")
    input_age = gr.Number(label="Input Age", minimum=18, maximum=83, value=18)
    input_gender = gr.Radio(
        ["Male", "Female"], label="Specify your gender", value="Male"
    )
    input_model = gr.Dropdown(
        [model.value for model in Model], label="Select model", value=Model.FRAN.value
    )
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
