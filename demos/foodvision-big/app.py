from venv import create
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict
from PIL import Image

with open("class_names.txt", "r") as f:
    class_names = [food_name.strip() for food_name in f.readlines()]

effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=len(class_names)
)

effnetb2.load_state_dict(
    torch.load(
        f="09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth",
        map_location=torch.device("cpu")
    )
)

def predict(img: Image) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on an image and returns the 
    prediction and the time taken

    Parameters
    ----------
    img : Image
        an Image

    Returns
    -------
    Tuple[Dict, float]
        Tuple[prediction probabilities, time taken]
    """

    start = timer()
    
    # trasnform and add batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)

    # put model on eval mode and turn on inference
    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    # create a prediction label: prediction prob dict
    pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }

    # calculate pred time
    pred_time = round(timer() - start, 5)

    return pred_labels_and_probs, pred_time

### Gradio App

title = "FoodVision Big"
description = "An EfficientNetB2 frature extractor CV model to classify images of food"
article = "TODO"

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

def main():
# create Gradio interface
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Label(num_top_classes=5, label="Predictions"),
            gr.Number(label="Prediction time (s)")
        ],
        examples=example_list,
        title=title,
        description=description,
        article=article
    )
    demo.launch()

if __name__ == "__main__":
    main()
