import os
import torch

import gradio as gr

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict
from PIL import Image

class_names = ["pizza", "steak", "sushi"]

# create effnetb2 model
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=len(class_names),
)

# load saved weights
effnetb2.load_state_dict(
    torch.load(
        f="modelname.pth",
        map_location=torch.device("cpu"),
    )
)

# predict function
def predict(img: Image) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on an image and returns the prediction
    and the time taken

    Parameters
    ----------
    img : Image
        Image to classify

    Returns
    -------
    Tuple[Dict, float]
        tuple with a dictionary that contains the probability that img belongs to
        each class and the time taken to make the prediction
        
        Example: ({"class1": 0.95, "class2": 0.02, "class3": 0.03}, 0.026)
    """
    start = timer()
    
    # transform target image and add batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)
    
    # put model into eval mode
    effnetb2.eval()
    with torch.inference_mode():
        preds_probs = torch.softmax(effnetb2(img), dim=1)
    
    # create a prediction label and pred prob dictionary
    pred_labels_and_probs = {
        class_names[i]: float(preds_probs[0][i]) for i in range(len(class_names))
    }
    
    # get prediction time
    pred_time = round(timer() - start, 5)
    
    return pred_labels_and_probs, pred_time

### Gradio app ###
title = "FoodVision Mini"
description = "An EfficientNetB2 feature extractor computer vision model to classify\
    images of pizza, steak and sushi"
article = "test"

# create exmaples list from "examples" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

def main():
    # create Gradio demo
    demo = gr.Interface(fn=predict,
                        inputs=gr.Image(type="pil"),
                        outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                                gr.Number(label="Prediction time (s)")],
                        examples=example_list,
                        title=title,
                        description=description,
                        article=article)

    # launch demo
    demo.launch()

if __name__ == '__main__':
    main()