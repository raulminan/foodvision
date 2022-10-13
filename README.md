# foodvision

A PyTorch CV model based on a pretrained EfficientNetB2 model to classify 101 types of food.

Project directory:

    ├── demos
    │   ├── foodvision-big
    │   ├── foodvision-mini
    |
    ├── foodvision
    │   ├── 09_PyTorch_Model_Deployment.ipynb
    │   ├── data_setup.py
    │   ├── engine.py
    │   ├── get_data.py
    │   ├── model_builder.py
    │   ├── models
    │   │   └── foodvision_tinyvgg_model.pth
    │   ├── train.py
    │   └── utils.py
    ├── LICENSE
    ├── README.md
    └── requirements.txt

In `foodvision/foodvision`, there are functions to set up data and train a PyTorch model. Additionally, in `model_builder.py`, I replicate the architechture of the TinyVGG model, another famous CV model, although this architechture isn't used later on. 

These functions are used in `foodvision/foodvision/09_PyTorch_Model_Deployment.ipynb` to train an EfficientNetB2 model using transfer learning. In this notebook, I also train a ViT model, but the foodvision demo ultimately uses EfficientNet because it's faster, and only slightly less accurate. 

Finally, `foodvision/demos` has the code to build the apps demos, which are available at huggingface. [`foodvision-mini`](https://huggingface.co/spaces/raulminan/foodvision-mini) can only classify three different foods (sushi, pizza and steak), while [`foodvision-big`](https://huggingface.co/spaces/raulminan/foodvision-big) can classify 101.
