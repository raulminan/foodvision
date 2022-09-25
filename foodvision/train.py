import os
import torch
import data_setup
import engine
import model_builder
import utils

from torchvision import transforms

def main():
    # set hyperparameters
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10 
    LEARNING_RATE = 0.001

    # set directories
    train_dir = "data/pizza_steak_sushi/train"
    test_dir = "data/pizza_steak_sushi/test"

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create transforms
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # create DataLoaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloader(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # create model
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # start training 
    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device,
                writer=None)

    # save the model
    utils.save_model(model=model,
                    target_dir="models",
                    model_name="foodvision_tinyvgg_model.pth")

if __name__ == '__main__':
    main()