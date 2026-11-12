import flwr as fl
import torch
from torch.utils.data import DataLoader
import os
import sys

# Allow imports from training folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from dataset import LiverDataset
from sannet_model import SANNetSegmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "weights",
    "federated_sannet_model.pth"
)

def start_client(image_path, mask_path):

    # ==========================
    # Dataset
    # ==========================

    dataset = LiverDataset(
        image_path,
        mask_path,
        image_size=256
    )

    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    print("Dataset Loaded")
    print("Images:", len(dataset.images))
    print("Masks :", len(dataset.masks))
    print("Samples:", len(dataset))

        # ==========================
    # Model
    # ==========================

    model = SANNetSegmentation()

    if os.path.exists(MODEL_PATH):

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        )

        if isinstance(checkpoint, dict):

            model.load_state_dict(checkpoint)

        else:

            state_dict = model.state_dict()

            for key, value in zip(state_dict.keys(), checkpoint):

                state_dict[key] = torch.tensor(value)

            model.load_state_dict(state_dict)

        print("Loaded SANNet weights")

    else:

        print("SANNet weights not found")

    model.to(DEVICE)

    # ==========================
    # Flower Client
    # ==========================
    

    class FlowerClient(fl.client.NumPyClient):

        def get_parameters(self, config):
            return [
                value.cpu().numpy()
                for value in model.state_dict().values()
            ]

        def set_parameters(self, parameters):

            params_dict = zip(
                model.state_dict().keys(),
                parameters
            )

            state_dict = {
                key: torch.tensor(value)
                for key, value in params_dict
            }

            model.load_state_dict(
                state_dict,
                strict=True
            )

        def fit(self, parameters, config):

            print("\nLocal training started")

            self.set_parameters(parameters)

            model.train()

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=1e-4
            )

            criterion = torch.nn.BCEWithLogitsLoss()

            EPOCHS = 1

            for epoch in range(EPOCHS):

                total_loss = 0

                for images, masks in train_loader:

                    images = images.to(DEVICE)
                    masks = masks.to(DEVICE)

                    optimizer.zero_grad()

                    outputs = model(images)

                    loss = criterion(outputs, masks)

                    loss.backward()

                    optimizer.step()

                    total_loss += loss.item()

                print(
                    f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}"
                )

            print("Local training completed")

            return (
                self.get_parameters(config),
                len(dataset),
                {}
            )

        def evaluate(self, parameters, config):

            self.set_parameters(parameters)

            return (
                0.0,
                len(dataset),
                {}
            )

    # ==========================
    # Start Flower Client
    # ==========================

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client()
    )