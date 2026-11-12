import sys
import os
import traceback

# =====================================
# Allow imports from project root
# =====================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(BASE_DIR)

import flwr as fl
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from training.sannet_model import SANNetSegmentation
from training.dataset import LiverDataset


# =====================================
# Configuration
# =====================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 1
BATCH_SIZE = 4


# =====================================
# Hospital Configuration
# =====================================

HOSPITAL = os.environ.get(
    "HOSPITAL_ID",
    "hospital1"
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    HOSPITAL,
    "images"
)

MASK_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    HOSPITAL,
    "masks"
)


# =====================================
# Flower Client
# =====================================

class LiverClient(fl.client.NumPyClient):

    def __init__(self):

        print("=" * 60)
        print("Loading SANNet model...")
        print("=" * 60)

        self.model = SANNetSegmentation().to(DEVICE)

        print("Model Loaded Successfully")

        print("Loading Dataset...")

        print("=" * 60)
        print(f"Hospital      : {HOSPITAL}")
        print(f"Image Folder  : {IMAGE_DIR}")
        print(f"Mask Folder   : {MASK_DIR}")
        print("=" * 60)

        self.dataset = LiverDataset(
            IMAGE_DIR,
            MASK_DIR
        )

        print(f"Dataset Size : {len(self.dataset)}")

        self.loader = DataLoader(
            self.dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

        print("Client Ready")


            # =====================================
    # Send Parameters
    # =====================================

    def get_parameters(self, config):

        return [
            value.cpu().numpy()
            for value in self.model.state_dict().values()
        ]


    # =====================================
    # Receive Parameters
    # =====================================

    def set_parameters(self, parameters):

        state_dict = self.model.state_dict()

        for key, value in zip(
            state_dict.keys(),
            parameters
        ):

            state_dict[key] = torch.tensor(value)

        self.model.load_state_dict(
            state_dict,
            strict=True
        )


    # =====================================
    # Local Training
    # =====================================

    def fit(self, parameters, config):

        try:

            print("\n" + "=" * 60)
            print(f"Training Local Model ({HOSPITAL})")
            print("=" * 60)

            self.set_parameters(parameters)

            self.model.train()

            for epoch in range(EPOCHS):

                print(f"\nEpoch {epoch + 1}/{EPOCHS}")

                running_loss = 0.0

                for batch_idx, (images, masks) in enumerate(self.loader):

                    print(f"Loading Batch {batch_idx}")

                    images = images.to(DEVICE)
                    masks = masks.to(DEVICE)

                    self.optimizer.zero_grad()

                    outputs = self.model(images)

                    loss = self.loss_fn(
                        outputs,
                        masks
                    )

                    loss.backward()

                    self.optimizer.step()

                    running_loss += loss.item()

                    print(
                        f"Batch {batch_idx} Loss = {loss.item():.4f}"
                    )

                avg_loss = running_loss / len(self.loader)

                print(
                    f"\nEpoch {epoch + 1} Average Loss = {avg_loss:.4f}"
                )

            print("\nLocal training completed successfully.")

            return (

                [
                    value.cpu().numpy()
                    for value in self.model.state_dict().values()
                ],

                len(self.dataset),

                {
                    "hospital": HOSPITAL,
                    "loss": float(avg_loss)
                }

            )

        except Exception:

            print("\n" + "=" * 60)
            print("CLIENT ERROR")
            print("=" * 60)

            traceback.print_exc()

            input("\nPress Enter to exit...")

            raise


            # =====================================
    # Evaluation
    # =====================================

    def evaluate(self, parameters, config):

        try:

            self.set_parameters(parameters)

            print("\n" + "=" * 60)
            print(f"Evaluating Model ({HOSPITAL})")
            print("=" * 60)

            self.model.eval()

            total_loss = 0.0

            with torch.no_grad():

                for images, masks in self.loader:

                    images = images.to(DEVICE)
                    masks = masks.to(DEVICE)

                    outputs = self.model(images)

                    loss = self.loss_fn(
                        outputs,
                        masks
                    )

                    total_loss += loss.item()

            avg_loss = total_loss / len(self.loader)

            print(f"Evaluation Loss : {avg_loss:.4f}")

            return (

                float(avg_loss),

                len(self.dataset),

                {
                    "hospital": HOSPITAL,
                    "loss": float(avg_loss)
                }

            )

        except Exception:

            print("\n" + "=" * 60)
            print("EVALUATION ERROR")
            print("=" * 60)

            traceback.print_exc()

            input("\nPress Enter to exit...")

            raise


# =====================================
# Start Client
# =====================================

if __name__ == "__main__":

    print("=" * 60)
    print("Starting Hospital Client...")
    print("=" * 60)

    print(f"Hospital : {HOSPITAL}")
    print(f"Image Dir: {IMAGE_DIR}")
    print(f"Mask Dir : {MASK_DIR}")
    print("=" * 60)

    fl.client.start_numpy_client(

        server_address="localhost:8080",

        client=LiverClient()

    )