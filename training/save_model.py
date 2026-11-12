import torch


def save_model(model, path="weights/best_model.pth"):

    torch.save(
        model.state_dict(),
        path
    )

    print(f"Model saved at {path}")


def load_model(model, path="weights/best_model.pth"):

    model.load_state_dict(
        torch.load(path, map_location="cpu")
    )

    model.eval()

    print("Model loaded successfully")

    return model