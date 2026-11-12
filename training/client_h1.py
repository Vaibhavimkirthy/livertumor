print("CLIENT H1 STARTED")

from federated_client import start_client

print("IMPORT SUCCESSFUL")

start_client(
    r"C:\Users\NIHAR\Desktop\Final\liver_project\datasets\hospital1\images",
    r"C:\Users\NIHAR\Desktop\Final\liver_project\datasets\hospital1\masks"
)

print("THIS SHOULD NEVER PRINT")