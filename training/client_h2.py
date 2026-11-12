print("CLIENT H2 STARTED")

from federated_client import start_client

print("IMPORT SUCCESSFUL")

start_client(
    r"C:\Users\NIHAR\Desktop\Final\liver_project\datasets\hospital2\images",
    r"C:\Users\NIHAR\Desktop\Final\liver_project\datasets\hospital2\masks"
)

print("THIS SHOULD NEVER PRINT")