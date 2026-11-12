import os
import random
import shutil

# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SOURCE_IMAGE_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "train",
    "images"
)

SOURCE_MASK_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "train",
    "masks"
)

FEDERATED_DIR = os.path.join(
    BASE_DIR,
    "datasets"
)

NUM_CLIENTS = 5

CLIENT_NAMES = [
    "hospital1",
    "hospital2",
    "hospital3",
    "hospital4",
    "hospital5"
]

# Fixed seed so the same split can be reproduced
RANDOM_SEED = 42


# ==========================================================
# CHECK SOURCE DIRECTORIES
# ==========================================================

if not os.path.exists(SOURCE_IMAGE_DIR):

    raise FileNotFoundError(
        f"Image directory not found:\n{SOURCE_IMAGE_DIR}"
    )

if not os.path.exists(SOURCE_MASK_DIR):

    raise FileNotFoundError(
        f"Mask directory not found:\n{SOURCE_MASK_DIR}"
    )


# ==========================================================
# GET IMAGE FILES
# ==========================================================

image_files = sorted(
    [
        f
        for f in os.listdir(SOURCE_IMAGE_DIR)
        if f.lower().endswith(".png")
    ]
)

print("=" * 70)
print("FEDERATED DATASET CREATION")
print("=" * 70)

print(
    f"Total training images found : {len(image_files)}"
)


# ==========================================================
# CHECK IMAGE / MASK MATCHING
# ==========================================================

valid_files = []

for filename in image_files:

    image_path = os.path.join(
        SOURCE_IMAGE_DIR,
        filename
    )

    mask_path = os.path.join(
        SOURCE_MASK_DIR,
        filename
    )

    if not os.path.exists(mask_path):

        print(
            f"WARNING: Mask missing for {filename}"
        )

        continue

    valid_files.append(filename)


print(
    f"Valid image/mask pairs       : {len(valid_files)}"
)


# ==========================================================
# GROUP IMAGES BY VOLUME
# ==========================================================

volumes = {}

for filename in valid_files:

    # Example:
    # volume-101_404.png
    #
    # Volume ID:
    # volume-101

    volume_id = filename.split("_")[0]

    if volume_id not in volumes:

        volumes[volume_id] = []

    volumes[volume_id].append(
        filename
    )


print(
    f"Unique volumes found         : {len(volumes)}"
)


# ==========================================================
# DISPLAY VOLUMES
# ==========================================================

print()
print("Volumes:")

for volume_id in sorted(volumes.keys()):

    print(
        f"  {volume_id}: "
        f"{len(volumes[volume_id])} slices"
    )


# ==========================================================
# SHUFFLE VOLUMES
# ==========================================================

volume_ids = list(
    volumes.keys()
)

random.seed(
    RANDOM_SEED
)

random.shuffle(
    volume_ids
)


# ==========================================================
# DISTRIBUTE VOLUMES ACROSS CLIENTS
# ==========================================================

client_volumes = {

    client: []

    for client in CLIENT_NAMES

}


for index, volume_id in enumerate(volume_ids):

    client = CLIENT_NAMES[
        index % NUM_CLIENTS
    ]

    client_volumes[
        client
    ].append(
        volume_id
    )


# ==========================================================
# CREATE CLIENT DIRECTORIES
# ==========================================================

for client in CLIENT_NAMES:

    client_image_dir = os.path.join(
        FEDERATED_DIR,
        client,
        "images"
    )

    client_mask_dir = os.path.join(
        FEDERATED_DIR,
        client,
        "masks"
    )

    os.makedirs(
        client_image_dir,
        exist_ok=True
    )

    os.makedirs(
        client_mask_dir,
        exist_ok=True
    )


# ==========================================================
# COPY DATA
# ==========================================================

for client in CLIENT_NAMES:

    print()
    print("-" * 70)

    print(
        f"Creating dataset for {client}"
    )

    print("-" * 70)

    total_slices = 0

    for volume_id in client_volumes[client]:

        files = volumes[
            volume_id
        ]

        print(
            f"{volume_id}: "
            f"{len(files)} slices"
        )

        for filename in files:

            source_image = os.path.join(
                SOURCE_IMAGE_DIR,
                filename
            )

            source_mask = os.path.join(
                SOURCE_MASK_DIR,
                filename
            )

            destination_image = os.path.join(
                FEDERATED_DIR,
                client,
                "images",
                filename
            )

            destination_mask = os.path.join(
                FEDERATED_DIR,
                client,
                "masks",
                filename
            )

            shutil.copy2(
                source_image,
                destination_image
            )

            shutil.copy2(
                source_mask,
                destination_mask
            )

            total_slices += 1

    print(
        f"Total slices copied: {total_slices}"
    )


# ==========================================================
# FINAL VERIFICATION
# ==========================================================

print()
print("=" * 70)
print("FEDERATED DATASET SUMMARY")
print("=" * 70)

total_all_clients = 0

for client in CLIENT_NAMES:

    image_dir = os.path.join(
        FEDERATED_DIR,
        client,
        "images"
    )

    mask_dir = os.path.join(
        FEDERATED_DIR,
        client,
        "masks"
    )

    images = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith(".png")
    ]

    masks = [
        f
        for f in os.listdir(mask_dir)
        if f.lower().endswith(".png")
    ]

    image_set = set(images)
    mask_set = set(masks)

    matching = image_set & mask_set

    print()
    print(f"{client}")
    print(
        f"  Volumes : "
        f"{len(client_volumes[client])}"
    )
    print(
        f"  Images  : "
        f"{len(images)}"
    )
    print(
        f"  Masks   : "
        f"{len(masks)}"
    )
    print(
        f"  Matching: "
        f"{len(matching)}"
    )

    if image_set != mask_set:

        print(
            "  WARNING: Image/mask mismatch!"
        )

    else:

        print(
            "  Status  : OK"
        )

    total_all_clients += len(images)


# ==========================================================
# FINAL CHECK
# ==========================================================

print()
print("=" * 70)

print(
    f"Original training pairs : {len(valid_files)}"
)

print(
    f"Federated client pairs   : {total_all_clients}"
)

if total_all_clients == len(valid_files):

    print(
        "CHECK PASSED: All training data was distributed."
    )

else:

    print(
        "WARNING: Distributed count does not match source count."
    )

print("=" * 70)

print()
print(
    "Federated dataset creation completed."
)