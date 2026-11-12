import os
import shutil
import random

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_IMAGES = os.path.join(
    BASE_DIR,
    "datasets",
    "train",
    "images"
)

SOURCE_MASKS = os.path.join(
    BASE_DIR,
    "datasets",
    "train",
    "masks"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "datasets_demo"
)

HOSPITALS = [
    "hospital1",
    "hospital2",
    "hospital3",
    "hospital4",
    "hospital5"
]

# Small demo dataset
IMAGES_PER_HOSPITAL = 25

# Fixed seed for reproducibility
random.seed(42)

# ============================================================
# START
# ============================================================

print("=" * 60)
print("CREATING DEMO FEDERATED DATASET")
print("=" * 60)

# ============================================================
# CHECK SOURCE DATASET
# ============================================================

if not os.path.exists(SOURCE_IMAGES):
    raise FileNotFoundError(
        f"Image folder not found:\n{SOURCE_IMAGES}"
    )

if not os.path.exists(SOURCE_MASKS):
    raise FileNotFoundError(
        f"Mask folder not found:\n{SOURCE_MASKS}"
    )

# ============================================================
# GET VALID IMAGE/MASK PAIRS
# ============================================================

image_files = sorted(
    [
        f for f in os.listdir(SOURCE_IMAGES)
        if f.lower().endswith(".png")
    ]
)

mask_files = set(
    [
        f for f in os.listdir(SOURCE_MASKS)
        if f.lower().endswith(".png")
    ]
)

valid_pairs = [
    f for f in image_files
    if f in mask_files
]

print()
print("Source images :", len(image_files))
print("Valid pairs   :", len(valid_pairs))

required = IMAGES_PER_HOSPITAL * len(HOSPITALS)

if len(valid_pairs) < required:
    raise RuntimeError(
        f"Not enough valid pairs. "
        f"Required: {required}, Found: {len(valid_pairs)}"
    )

# ============================================================
# REMOVE OLD DEMO DATASET
# ============================================================

if os.path.exists(OUTPUT_DIR):
    print()
    print("Removing previous demo dataset...")
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SHUFFLE DATA
# ============================================================

random.shuffle(valid_pairs)

# ============================================================
# CREATE HOSPITAL DATASETS
# ============================================================

index = 0

for hospital in HOSPITALS:

    print()
    print("=" * 60)
    print(f"Creating {hospital}")
    print("=" * 60)

    hospital_images = os.path.join(
        OUTPUT_DIR,
        hospital,
        "images"
    )

    hospital_masks = os.path.join(
        OUTPUT_DIR,
        hospital,
        "masks"
    )

    os.makedirs(
        hospital_images,
        exist_ok=True
    )

    os.makedirs(
        hospital_masks,
        exist_ok=True
    )

    selected = valid_pairs[
        index:index + IMAGES_PER_HOSPITAL
    ]

    index += IMAGES_PER_HOSPITAL

    for filename in selected:

        source_image = os.path.join(
            SOURCE_IMAGES,
            filename
        )

        source_mask = os.path.join(
            SOURCE_MASKS,
            filename
        )

        destination_image = os.path.join(
            hospital_images,
            filename
        )

        destination_mask = os.path.join(
            hospital_masks,
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

    print(f"Images copied : {len(selected)}")
    print(f"Masks copied  : {len(selected)}")
    print("Status        : OK")

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("DEMO FEDERATED DATASET CREATED")
print("=" * 60)

total = 0

for hospital in HOSPITALS:

    image_dir = os.path.join(
        OUTPUT_DIR,
        hospital,
        "images"
    )

    mask_dir = os.path.join(
        OUTPUT_DIR,
        hospital,
        "masks"
    )

    images = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(".png")
    ]

    masks = [
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(".png")
    ]

    matching = len(
        set(images) & set(masks)
    )

    print()
    print(hospital)
    print("Images  :", len(images))
    print("Masks   :", len(masks))
    print("Matching:", matching)

    total += len(images)

print()
print("Total demo images:", total)
print()
print("Location:")
print(OUTPUT_DIR)
print("=" * 60)