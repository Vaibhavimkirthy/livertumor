from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from training.predict import predict_image


# ==========================================================
# BASE DIRECTORY
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)


# ==========================================================
# FOLDERS
# ==========================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

RESULT_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "results"
)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)


# ==========================================================
# DATASET MASK FOLDERS
# ==========================================================

DATASET_MASK_FOLDERS = [

    os.path.join(
        BASE_DIR,
        "datasets",
        "train",
        "masks"
    ),

    os.path.join(
        BASE_DIR,
        "datasets",
        "val",
        "masks"
    ),

    os.path.join(
        BASE_DIR,
        "datasets",
        "test",
        "masks"
    )
]


# ==========================================================
# VALIDATE UPLOADED FILE
# ==========================================================

def is_valid_ct_scan(filename):

    allowed_extensions = (
        ".png",
        ".jpg",
        ".jpeg"
    )

    return filename.lower().endswith(
        allowed_extensions
    )


# ==========================================================
# FIND GROUND-TRUTH MASK
# ==========================================================

def find_ground_truth_mask(image_filename):

    """
    Our prepared dataset uses the SAME filename
    for image and mask.

    Example:

    Image:
    volume-101_404.png

    Mask:
    volume-101_404.png
    """

    filename = os.path.basename(
        image_filename
    )


    # ------------------------------------------------------
    # Search train / validation / test
    # ------------------------------------------------------

    for mask_dir in DATASET_MASK_FOLDERS:

        mask_path = os.path.join(
            mask_dir,
            filename
        )

        if os.path.exists(mask_path):

            return mask_path


    # ------------------------------------------------------
    # No matching ground truth
    # ------------------------------------------------------

    return None


# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ==========================================================
# PREDICTION PAGE
# ==========================================================

@app.route("/prediction")
def prediction():

    return render_template(
        "prediction.html"
    )


# ==========================================================
# DASHBOARD
# ==========================================================

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# ==========================================================
# GRAPH
# ==========================================================

@app.route("/graph")
def graph():

    return render_template(
        "graph.html"
    )


# ==========================================================
# ABOUT
# ==========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ==========================================================
# PREDICT
# ==========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # ======================================================
    # CHECK FILE
    # ======================================================

    if "image" not in request.files:

        return "No image uploaded."


    file = request.files["image"]


    if file.filename == "":

        return "Please select an image."


    # ======================================================
    # VALIDATE EXTENSION
    # ======================================================

    if not is_valid_ct_scan(
        file.filename
    ):

        return render_template(

            "result.html",

            image="",

            mask="",

            prediction="Invalid Input",

            confidence="-",

            dice="-",

            iou="-",

            precision="-",

            recall="-",

            message=(
                "Please upload a Liver CT image only."
            )
        )


    # ======================================================
    # SECURE FILENAME
    # ======================================================

    filename = secure_filename(
        file.filename
    )


    # ======================================================
    # SAVE UPLOADED IMAGE
    # ======================================================

    upload_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    file.save(
        upload_path
    )


    # ======================================================
    # RESULT PATH
    # ======================================================

    result_path = os.path.join(
        app.config["RESULT_FOLDER"],
        "predicted_mask.png"
    )


    # ======================================================
    # FIND GROUND-TRUTH MASK
    # ======================================================

    gt_mask_path = find_ground_truth_mask(
        filename
    )


    print()
    print("=" * 60)
    print("IMAGE:", filename)
    print(
        "Ground Truth Mask:",
        gt_mask_path
    )
    print("=" * 60)


    # ======================================================
    # PREDICTION
    # ======================================================

    (
        _,
        diagnosis,
        confidence,
        severity,
        dice,
        iou,
        precision,
        recall
    ) = predict_image(

        upload_path,

        result_path,

        gt_mask_path
    )


    # ======================================================
    # RESULT PAGE
    # ======================================================

    return render_template(

        "result.html",

        image=filename,

        mask="predicted_mask.png",

        prediction=diagnosis,

        severity=severity,

        confidence=confidence,

        dice=dice,

        iou=iou,

        precision=precision,

        recall=recall,

        message=""
    )


# ==========================================================
# RUN FLASK
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )