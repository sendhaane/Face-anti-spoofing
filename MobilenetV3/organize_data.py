import os
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = "test_images"
OUTPUT_DIR = "test_images_organized"

LIVE_FOLDER_NAME = "live"
SPOOF_FOLDER_NAME = "spoof"

# Number of images to copy from each class
MAX_IMAGES_PER_CLASS = 600


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

live_output = os.path.join(
    OUTPUT_DIR,
    LIVE_FOLDER_NAME
)

spoof_output = os.path.join(
    OUTPUT_DIR,
    SPOOF_FOLDER_NAME
)

os.makedirs(
    live_output,
    exist_ok=True
)

os.makedirs(
    spoof_output,
    exist_ok=True
)


# ============================================================
# VALID IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# COUNTERS
# ============================================================

live_images = 0
spoof_images = 0

missing_live_bb = 0
missing_spoof_bb = 0

duplicate_files = 0


# ============================================================
# FIND BB FILE
#
# Example:
#
# 496691.jpg
# 496691_BB.txt
#
# We use the BB only to verify that the image has
# a corresponding annotation.
#
# The BB itself is NOT copied.
# ============================================================

def find_bb_file(folder, image_filename):

    image_name = os.path.splitext(
        image_filename
    )[0]

    expected_bb_prefix = image_name + "_BB"

    for filename in os.listdir(folder):

        filename_without_ext = os.path.splitext(
            filename
        )[0]

        if filename_without_ext == expected_bb_prefix:

            return os.path.join(
                folder,
                filename
            )

    return None


# ============================================================
# COPY IMAGES
# ============================================================

def process_class(
    class_name,
    output_folder
):

    global live_images
    global spoof_images
    global missing_live_bb
    global missing_spoof_bb
    global duplicate_files

    print()
    print("=" * 60)
    print(f"PROCESSING: {class_name.upper()}")
    print("=" * 60)

    # --------------------------------------------------------
    # Find all class folders
    # --------------------------------------------------------

    class_folders = []

    for root, dirs, files in os.walk(
        SOURCE_DIR
    ):

        if os.path.basename(
            root
        ).lower() == class_name.lower():

            class_folders.append(root)

    # Deterministic ordering
    class_folders.sort()

    print(
        f"Found {len(class_folders)} "
        f"{class_name} folders"
    )

    # --------------------------------------------------------
    # Process each class folder
    # --------------------------------------------------------

    for class_folder in class_folders:

        # Stop once we reach required number
        if class_name.lower() == "live":

            if live_images >= MAX_IMAGES_PER_CLASS:
                break

        else:

            if spoof_images >= MAX_IMAGES_PER_CLASS:
                break

        print()
        print(
            f"Reading: {class_folder}"
        )

        # Sort for reproducibility
        filenames = sorted(
            os.listdir(class_folder)
        )

        for filename in filenames:

            # ------------------------------------------------
            # Stop after reaching limit
            # ------------------------------------------------

            if class_name.lower() == "live":

                if live_images >= MAX_IMAGES_PER_CLASS:
                    break

            else:

                if spoof_images >= MAX_IMAGES_PER_CLASS:
                    break

            # ------------------------------------------------
            # Only process images
            # ------------------------------------------------

            if not filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):
                continue

            image_path = os.path.join(
                class_folder,
                filename
            )

            # ------------------------------------------------
            # Verify corresponding BB exists
            #
            # IMPORTANT:
            # BB is NOT copied.
            # ------------------------------------------------

            bb_path = find_bb_file(
                class_folder,
                filename
            )

            if bb_path is None:

                print(
                    f"[WARNING] BB not found: "
                    f"{image_path}"
                )

                if class_name.lower() == "live":
                    missing_live_bb += 1
                else:
                    missing_spoof_bb += 1

                continue

            # ------------------------------------------------
            # Destination
            # ------------------------------------------------

            destination_image = os.path.join(
                output_folder,
                filename
            )

            # ------------------------------------------------
            # Check duplicate
            # ------------------------------------------------

            if os.path.exists(
                destination_image
            ):

                print(
                    f"[DUPLICATE] {filename}"
                )

                duplicate_files += 1

                continue

            # ------------------------------------------------
            # Copy ONLY image
            # ------------------------------------------------

            shutil.copy2(
                image_path,
                destination_image
            )

            # ------------------------------------------------
            # Update counters
            # ------------------------------------------------

            if class_name.lower() == "live":

                live_images += 1

                current_count = live_images

            else:

                spoof_images += 1

                current_count = spoof_images

            print(
                f"[{current_count}/"
                f"{MAX_IMAGES_PER_CLASS}] "
                f"[COPIED] "
                f"{filename}"
            )


# ============================================================
# PROCESS LIVE
# ============================================================

process_class(
    LIVE_FOLDER_NAME,
    live_output
)


# ============================================================
# PROCESS SPOOF
# ============================================================

process_class(
    SPOOF_FOLDER_NAME,
    spoof_output
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("DATASET ORGANIZATION COMPLETE")
print("=" * 60)

print()

print("LIVE")
print(
    f"  Images copied       : {live_images}"
)
print(
    f"  Missing BB          : {missing_live_bb}"
)

print()

print("SPOOF")
print(
    f"  Images copied       : {spoof_images}"
)
print(
    f"  Missing BB          : {missing_spoof_bb}"
)

print()

print(
    f"Duplicate files      : {duplicate_files}"
)

print()

print("TOTAL DATA")
print(
    f"  LIVE                : {live_images}"
)
print(
    f"  SPOOF               : {spoof_images}"
)
print(
    f"  TOTAL               : "
    f"{live_images + spoof_images}"
)

print()

print("Output:")
print(
    f"  {live_output}"
)
print(
    f"  {spoof_output}"
)

print()

print("Only images were copied.")

print("Original dataset was NOT modified.")

print("=" * 60)