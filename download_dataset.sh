#!/bin/bash

MBTAG_URL="http://www.ee.columbia.edu/~thierry/artist_term.db"
LYRICS_URL="http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_dataset.db"
USER_DATA_URL="http://labrosa.ee.columbia.edu/~dpwe/tmp/train_triplets.txt.zip"
METADATA_URL="http://millionsongdataset.com/sites/default/files/AdditionalFiles/track_metadata.db"
MBTAG_OUTPUT="${2:-$(basename "$MBTAG_URL")}"
LYRICS_OUTPUT="${2:-$(basename "$LYRICS_URL")}"
USER_DATA_OUTPUT="${2:-$(basename "$USER_DATA_URL")}"
METADATA_OUTPUT="${2:-$(basename "$METADATA_URL")}"

# === Configuration ===
DATA_DIR="data"

# === Create 'data' Directory ===
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating directory '$DATA_DIR'..."
    mkdir -p "$DATA_DIR"
    if [ $? -ne 0 ]; then
        echo "Failed to create directory '$DATA_DIR'."
        exit 1
    fi
else
    echo "Directory '$DATA_DIR' already exists."
fi

# --- Using wget ---
download_msd_parts() {

    echo "Starting download from: $MBTAG_URL"
    wget --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)" -O "$DATA_DIR/$MBTAG_OUTPUT" "$MBTAG_URL"

    if [ $? -eq 0 ]; then
        echo "Download completed successfully. Saved as '$DATA_DIR/$MBTAG_OUTPUT'."
    else
        echo "Download failed."
        exit 1
    fi

    echo "Starting download from: $LYRICS_URL"
    wget --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)" -O "$DATA_DIR/$LYRICS_OUTPUT" "$LYRICS_URL"

    if [ $? -eq 0 ]; then
        echo "Download completed successfully. Saved as '$DATA_DIR/$LYRICS_OUTPUT'."
    else
        echo "Download failed."
        exit 1
    fi

    echo "Starting download from: $USER_DATA_URL"
    wget --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)" -O "$DATA_DIR/$USER_DATA_OUTPUT" "$USER_DATA_URL"

    if [ $? -eq 0 ]; then
        echo "Download completed successfully. Saved as '$DATA_DIR/$USER_DATA_OUTPUT'."
    else
        echo "Download failed."
        exit 1
    fi

    echo "Starting download from: $METADATA_URL"
    wget --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)" -O "$DATA_DIR/$METADATA_OUTPUT" "$METADATA_URL"

    if [ $? -eq 0 ]; then
        echo "Download completed successfully. Saved as '$DATA_DIR/$METADATA_OUTPUT'."
    else
        echo "Download failed."
        exit 1
    fi
}


extract_and_delete_user_zip() {
    ZIP_FILE_PATH="$DATA_DIR/$USER_DATA_OUTPUT"

    # Check if the file exists
    if [ ! -f "$ZIP_FILE_PATH" ]; then
        echo "Error: File '$ZIP_FILE_PATH' does not exist."
        exit 1
    fi

    # Check if the file is a .zip archive
    if [[ "$USER_DATA_OUTPUT" != *.zip ]]; then
        echo "Error: File '$USER_DATA_OUTPUT' is not a .zip archive."
        exit 1
    fi

    echo "Extracting '$USER_DATA_OUTPUT' into '$DATA_DIR'..."
    unzip -o "$ZIP_FILE_PATH" -d "$DATA_DIR"

    if [ $? -eq 0 ]; then
        echo "Extraction completed successfully."
    else
        echo "Error: Extraction failed."
        exit 1
    fi

    echo "Deleting archive '$USER_DATA_OUTPUT'..."
    rm "$ZIP_FILE_PATH"

    if [ $? -eq 0 ]; then
        echo "Archive '$USER_DATA_OUTPUT' deleted successfully."
    else
        echo "Warning: Failed to delete archive '$USER_DATA_OUTPUT'. Please delete it manually."
    fi
}


# === Execute Download ===
download_msd_parts
extract_and_delete_user_zip

echo "All operations completed successfully."