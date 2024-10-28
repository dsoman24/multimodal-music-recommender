#!/bin/bash

SUBSET_URL="http://labrosa.ee.columbia.edu/~dpwe/tmp/millionsongsubset.tar.gz"
MBTAG_URL="http://www.ee.columbia.edu/~thierry/artist_term.db"
SUBSET_OUTPUT="${2:-$(basename "$SUBSET_URL")}"
MBTAG_OUTPUT="${2:-$(basename "$MBTAG_URL")}"

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
download_msd_subset() {
    # echo "Starting download from: $SUBSET_URL"
    # wget -O "$DATA_DIR/$SUBSET_OUTPUT" "$SUBSET_URL"

    # if [ $? -eq 0 ]; then
    #     echo "Download completed successfully. Saved as '$DATA_DIR/$SUBSET_OUTPUT'."
    # else
    #     echo "Download failed."
    #     exit 1
    # fi

    echo "Starting download from: $MBTAG_URL"
    wget --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)" -O "$DATA_DIR/$MBTAG_OUTPUT" "$MBTAG_URL"

    if [ $? -eq 0 ]; then
        echo "Download completed successfully. Saved as '$DATA_DIR/$MBTAG_OUTPUT'."
    else
        echo "Download failed."
        exit 1
    fi
}

extract_tar_gz() {
    FILE_PATH="$DATA_DIR/$SUBSET_OUTPUT"

    # Check if the file exists
    if [ ! -f "$FILE_PATH" ]; then
        echo "Error: File '$FILE_PATH' does not exist."
        exit 1
    fi

    # Check if the file is a .tar.gz archive
    if [[ "$SUBSET_OUTPUT" != *.tar.gz ]]; then
        echo "Error: File '$OUTPUT' is not a .tar.gz archive."
        exit 1
    fi

    echo "Extracting '$SUBSET_OUTPUT' into '$DATA_DIR'..."
    tar -xzf "$FILE_PATH" -C "$DATA_DIR"

    if [ $? -eq 0 ]; then
        echo "Extraction completed successfully."
    else
        echo "Error: Extraction failed."
        exit 1
    fi
}

delete_msd_archive() {
    FILE_PATH="$DATA_DIR/$SUBSET_OUTPUT"

    echo "Deleting archive '$SUBSET_OUTPUT'..."
    rm "$FILE_PATH"

    if [ $? -eq 0 ]; then
        echo "Archive '$SUBSET_OUTPUT' deleted successfully."
    else
        echo "Warning: Failed to delete archive '$SUBSET_OUTPUT'. Please delete it manually."
    fi
}

# === Execute Download ===
download_msd_subset
extract_tar_gz
delete_msd_archive

echo "All operations completed successfully."