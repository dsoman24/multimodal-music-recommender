# multimodal-music-recommender
CS 4644 Project, developing a deep learning multimodal music recommender system. [Read the paper here](https://github.com/dsoman24/multimodal-music-recommender/blob/main/CS_4644_Project.pdf).

## Setup

### Requirements

To download all requirements, run:

```bash
pip3 install -r requirements.txt
```

To update the requirements file use pipreqs:

```bash
pip3 install pipreqs
python3 -m  pipreqs.pipreqs
```


### Downloading the Million Song Dataset

To download the dataset used by this project, run:

```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

This download does not download the full MSD, only the necessary files (tags, lyrics, metadata, user history)

To download the MSD subset, run:
```bash
chmod +x download_subset.sh
./download_subset.sh
```
