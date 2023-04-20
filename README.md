# Smlr - A Simple Image Clustering Script using CLIP and DBSCAN

Smlr is a simple script for machine learning hobbyists looking for a quick way to minimize overrepresentation of concepts in image datasets. This script clusters conceptually similar images in a directory using DBScan with CLIP embeddings.

## Requirements

To use this script, ensure you have the following packages installed in your Python environment:

- torch
- torchvision
- numpy
- Pillow
- scikit-learn
- transformers
- tqdm
- pathlib

If you haven't already, you can install the required packages using pip in one go:

```bash
pip install torch torchvision numpy Pillow scikit-learn transformers tqdm pathlib
```

## How it works

The script processes the input images in the following steps:

1. Embeddings are generated for each image in the specified directory using a CLIP model from the HuggingFace Transformers library.
2. The DBSCAN clustering algorithm processes these embeddings and clusters them based on similarity.
3. Images corresponding to the embeddings in each cluster are moved to separate folders, with folder names like "cluster_0", "cluster_1", etc.
4. Images not included in any cluster are moved to a folder named "unique".

## Usage

To use this script, simply run it like so:

```bash
python smlr.py --image_directory /path/to/your/image_directory
```

### Optional arguments

- `--clip_model`: The pre-trained CLIP model to use for generating embeddings. Options listed from least to most demanding are `openai/clip-vit-base-patch16`, `openai/clip-vit-base-patch32`, `openai/clip-vit-large-patch14`, `openai/clip-vit-large-patch14-336` (default).
- `--eps`: The `eps` value for DBSCAN clustering. Lower values will reduce false positives but may miss more. Default is `4.5`.
- `--batch_size`: The batch size to use when generating CLIP embeddings. Higher values will require more VRAM. Default is `192`.

After running the script, you'll find the clustered images in separate folders within the input directory, and the unique images in the "unique" folder.

## Notes

- The default CLIP model and batch size are selected based on what runs well on a 24 GB RTX 3090. Users with less VRAM may encounter OOM errors with these settings. If you have a less powerful graphics card, try starting with a batch size of 32 and the `clip-vit-base-patch32` model, and then work up (or down) from there until you find what works best for you.
- Different CLIP models may require different eps values to work well.
- While I've tested this script a fair few times and not had suffered any data loss it's always a good practise to have your data backed up in at least one place.

## Thanks
Cheers to [Theodoros Ntakouris](https://github.com/ntakouris) for outlining a clear starting point for this script in his [Medium article](https://zarkopafilis.medium.com/image-deduplication-using-openais-clip-and-community-detection-2504f0437e7e).
