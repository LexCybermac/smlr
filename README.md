# Smlr - A Simple Image Clustering Script using CLIP and Hierarchical Clustering


Smlr is a simple script for machine learning hobbyists looking for a quick way to minimize overrepresentation of concepts in image datasets. This script groups similar images in a directory using hierarchical clustering with CLIP embeddings and the Annoy library for efficient similarity search.

## Requirements

To use this script, ensure you have the following packages installed in your Python environment:

- torch
- torchvision
- numpy
- Pillow
- transformers
- tqdm
- pathlib
- annoy
- scipy

If you haven't already, you can install the required packages using pip in one go:

```bash
pip install torch torchvision numpy Pillow transformers tqdm pathlib annoy scipy
```

## How it Works

1. Checks if there's an existing embeddings file and loads it if found. If not, it generates embeddings for each image in the specified directory using a CLIP model from the HuggingFace Transformers library.
2. Saves the generated embeddings to a file in the image directory for future use*.
3. Builds an Annoy index for efficient similarity search using the generated embeddings.
4. Computes a distance matrix from the Annoy index.
5. Applies hierarchical clustering to the distance matrix and assigns clusters based on a given threshold.
6. Moves images corresponding to the embeddings in each cluster to separate folders, with names like "cluster_0", "cluster_1", and so on.
7. Moves images not included in any cluster to a folder named "unique".
8. If there are any corrupted or damaged images, they get moved to a folder named "corrupted".

*The script saves the generated embeddings in the image directory as 'embeddings.npy'. If this file is detected during future runs, the script will ask you if you want to use the existing embeddings or make new ones. This is useful if something goes wrong during the clustering process or if you want to try different thresholds without going through the whole embeddings generation process again.

## Usage

If you're running on a 3090 or a similarly beefy GPU you can quickly get started with the following command:

```bash
python smlr.py --image_directory /path/to/your/image_directory
```

### Some Extra Options for those who like to tinker or have less VRAM to spare

- `--clip_model`: Choose the pre-trained CLIP model for generating embeddings. Options from least to most demanding: `openai/clip-vit-base-patch16`, `openai/clip-vit-base-patch32`, `openai/clip-vit-large-patch14`, `openai/clip-vit-large-patch14-336` (default).
- `--threshold`: Set the threshold value for hierarchical clustering. Lower values will reduce false positives but may miss more. Default is `0.22`.
- `--batch_size`: Pick the batch size when generating CLIP embeddings. Higher values need more VRAM. Default is `192`.

After running the script, you'll find the clustered images in separate folders within the input directory, the unique images in the "unique" folder, and any corrupted images in the "corrupted" folder.

## A Few Notes

- The default CLIP model and batch size work well on a 24 GB RTX 3090. If you have less VRAM, you might run into OOM errors with these settings. If your graphics card is not as chunky, try starting with a batch size of 32 and the `clip-vit-base-patch32` model, and then work up (or down) from there until you find what works best for you.
- Different CLIP models might need different threshold values to work well. Feel free to play around with the options to see what works best for your specific case.
- While I've tested this script a fair few times and not had suffered any data loss it's always a good practise to have your data backed up before doing anything with it.

## Thanks
Cheers to [Theodoros Ntakouris](https://github.com/ntakouris) for outlining a clear starting point for this script in his [Medium article](https://zarkopafilis.medium.com/image-deduplication-using-openais-clip-and-community-detection-2504f0437e7e).
