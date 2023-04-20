import torch
import numpy as np
import shutil
from PIL import Image
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import DBSCAN
from transformers import CLIPProcessor, CLIPModel
import argparse
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Image clustering using CLIP embeddings and DBSCAN."
    )
    parser.add_argument(
        "--image_directory",
        type=str,
        required=True,
        help="Path to the directory containing the images to cluster.",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14-336",
        help="CLIP model to use for generating embeddings. Options listed from least to most demanding: openai/clip-vit-base-patch16, openai/clip-vit-base-patch32, openai/clip-vit-large-patch14, openai/clip-vit-large-patch14-336 (default)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=4.5,
        help="Eps value for DBSCAN clustering. Lower values will reduce false positives but may miss more. (default: 4.5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=192,
        help="Batch size for processing images. Higher values will require more VRAM. (default: 192)",
    )
    args = parser.parse_args()

    image_directory = Path(args.image_directory)
    clip_model = args.clip_model
    eps = args.eps
    batch_size = args.batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model)
    allowed_extensions = {".jpeg", ".jpg", ".png", ".webp"}
    images_to_paths = {
        image_path.stem: image_path
        for image_path in image_directory.iterdir()
        if image_path.suffix.lower() in allowed_extensions
    }
    all_image_ids = list(images_to_paths.keys())
    all_embeddings = []

    total_images = len(all_image_ids)
    progress_bar = tqdm(total=total_images, desc="Processing images")

    for i in range(0, total_images, batch_size):
        batch_image_ids = all_image_ids[i : i + batch_size]
        batch_images = [
            Image.open(images_to_paths[image_id]) for image_id in batch_image_ids
        ]
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(
            device
        )

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        all_embeddings.extend(outputs.cpu().numpy())

        progress_bar.update(len(batch_image_ids))

    progress_bar.close()
    print("Clustering images...")

    images_to_embeddings = {
        image_id: tensor_embedding
        for image_id, tensor_embedding in zip(all_image_ids, all_embeddings)
    }

    clustering = DBSCAN(min_samples=2, eps=eps).fit(
        np.stack(list(images_to_embeddings.values()))
    )

    image_id_communities = defaultdict(set)

    for image_id, cluster_idx in zip(images_to_embeddings.keys(), clustering.labels_):
        image_id_communities[cluster_idx].add(image_id)

    output_directory = image_directory

    for idx, image_id_community in enumerate(image_id_communities.values()):
        if len(image_id_community) < 2:
            continue

        similarities = [
            image_similarity(
                images_to_embeddings[image_id1], images_to_embeddings[image_id2]
            )
            for image_id1 in image_id_community
            for image_id2 in image_id_community
            if image_id1 != image_id2
        ]

        if min(similarities) >= 0.5:
            cluster_directory = output_directory / f"cluster_{idx}"
            cluster_directory.mkdir(parents=True, exist_ok=True)

            for image_id in image_id_community:
                source = images_to_paths[image_id]
                destination = cluster_directory / source.name
                shutil.move(source, destination)

    independent_directory = output_directory / "unique"
    independent_directory.mkdir(parents=True, exist_ok=True)
    for image_id in image_id_communities[-1]:
        source = images_to_paths[image_id]
        destination = independent_directory / source.name
        shutil.move(source, destination)


def image_similarity(embedding1, embedding2):
    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    return similarity


if __name__ == "__main__":
    main()