import torch
import numpy as np
import shutil
from PIL import Image
from pathlib import Path
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
import argparse
from tqdm.auto import tqdm
import faiss
from scipy.spatial.distance import squareform

def process_images(image_directory, clip_model, threshold, batch_size, search_chunk_size):
    image_directory = Path(image_directory)
    device = "cuda"

    embeddings_file = image_directory / 'embeddings.npy'
    regenerate_embeddings = check_and_load_embeddings(embeddings_file)

    model = CLIPModel.from_pretrained(clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model)
    allowed_extensions = {".jpeg", ".jpg", ".png", ".webp"}

    images_to_paths, all_image_ids = get_images_to_paths(image_directory, allowed_extensions)
    damaged_image_ids, all_embeddings = generate_embeddings(all_image_ids, images_to_paths, model, processor, device, batch_size, regenerate_embeddings, embeddings_file)

    if regenerate_embeddings:
        np.save(embeddings_file, all_embeddings)

    print("Creating FAISS index...")
    faiss_index, opq = build_faiss_index(all_embeddings)

    print("Performing similarity search...")
    distances, indices = search_faiss_index(faiss_index, opq, all_embeddings, search_chunk_size)  # Pass the opq variable

    print("Processing results...")
    image_id_clusters = process_results(all_image_ids, indices, distances, threshold)
    organize_images(images_to_paths, image_directory, image_id_clusters, damaged_image_ids)

def check_and_load_embeddings(embeddings_file):
    if embeddings_file.exists():
        use_existing_embeddings = input("Embeddings file found. Do you want to use existing embeddings? (Y/N) ").strip().lower()
        if use_existing_embeddings in ('', 'y', 'yes'):
            print("Loading embeddings from file...")
            all_embeddings = np.load(embeddings_file)
            return False
    return True

def get_images_to_paths(image_directory, allowed_extensions):
    images_to_paths = {
        image_path.stem: image_path
        for image_path in image_directory.iterdir()
        if image_path.suffix.lower() in allowed_extensions
    }
    return images_to_paths, list(images_to_paths.keys())

def generate_embeddings(all_image_ids, images_to_paths, model, processor, device, batch_size, regenerate_embeddings, embeddings_file):
    if not regenerate_embeddings:
        return set(), np.load(embeddings_file)

    damaged_image_ids, all_embeddings = set(), []
    progress_bar = tqdm(total=len(all_image_ids), desc="Generating CLIP embeddings")

    for i in range(0, len(all_image_ids), batch_size):
        batch_image_ids, batch_images = process_image_batch(all_image_ids, i, batch_size, images_to_paths, damaged_image_ids)
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        all_embeddings.extend(outputs.cpu().numpy())
        progress_bar.update(len(batch_image_ids))

    progress_bar.close()
    return damaged_image_ids, all_embeddings

def process_image_batch(all_image_ids, start_idx, batch_size, images_to_paths, damaged_image_ids):
    batch_image_ids = all_image_ids[start_idx: start_idx + batch_size]
    batch_images = []

    for image_id in batch_image_ids:
        try:
            image = Image.open(images_to_paths[image_id])
            image.load()
            batch_images.append(image)
        except OSError:
            print(f"\nError processing image {images_to_paths[image_id]}, marking as corrupted.")
            damaged_image_ids.add(image_id)

    return batch_image_ids, batch_images

def build_faiss_index(all_embeddings):
    embeddings = np.array(all_embeddings).astype('float32')
    n_dimensions = embeddings.shape[1]
    nlist = int(4 * np.sqrt(len(embeddings)))
    m = 64
    D = 4 * m
    n_bits = 8
    opq = faiss.OPQMatrix(n_dimensions, m, D)
    opq.train(embeddings)
    embeddings = opq.apply_py(embeddings)
    quantizer = faiss.IndexHNSWFlat(D, 32)
    quantizer.hnsw.efConstruction = 40
    faiss_index = faiss.IndexIVFPQ(quantizer, D, nlist, m, n_bits)
    faiss_index.opq = opq
    faiss_index.train(embeddings)
    faiss_index.add(embeddings)

    return faiss_index, opq

def process_results(all_image_ids, indices, distances, threshold):
    image_id_clusters = defaultdict(set)
    image_to_cluster_mapping = dict()

    def merge_clusters(cluster_id1, cluster_id2):
        if cluster_id1 == cluster_id2:
            return

        if len(image_id_clusters[cluster_id1]) < len(image_id_clusters[cluster_id2]):
            cluster_id1, cluster_id2 = cluster_id2, cluster_id1

        new_cluster = image_id_clusters[cluster_id1].union(image_id_clusters[cluster_id2])
        image_pairs = [(image1, image2) for image1 in new_cluster for image2 in new_cluster if image1 < image2]

        for image_pair in image_pairs:
            index1, index2 = all_image_ids.index(image_pair[0]), all_image_ids.index(image_pair[1])
            if distances[index1, index2] >= threshold and distances[index2, index1] >= threshold:
                return

        image_id_clusters[cluster_id1].update(image_id_clusters[cluster_id2])
        for image_id in image_id_clusters[cluster_id2]:
            image_to_cluster_mapping[image_id] = cluster_id1

        del image_id_clusters[cluster_id2]

    for i, (image_id, row_indices, row_distances) in enumerate(zip(all_image_ids, indices, distances)):
        for j, (neighbor_index, distance) in enumerate(zip(row_indices, row_distances)):
            if i == j:
                continue

            if distance < threshold:
                image_id1, image_id2 = image_id, all_image_ids[neighbor_index]

                if image_id1 in image_to_cluster_mapping and image_id2 in image_to_cluster_mapping:
                    cluster_id1, cluster_id2 = image_to_cluster_mapping[image_id1], image_to_cluster_mapping[image_id2]
                    merge_clusters(cluster_id1, cluster_id2)

                elif image_id1 in image_to_cluster_mapping:
                    cluster_id = image_to_cluster_mapping[image_id1]
                    image_id_clusters[cluster_id].add(image_id2)
                    image_to_cluster_mapping[image_id2] = cluster_id

                elif image_id2 in image_to_cluster_mapping:
                    cluster_id = image_to_cluster_mapping[image_id2]
                    image_id_clusters[cluster_id].add(image_id1)
                    image_to_cluster_mapping[image_id1] = cluster_id

                else:
                    cluster_id = len(image_id_clusters)
                    image_id_clusters[cluster_id].add(image_id1)
                    image_id_clusters[cluster_id].add(image_id2)
                    image_to_cluster_mapping[image_id1] = cluster_id
                    image_to_cluster_mapping[image_id2] = cluster_id

    return image_id_clusters

def organize_images(images_to_paths, image_directory, image_id_clusters, damaged_image_ids):
    for idx, image_id_cluster in enumerate(image_id_clusters.values()):
        if len(image_id_cluster) < 2:
            continue

        move_images_to_directory(image_directory, f"cluster_{idx}", image_id_cluster, images_to_paths)

    unique_image_ids = set(images_to_paths.keys()) - set(damaged_image_ids) - {image_id for cluster in image_id_clusters.values() for image_id in cluster if len(cluster) >= 2}
    move_images_to_directory(image_directory, "unique", unique_image_ids, images_to_paths)

    if damaged_image_ids:
        move_images_to_directory(image_directory, "corrupted", damaged_image_ids, images_to_paths)

def move_images_to_directory(image_directory, folder_name, image_ids, images_to_paths):
    output_directory = image_directory / folder_name
    output_directory.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        source = images_to_paths[image_id]
        destination = output_directory / source.name
        shutil.move(source, destination)

def search_faiss_index(faiss_index, opq, all_embeddings, search_chunk_size):
    distances = np.empty((0, len(all_embeddings)), dtype='float32')
    indices = np.empty((0, len(all_embeddings)), dtype='int64')

    for i in tqdm(range(0, len(all_embeddings), search_chunk_size), desc="Searching FAISS index"):
        chunk = np.array(all_embeddings[i:i + search_chunk_size], dtype='float32')
        chunk = opq.apply_py(chunk)  # Apply the OPQ transformation
        chunk_distances, chunk_indices = faiss_index.search(chunk, len(all_embeddings))
        distances = np.concatenate([distances, chunk_distances])
        indices = np.concatenate([indices, chunk_indices])

    return distances, indices

def main():
    parser = argparse.ArgumentParser(description="Finding conceptually similar images using CLIP and FAISS-based similarity search.")
    parser.add_argument("--image_directory", type=Path, required=True, help="Path to the directory containing the images to cluster.")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14-336", help="CLIP model to use for generating embeddings. Options listed from least to most demanding: openai/clip-vit-base-patch16, openai/clip-vit-base-patch32, openai/clip-vit-large-patch14, openai/clip-vit-large-patch14-336 (default)")
    parser.add_argument("--threshold", type=float, default=20, help="Threshold for similarity search. Lower values will reduce false positives but may miss more. (default: 20)")
    parser.add_argument("--batch_size", type=int, default=192, help="Batch size for generating CLIP embeddings. Higher values will require more VRAM. (default: 192)")
    parser.add_argument("--search_chunk_size", type=int, default=1024, help="Chunk size for searching the FAISS index. Lower values will reduce memory consumption. (default: 1024)")
    args = parser.parse_args()

    process_images(args.image_directory, args.clip_model, args.threshold, args.batch_size, args.search_chunk_size)

if __name__ == "__main__":
    main()