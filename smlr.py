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

class ImageClusterer:
    ALLOWED_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}
    DEVICE = "cuda"

    def __init__(self, image_directory, clip_model, threshold, batch_size, search_chunk_size):
        self.image_directory = Path(image_directory)
        self.clip_model = clip_model
        self.threshold = threshold
        self.batch_size = batch_size
        self.search_chunk_size = search_chunk_size
        self.embeddings_file = self.image_directory / 'embeddings.npy'
        self.model = CLIPModel.from_pretrained(clip_model).to(self.DEVICE)
        self.processor = CLIPProcessor.from_pretrained(clip_model)
        self.images_to_paths, self.all_image_ids = self.get_image_path_dictionary()
        self.cluster_counter = 1

    def get_image_path_dictionary(self):
        images_to_paths = {
            image_path.stem: image_path
            for image_path in self.image_directory.iterdir()
            if image_path.suffix.lower() in self.ALLOWED_EXTENSIONS
        }
        all_image_ids = list(images_to_paths.keys())
        return images_to_paths, all_image_ids

    def generate_and_organise_clusters(self):
        corrupted_image_ids, all_embeddings = self.load_or_generate_embeddings()
        if self.regenerate_embeddings:
            np.save(self.embeddings_file, all_embeddings)
        print("Creating FAISS index...")
        faiss_index, opq = self.build_faiss_index(all_embeddings)
        print("Performing similarity search...")
        distances, indices = self.search_faiss_index(faiss_index, opq, all_embeddings, self.search_chunk_size)
        print("Processing results...")
        image_id_clusters = self.organise_identified_clusters(indices, distances)
        self.group_and_move_images(image_id_clusters, corrupted_image_ids)

    def load_or_generate_embeddings(self):
        if self.embeddings_file.exists():
            use_existing_embeddings = input("Embeddings file found. Do you want to use existing embeddings? (Y/N) ").strip().lower()
            if use_existing_embeddings in ('', 'y', 'yes'):
                print("Loading embeddings from file...")
                self.all_embeddings = np.load(self.embeddings_file)
                self.regenerate_embeddings = False
                return set(), self.all_embeddings
        self.regenerate_embeddings = True
        corrupted_image_ids = set()
        self.all_embeddings = []
        progress_bar = tqdm(total=len(self.all_image_ids), desc="Generating CLIP embeddings")
        for i in range(0, len(self.all_image_ids), self.batch_size):
            batch_image_ids = self.all_image_ids[i: i + self.batch_size]
            batch_images = []
            for image_id in batch_image_ids:
                try:
                    image = Image.open(self.images_to_paths[image_id])
                    image.load()
                    batch_images.append(image)
                except OSError:
                    print(f"\nError processing image {self.images_to_paths[image_id]}, marking as corrupted.")
                    corrupted_image_ids.add(image_id)
            inputs = self.processor(images=batch_images, return_tensors="pt", padding=True).to(self.DEVICE)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            self.all_embeddings.extend(outputs.cpu().numpy())
            progress_bar.update(len(batch_image_ids))
        progress_bar.close()
        return corrupted_image_ids, np.array(self.all_embeddings)

    def build_faiss_index(self, all_embeddings):
        embeddings_float32 = np.array(all_embeddings).astype('float32')
        num_dimensions = embeddings_float32.shape[1]
        num_cells = int(4 * np.sqrt(len(embeddings_float32)))
        num_subquantizers = 64
        num_centroids = 4 * num_subquantizers
        bits_per_code = 8
        product_quantizer_matrix = faiss.OPQMatrix(num_dimensions, num_subquantizers, num_centroids)
        product_quantizer_matrix.train(embeddings_float32)
        embeddings_transformed = product_quantizer_matrix.apply_py(embeddings_float32)
        hnsw_quantizer = faiss.IndexHNSWFlat(num_centroids, 32)
        hnsw_quantizer.hnsw.efConstruction = 40
        ivfpq_index = faiss.IndexIVFPQ(hnsw_quantizer, num_centroids, num_cells, num_subquantizers, bits_per_code)
        ivfpq_index.opq = product_quantizer_matrix
        ivfpq_index.train(embeddings_transformed)
        ivfpq_index.add(embeddings_transformed)
        return ivfpq_index, product_quantizer_matrix

    def search_faiss_index(self, faiss_index, opq, all_embeddings, search_chunk_size):
        distances, indices = [], []
        for i in tqdm(range(0, len(all_embeddings), search_chunk_size), desc="Searching FAISS index"):
            chunk_distances, chunk_indices = faiss_index.search(opq.apply_py(np.array(all_embeddings[i:i + search_chunk_size], dtype='float32')), len(all_embeddings))
            distances.append(chunk_distances)
            indices.append(chunk_indices)
        return np.concatenate(distances), np.concatenate(indices)

    def organise_identified_clusters(self, indices, distances):
        image_id_clusters = defaultdict(set)
        image_to_cluster_mapping = defaultdict(lambda: len(image_to_cluster_mapping))
        for i, (image_id, row_indices, row_distances) in enumerate(zip(self.all_image_ids, indices, distances)):
            for j, (neighbor_index, distance) in enumerate(zip(row_indices, row_distances)):
                if i == j or distance >= self.threshold:
                    continue
                image_id1, image_id2 = image_id, self.all_image_ids[neighbor_index]
                cluster_id1, cluster_id2 = image_to_cluster_mapping[image_id1], image_to_cluster_mapping[image_id2]
                if cluster_id1 != cluster_id2:
                    if len(image_id_clusters[cluster_id1]) > len(image_id_clusters[cluster_id2]):
                        cluster_id1, cluster_id2 = cluster_id2, cluster_id1
                    image_id_clusters[cluster_id1].update(image_id_clusters[cluster_id2])
                    for image_id in image_id_clusters[cluster_id1]:
                        image_to_cluster_mapping[image_id] = cluster_id1
                    del image_id_clusters[cluster_id2]
                else:
                    image_id_clusters[cluster_id1].add(image_id1)
                    image_id_clusters[cluster_id1].add(image_id2)
        return image_id_clusters

    def group_and_move_images(self, image_id_clusters, corrupted_image_ids):
        for image_id_cluster in image_id_clusters.values():
            if len(image_id_cluster) < 2:
                continue
            self._move_images_to_directory(f"cluster_{self.cluster_counter}", image_id_cluster)
            self.cluster_counter += 1
        unique_image_ids = set(self.images_to_paths.keys()) - corrupted_image_ids - {image_id for cluster in image_id_clusters.values() for image_id in cluster if len(cluster) >= 2}
        self._move_images_to_directory("unique", unique_image_ids)
        if corrupted_image_ids:
            self._move_images_to_directory("corrupted", corrupted_image_ids)

    def _move_images_to_directory(self, folder_name, image_ids):
        output_directory = self.image_directory / folder_name
        output_directory.mkdir(parents=True, exist_ok=True)
        for image_id in image_ids:
            shutil.move(self.images_to_paths[image_id], output_directory / self.images_to_paths[image_id].name)

def main():
    parser = argparse.ArgumentParser(description="Cluster images based on their conceptual similarity using CLIP and FAISS.")
    parser.add_argument("--image_directory", type=Path, required=True, help="Path to the directory containing the images to cluster.")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14-336", help="CLIP model to use for generating embeddings. Options listed from least to most demanding: openai/clip-vit-base-patch16, openai/clip-vit-base-patch32, openai/clip-vit-large-patch14, openai/clip-vit-large-patch14-336 (default)")
    parser.add_argument("--threshold", type=float, default=20, help="Threshold for similarity search. Lower values will reduce false positives but may miss more. (default: 20)")
    parser.add_argument("--batch_size", type=int, default=192, help="Batch size for generating CLIP embeddings. Higher values will require more VRAM. (default: 192)")
    parser.add_argument("--search_chunk_size", type=int, default=512, help="Chunk size for searching the FAISS index. Lower values will reduce memory consumption. (default: 512)")
    args = parser.parse_args()

    image_clusterer = ImageClusterer(args.image_directory, args.clip_model, args.threshold, args.batch_size, args.search_chunk_size)
    image_clusterer.generate_and_organise_clusters()

if __name__ == "__main__":
    main()