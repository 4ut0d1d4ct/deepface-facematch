import os
import logging
import argparse
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_subject_embedding(subjectPath, model_name='VGG-Face'):
    """
    Get the embedding for the subject image.

    Args:
        subjectPath (str): Path to the subject image.
        model_name (str): Name of the model to use for embedding.

    Returns:
        embedding: Embedding of the subject image.
    """
    try:
        return DeepFace.represent(img_path=subjectPath, model_name=model_name)
    except Exception as e:
        logger.error(f"Error getting embedding for subject image: {e}")
        return None

def get_doubles_embeddings(doublesDir, model_name='VGG-Face'):
    """
    Get the embeddings for all images in the doubles directory.

    Args:
        doublesDir (str): Path to the doubles directory.
        model_name (str): Name of the model to use for embedding.

    Returns:
        embeddings: Dictionary of image filenames and their embeddings.
    """
    embeddings = {}
    for filename in os.listdir(doublesDir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            imagePath = os.path.join(doublesDir, filename)
            try:
                embedding = DeepFace.represent(img_path=imagePath, model_name=model_name)[0]["embedding"]
                embeddings[filename] = embedding
            except Exception as e:
                logger.error(f"Error getting embedding for double image {filename}: {e}")
    return embeddings

def main(subjectPath="data/subject/stock_twins_1.jpg", doublesPath="data/doubles", model_name='VGG-Face', verbose=True):
    """
    Main function to get embeddings and calculate similarities.

    Args:
        subjectPath (str): Path to the subject image.
        doublesPath (str): Path to the doubles directory.
        model_name (str): Name of the model to use for embedding.
        verbose (bool): Whether to print verbose output.

    Returns:
        rankings: Sorted dict of doubles images ranked by similarity to the subject image.
    """
    if verbose:
        logger.info(f"Getting the embedding for the subject image: {subjectPath}")
    subjectEmbedding = get_subject_embedding(subjectPath, model_name)
    if subjectEmbedding is None:
        return []

    if verbose:
        logger.info(f"Getting the embeddings for the doubles images in: {doublesPath}")
    doublesEmbeddings = get_doubles_embeddings(doublesPath, model_name)

    # Calculate similarities
    similarities = {}
    for imageName, embedding in doublesEmbeddings.items():
        if verbose:
            logger.info(f"Calculating similarity for double: {imageName}")
        similarity = cosine_similarity([subjectEmbedding[0]['embedding']], [embedding])[0][0]
        similarities[imageName] = similarity

    # Sort images by similarity (higher similarity first)
    rankings = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

    if verbose:
        logger.info(f"Doubles images sorted by similarity to the subject image:")
        for imageName, similarity in rankings:
            logger.info(f"{imageName}: {similarity}")

    return rankings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFace Face Match")
    parser.add_argument("--subjectPath", type=str, default="data/subject/stock_twins_1.jpg", help="Path to the subject image")
    parser.add_argument("--doublesPath", type=str, default="data/doubles", help="Path to the doubles directory")
    parser.add_argument("--model_name", type=str, default="VGG-Face", help="Name of the model to use for embedding")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    parser.add_argument("--no-verbose", action='store_false', dest='verbose', help="Disable verbose output")
    parser.set_defaults(verbose=True)
    
    args = parser.parse_args()
    main(args.subjectPath, args.doublesPath, args.model_name, args.verbose)