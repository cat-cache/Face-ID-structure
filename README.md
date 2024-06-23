# Face-ID Structure

A facial recognition pipeline with a supporting Flask API for implementing face ID locks in applications.

## Pipeline Overview (Colab Notebook)

**Objective:** Implement face recognition using DeepFace with MTCNN and RetinaFace as detectors.

**Dependencies:** Requires OpenCV, DeepFace, and associated libraries.

**Execution:** 
- Preprocessing: Prepare video and image inputs for processing (extracting sutable frames from video before converting to embeddings)
- Face Detection: Use MTCNN and RetinaFace for detecting faces.
- Embedding Extraction: Use the ArcFace model to generate face embeddings.

**Output:** Generates face embeddings and provides a cosine similarity score.

## Flask API Conversion

**Purpose:** Convert the Colab notebook pipeline into a RESTful Flask API for facial recognition applications.

**Endpoints:**
- `/extract_embeddings_from_video`: Extracts face embeddings from a video file.
- `/extract_embeddings_from_image`: Extracts face embeddings from an image file.
- `/compare_embeddings`: Compares two sets of embeddings and returns a similarity score.

**Usage:** Accepts video/image inputs, extracts embeddings, and calculates similarity scores.



## Usage

1. **Run the Flask API:**
    ```sh
    python main.py
    ```

2. **API Endpoints:**

    - **Extract Embeddings from Video:**
        ```sh
        POST /extract_embeddings_from_video
        ```
        - Request: Multipart/form-data with 'video' file.
        - Response: JSON with extracted embeddings.

    - **Extract Embeddings from Image:**
        ```sh
        POST /extract_embeddings_from_image
        ```
        - Request: Multipart/form-data with 'image' file.
        - Response: JSON with extracted embedding.

    - **Compare Embeddings:**
        ```sh
        POST /compare_embeddings
        ```
        - Request: JSON with 'embedding1' and 'embedding2'.
        - Response: JSON with similarity score.   

    - **Home:**
        ```sh
        GET /
        ```
        - Response: "Welcome message"

## Dependencies

- Flask
- OpenCV (Headless)
- DeepFace
- Numpy
- Matplotlib
- Pillow

## Future Enhancements

- Real-time video processing.
- Scalability improvements.
- Additional face recognition models.
- Streamline the API to not only provide embeddings but also perform face-ID locking.


## Acknowledgments

- [DeepFace](https://github.com/serengil/deepface): A lightweight face recognition and facial attribute analysis framework for Python.
