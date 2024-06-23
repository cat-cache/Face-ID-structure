# Face-ID structure
 Facial locking pipeline along with a helper flask API for the same
 
# Pipeline Overview (Colab Notebook)
Objective: Implement face recognition using DeepFace with MTCNN and RetinaFace as detectors (used the DeepFace library).
Dependencies: Requires OpenCV, DeepFace, and associated libraries.
Execution: Includes preprocessing, face detection using MTCNN and RetinaFace, embedding extraction with ArcFace model.
Output: Generates face embeddings and provides cosine similarity score.
# Flask API Conversion
Purpose: Convert the Colab notebook pipeline into a RESTful Flask API for making face-id locks for applications.
Endpoints: Includes /extract_embeddings_from_video, /extract_embeddings_from_image, /compare_embeddings.
Usage: Accepts video/image inputs, extracts embeddings, and calculates similarity scores.
Integration: Designed for integration into web applications and services.
Dependencies
Requires opencv-python-headless, deepface, and other specified libraries.



Future Enhancements
Potential enhancements include real-time video processing, scalability improvements, and additional face recognition models + streamline this API into not only providing embeddings but "doing the locking"
