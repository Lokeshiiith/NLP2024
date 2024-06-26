# README

## Singular Value Decomposition (SVD) and Classification

### Files Included:

- `svd.py`: Python script containing code for Singular Value Decomposition (SVD) and word embeddings generation.
- `svd-classification.py`: Python script implementing a classification model using SVD-generated word embeddings.
- `svd-similarity.png`: Image file showing 10 similar words for each of 5 different input words.

### Overview:

This repository contains code related to Singular Value Decomposition (SVD) for word embeddings generation and a classification model based on these embeddings.

### Instructions:

1. **SVD Script (`svd.py`):**

   - This script performs Singular Value Decomposition (SVD) on word co-occurrence matrices to generate word embeddings.
   - The generated word embeddings are saved as `svd-word-vectors.pt`.
   - Additionally, `word2index.json` and `index2word.json` files are created to store word-to-index and index-to-word mappings.

2. **Classification Script (`svd-classification.py`):**

   - The classification script uses the word embeddings generated by SVD for text classification.
   - It loads the embeddings and performs classification tasks.
   - The script outputs test metrics such as Test Loss, Test Accuracy, Precision, Recall, F1 Score, and the Confusion Matrix.

3. **Similarity Visualization (`svd-similarity.png`):**

   - ![Similar Words](svd.png)
   - This image file shows 10 similar words for each of 5 different input words.
   - The similar words are determined based on the cosine similarity of word embeddings obtained from SVD.

### Running the Scripts:

1. Ensure you have the necessary libraries installed (e.g., NumPy, Pandas, Matplotlib, Scikit-Learn, Torch).
2. Run `svd.py` to generate word embeddings using SVD and save the embeddings, word-to-index, and index-to-word mappings.
3. Run `svd-classification.py` to perform classification tasks using the generated embeddings.
4. Review the output metrics and visualizations for insights into word similarities and classification performance.

### Additional Notes:

- Adjust parameters such as window size, embedding size, and input words as needed in the scripts.
- Customize the color map, plot layout, and legend placement in visualizations as desired.
- Feel free to explore different input words and experiment with various classification tasks using SVD-generated embeddings.

For more details, refer to the code comments and documentation within the scripts.

## Note

The model files for this assignment are available for download at " https://github.com/Lokeshiiith/NLP2024/tree/main/Assignment-3 " [here](lhttps://github.com/Lokeshiiith/NLP2024/tree/main/Assignment-3). Due to file size limitations on Moodle, we have provided the files on GitHub for your convenience.
