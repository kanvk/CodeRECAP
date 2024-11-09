from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Document vectorization
def vectorize_documents_tfidf(file_infos):
    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()

    file_contents = [file_info["code_snippet"] for file_info in file_infos]

    tfidf_matrix = vectorizer.fit_transform(file_contents)

    return tfidf_matrix, vectorizer


# Query comparison
def search_tfidf(query, vectorizer, tfidf_matrix, file_infos, k):
    file_names = [file_info["name"] for file_info in file_infos]

    query_tfidf = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # Get file names in descending order of similarity without returning similarity scores
    results = sorted(
        zip(file_names, cosine_similarities), key=lambda x: x[1], reverse=True
    )
    ranked_file_names = [file_name for file_name, _ in results[:k]]

    return ranked_file_names


# Function to read and preprocess Python files
# def load_files_from_directory(directory):
#     file_contents = []
#     filenames = []
#
#     for filename in os.listdir(directory):
#         if filename.endswith(".py"):
#             with open(os.path.join(directory, filename), "r") as file:
#                 content = file.read()
#                 file_contents.append(content)
#                 filenames.append(filename)
#     return file_contents, filenames
