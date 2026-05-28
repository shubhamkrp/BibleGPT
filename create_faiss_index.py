# build_index.py (Run this locally on your computer)
import json
import pickle
import faiss
from sentence_transformers import SentenceTransformer

BIBLE_BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth",
    "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah",
    "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah",
    "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah",
    "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi", "Matthew", "Mark", "Luke",
    "John", "Acts", "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians", "Philippians",
    "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon",
    "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation"
]

print("1. Loading Bible JSON...")
with open("bible.json", "r", encoding="utf-8") as f:
    bible_data = json.load(f)

bible_documents = []

print("2. Formatting verses...")
for b_idx, book in enumerate(bible_data):
    book_name = BIBLE_BOOKS[b_idx] if b_idx < len(BIBLE_BOOKS) else f"Book {b_idx+1}"
    for c_idx, chapter in enumerate(book):
        for v_idx, verse in enumerate(chapter):
            ref = f"{book_name} {c_idx+1}:{v_idx+1}"
            bible_documents.append(f"{ref} - {verse}")

print(f"Total verses processed: {len(bible_documents)}")

print("3. Loading Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("4. Generating Embeddings (This will take a minute or two)...")
embeddings = embedder.encode(bible_documents, convert_to_numpy=True, show_progress_bar=True)

print("5. Building FAISS Index...")
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings)

print("6. Saving to disk...")
faiss.write_index(faiss_index, "bible_faiss.index")
with open("bible_docs.pkl", 'wb') as f:
    pickle.dump(bible_documents, f)

print("✅ Success! You can now upload 'bible_faiss.index' and 'bible_docs.pkl' to your Hugging Face Space.")
