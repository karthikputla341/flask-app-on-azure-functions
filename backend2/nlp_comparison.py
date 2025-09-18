import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textdistance
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class DocumentComparator:
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(stop_words='english')
        # Simple word2vec model for semantic similarity (would be better pre-trained in production)
        self.word2vec_model = None
        
    def preprocess_text(self, text):
        """Clean and preprocess text for comparison"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
    
    def extract_entries(self, text):
        """Extract individual entries from document text"""
        # Split by common delimiters (newlines, commas, semicolons, etc.)
        entries = re.split(r'[\n,;]+', text)
        # Clean each entry
        entries = [entry.strip() for entry in entries if entry.strip()]
        return entries
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two text entries using multiple methods"""
        if not text1 or not text2:
            return 0.0
            
        # Preprocess texts
        processed1 = self.preprocess_text(text1)
        processed2 = self.preprocess_text(text2)
        
        # If texts are identical after preprocessing
        if processed1 == processed2:
            return 1.0
            
        # Calculate multiple similarity scores
        similarities = []
        
        # 1. Jaccard similarity
        set1, set2 = set(processed1.split()), set(processed2.split())
        if set1 or set2:  # Avoid division by zero
            jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
            similarities.append(jaccard)
        
        # 2. Cosine similarity with TF-IDF
        try:
            tfidf_matrix = self.vectorizer.fit_transform([processed1, processed2])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            similarities.append(cosine_sim)
        except:
            pass
            
        # 3. Levenshtein similarity (normalized)
        max_len = max(len(processed1), len(processed2))
        if max_len > 0:
            lev_sim = 1 - (textdistance.levenshtein(processed1, processed2) / max_len)
            similarities.append(lev_sim)
        
        # Return average of all calculated similarities
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def compare_documents(self, doc1_text, doc2_text, similarity_threshold=0.7):
        """Compare two documents and find matching entries"""
        # Extract entries from both documents
        doc1_entries = self.extract_entries(doc1_text)
        doc2_entries = self.extract_entries(doc2_text)
        
        # Preprocess all entries for better matching
        doc1_processed = [self.preprocess_text(entry) for entry in doc1_entries]
        doc2_processed = [self.preprocess_text(entry) for entry in doc2_entries]
        
        # Find matches
        matches = []
        doc1_only = []
        doc2_only = list(doc2_entries)  # Start with all entries from doc2
        
        for i, entry1 in enumerate(doc1_entries):
            best_match_index = -1
            best_similarity = 0
            
            for j, entry2 in enumerate(doc2_entries):
                similarity = self.calculate_similarity(entry1, entry2)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_index = j
            
            if best_similarity >= similarity_threshold and best_match_index != -1:
                matches.append({
                    'doc1_entry': entry1,
                    'doc2_entry': doc2_entries[best_match_index],
                    'similarity': best_similarity,
                    'status': 'Match' if best_similarity > 0.9 else 'Partial'
                })
                # Remove from doc2_only list
                if best_match_index < len(doc2_only):
                    doc2_only[best_match_index] = None
            else:
                doc1_only.append(entry1)
        
        # Clean up doc2_only (remove None values)
        doc2_only = [entry for entry in doc2_only if entry is not None]
        
        # Prepare results
        results = {
            'total_doc1_entries': len(doc1_entries),
            'total_doc2_entries': len(doc2_entries),
            'matched_entries': matches,
            'doc1_only_entries': doc1_only,
            'doc2_only_entries': doc2_only,
            'match_percentage': len(matches) / len(doc1_entries) * 100 if doc1_entries else 0
        }
        
        return results