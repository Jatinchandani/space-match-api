import pandas as pd
import numpy as np
import json
import os
import faiss
import pickle
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpaceMatch")

class SpaceMatchVectorStore:
    """
    Vector store for SpaceMatch property listings using FAISS
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.property_data = []
        self.embeddings = None

        logger.info(f"Initialized with model: {model_name}, dimension: {self.dimension}")

    def create_property_text(self, prop: Dict[str, Any]) -> str:
        amenities = ', '.join(prop.get('amenities', [])) or 'basic amenities'
        features = []
        if prop.get('pet_friendly'): features.append('pet-friendly')
        if prop.get('parking_available'): features.append('parking available')
        if prop.get('furnished'): features.append('furnished')
        if prop.get('utilities_included'): features.append('utilities included')

        text = f"{prop.get('bedrooms', 0)} BHK {prop.get('property_type')} in {prop.get('city')}, " \
               f"{prop.get('state')} {prop.get('zip_code')} with {prop.get('sqft')} sqft, " \
               f"₹{prop.get('monthly_rent')} monthly, {prop.get('bathrooms')} bathrooms, " \
               f"amenities: {amenities}, features: {', '.join(features) or 'standard features'}. " \
               f"{prop.get('description', '')} {prop.get('title', '')}"
        return text

    def load_data(self, filepath: str):
        logger.info(f"Loading data from {filepath}")
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")

        self.property_data = df.to_dict('records')
        for prop in self.property_data:
            if isinstance(prop.get('amenities'), str):
                try:
                    prop['amenities'] = json.loads(prop['amenities'])
                except:
                    try:
                        prop['amenities'] = eval(prop['amenities'])
                    except:
                        prop['amenities'] = []
        logger.info(f"Loaded {len(self.property_data)} records")

    def create_embeddings(self):
        if not self.property_data:
            raise ValueError("No property data loaded.")

        texts = [self.create_property_text(p) for p in self.property_data]
        self.embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        logger.info(f"Generated {len(self.embeddings)} embeddings")

    def build_index(self):
        if self.embeddings is None:
            raise ValueError("Call create_embeddings() first")

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(np.array(self.embeddings).astype('float32'))
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")

    def save_index(self, path='spacematch_index'):
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}_data.pkl", 'wb') as f:
            pickle.dump({
                'property_data': self.property_data,
                'embeddings': self.embeddings,
                'dimension': self.dimension
            }, f)
        logger.info("Index saved successfully")

    def load_index(self, path='spacematch_index'):
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}_data.pkl", 'rb') as f:
            data = pickle.load(f)
            self.property_data = data['property_data']
            self.embeddings = data['embeddings']
            self.dimension = data['dimension']
        logger.info("Index loaded successfully")

    def search(self, query: str, k=10) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("Index not built or loaded")

        query_vector = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_vector.astype('float32'), k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.property_data):
                result = self.property_data[idx].copy()
                result['similarity_score'] = float(score)
                result['search_rank'] = rank + 1
                results.append(result)
        return results

    def get_stats(self) -> Dict[str, Any]:
        if not self.property_data:
            return {"error": "No data loaded"}
        df = pd.DataFrame(self.property_data)
        return {
            'total_properties': len(df),
            'cities': df['city'].nunique(),
            'property_types': df['property_type'].value_counts().to_dict(),
            'avg_rent': df['monthly_rent'].mean(),
            'rent_range': [df['monthly_rent'].min(), df['monthly_rent'].max()],
            'avg_sqft': df['sqft'].mean(),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.dimension
        }

if __name__ == '__main__':
    vs = SpaceMatchVectorStore()
    data_path = 'spacematch_properties.json'

    if not os.path.exists(data_path):
        logger.error("Dataset not found. Please generate it first.")
    else:
        vs.load_data(data_path)
        vs.create_embeddings()
        vs.build_index()
        vs.save_index()

        logger.info("Running sample queries:")
        sample_queries = [
            "2 BHK flat in Bangalore with lift and CCTV",
            "Furnished apartment in Mumbai under ₹30,000",
            "Studio in Pune near park",
            "Vaastu compliant flat in Chennai with power backup"
        ]

        for query in sample_queries:
            print(f"\nQuery: {query}")
            results = vs.search(query, k=3)
            for res in results:
                print(f" - {res['title']} | ₹{res['monthly_rent']} | {res['city']}, {res['state']} | Score: {res['similarity_score']:.3f}")