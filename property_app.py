# -----------------------------------------------
# SpaceMatch: AI-Powered Property Matching API + Gradio UI
# -----------------------------------------------

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
import numpy as np
import logging
from datetime import datetime
import gradio as gr

from vector_store_loader import SpaceMatchVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SpaceMatch API",
    description="AI-powered property matching chat endpoint",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = None

class ChatRequest(BaseModel):
    message: str
    max_results: Optional[int] = 10
    user_id: Optional[str] = None

class PropertyResult(BaseModel):
    listing_id: str
    title: str
    description: str
    monthly_rent: int
    bedrooms: int
    bathrooms: int
    sqft: int
    city: str
    state: str
    property_type: str
    amenities: List[str]
    similarity_score: float
    full_address: str
    available_date: str
    contact_email: str
    contact_phone: str

class ChatResponse(BaseModel):
    message: str
    properties: List[PropertyResult]
    total_found: int
    query_processed: str
    timestamp: str

class PropertyFilter:
    def __init__(self, query: str):
        self.query = query.lower()
        self.filters = self._parse_filters()

    def _parse_filters(self) -> Dict[str, Any]:
        filters = {}
        for keyword in ["under", "below", "less than", "max", "maximum"]:
            if keyword in self.query:
                try:
                    idx = self.query.split().index(keyword.split()[-1])
                    price = self.query.split()[idx + 1].replace("$", "").replace(",", "")
                    filters['max_price'] = int(price)
                    break
                except (ValueError, IndexError):
                    continue
        for pattern in ["studio", "1 bedroom", "2 bedroom", "3 bedroom", "4 bedroom", "5 bedroom"]:
            if pattern in self.query:
                filters['bedrooms'] = 0 if pattern == "studio" else int(pattern.split()[0])
                break
        for ptype in ["apartment", "house", "condo", "townhouse", "studio"]:
            if ptype in self.query:
                filters['property_type'] = ptype
                break
        filters['amenities'] = []
        for keyword, amenity in {
            'pet': 'pet_friendly',
            'parking': 'parking_available',
            'furnished': 'furnished',
            'utilities': 'utilities_included',
            'pool': 'pool',
            'gym': 'gym'
        }.items():
            if keyword in self.query:
                filters['amenities'].append(amenity)
        return filters

    def apply_filters(self, properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = properties
        if 'max_price' in self.filters:
            filtered = [p for p in filtered if p.get('monthly_rent', 0) <= self.filters['max_price']]
        if 'bedrooms' in self.filters:
            filtered = [p for p in filtered if p.get('bedrooms', 0) == self.filters['bedrooms']]
        if 'property_type' in self.filters:
            filtered = [p for p in filtered if p.get('property_type', '').lower() == self.filters['property_type']]
        for amenity in self.filters.get('amenities', []):
            if amenity in ['pet_friendly', 'parking_available', 'furnished', 'utilities_included']:
                filtered = [p for p in filtered if p.get(amenity, False)]
            else:
                filtered = [p for p in filtered if amenity.lower() in [a.lower() for a in p.get('amenities', [])]]
        return filtered

class QueryProcessor:
    def enhance_query(self, query: str) -> str:
        enhanced = query
        if any(x in query.lower() for x in ['cheap', 'affordable', 'budget']):
            enhanced += " low cost economical"
        if any(x in query.lower() for x in ['luxury', 'upscale', 'premium']):
            enhanced += " high-end expensive luxury amenities"
        if 'family' in query.lower():
            enhanced += " spacious multiple bedrooms safe neighborhood"
        if 'young professional' in query.lower() or 'professional' in query.lower():
            enhanced += " modern amenities downtown urban"
        return enhanced

def initialize_vector_store():
    global vector_store
    try:
        vector_store = SpaceMatchVectorStore()
        if os.path.exists('spacematch_index.faiss'):
            logger.info("Loading existing vector store index...")
            vector_store.load_index('spacematch_index')
        else:
            logger.info("Creating new vector store index...")
            if os.path.exists('spacematch_properties.json'):
                vector_store.load_data('spacematch_properties.json')
            elif os.path.exists('spacematch_properties.csv'):
                vector_store.load_data('spacematch_properties.csv')
            else:
                raise FileNotFoundError("No property data file found")
            vector_store.create_embeddings()
            vector_store.build_index()
            vector_store.save_index('spacematch_index')
        logger.info("Vector store initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        return False

def chat_with_user(input_query):
    if not vector_store:
        return "Vector store not loaded. Please check logs."
    try:
        processor = QueryProcessor()
        enhanced_query = processor.enhance_query(input_query)
        results = vector_store.search(enhanced_query, k=10)
        filtered = PropertyFilter(input_query).apply_filters(results)
        display = "\n\n".join([
            f"🏠 {p['title']} — ₹{p['monthly_rent']}/mo in {p['city']}, {p['state']}\n{p['description']}\nContact: {p['contact_email']}"
            for p in filtered[:5]
        ])
        return display if display else "No matches found. Try a different query."
    except Exception as e:
        logger.error(f"Gradio chat error: {str(e)}")
        return f"❌ Error: {str(e)}"

@app.on_event("startup")
async def startup_event():
    logger.info("Starting SpaceMatch API...")
    initialize_vector_store()

def launch_gradio():
    iface = gr.Interface(
        fn=chat_with_user,
        inputs=gr.Textbox(lines=2, placeholder="e.g. Furnished 2BHK office in Gurgaon under ₹1L"),
        outputs="text",
        title="SpaceMatch Property Finder",
        description="Enter a natural query to find matching properties",
    )
    iface.launch()

if __name__ == "__main__":
    import sys
    if "gradio" in sys.argv:
        initialize_vector_store()
        launch_gradio()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)