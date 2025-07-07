# -----------------------------------------------
# SpaceMatch: AI-Powered Property Matching API
# -----------------------------------------------
# Description:
# This is a full-featured FastAPI backend designed for SpaceMatch,
# supporting natural language property search with filters, semantic matching,
# and AI-enhanced query handling. It integrates a custom vector store
# and exposes various endpoints for chat, search, health, stats, and more.
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

from vector_store_loader import SpaceMatchVectorStore

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI App Initialization
# -----------------------------
app = FastAPI(
    title="SpaceMatch API",
    description="AI-powered property matching chat endpoint",
    version="1.0.0"
)

# Enable CORS for all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global vector store instance (initialized at startup)
vector_store = None

# -----------------------------
# Pydantic Models for API Schema
# -----------------------------
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

# -----------------------------
# Filter Handler
# -----------------------------
class PropertyFilter:
    def __init__(self, query: str):
        self.query = query.lower()
        self.filters = self._parse_filters()

    def _parse_filters(self) -> Dict[str, Any]:
        filters = {}

        # Extract max price
        for keyword in ["under", "below", "less than", "max", "maximum"]:
            if keyword in self.query:
                try:
                    idx = self.query.split().index(keyword.split()[-1])
                    price = self.query.split()[idx + 1].replace("$", "").replace(",", "")
                    filters['max_price'] = int(price)
                    break
                except (ValueError, IndexError):
                    continue

        # Bedroom filter
        for pattern in ["studio", "1 bedroom", "2 bedroom", "3 bedroom", "4 bedroom", "5 bedroom"]:
            if pattern in self.query:
                filters['bedrooms'] = 0 if pattern == "studio" else int(pattern.split()[0])
                break

        # Property type filter
        for ptype in ["apartment", "house", "condo", "townhouse", "studio"]:
            if ptype in self.query:
                filters['property_type'] = ptype
                break

        # Amenities
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
        # Apply filters to property listings
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

# -----------------------------
# Query Enhancement Logic
# -----------------------------
class QueryProcessor:
    def enhance_query(self, query: str) -> str:
        # Add semantic context to improve matching
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

# -----------------------------
# Vector Store Initialization
# -----------------------------
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

# -----------------------------
# Response Message Generator
# -----------------------------
def generate_response_message(query: str, properties: List[Dict], total_found: int) -> str:
    if not properties:
        return f"No properties matched your search for '{query}'. Try adjusting your filters."
    cities = list(set(p['city'] for p in properties))
    types = list(set(p['property_type'] for p in properties))
    min_rent = min(p['monthly_rent'] for p in properties)
    max_rent = max(p['monthly_rent'] for p in properties)
    return f"Found {len(properties)} {', '.join(types)} in {', '.join(cities)} from ₹{min_rent:,} to ₹{max_rent:,}/month."

# -----------------------------
# NumPy-safe JSON Conversion
# -----------------------------
def convert_np_types(obj):
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

# -----------------------------
# Startup Event
# -----------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Starting SpaceMatch API...")
    success = initialize_vector_store()
    if not success:
        logger.error("Vector store initialization failed. API may not function properly.")

# -----------------------------
# Chat Endpoint
# -----------------------------

@app.get("/chat", response_model=ChatResponse)
async def chat(query: str, max_results: int = 10, user_id: Optional[str] = None):
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        qp = QueryProcessor()
        enhanced_query = qp.enhance_query(query)

        results = vector_store.search(enhanced_query, k=max_results)
        pf = PropertyFilter(query)
        filtered = pf.apply_filters(results)

        response = {
            "message": generate_response_message(query, filtered, len(filtered)),
            "properties": [convert_np_types(p) for p in filtered],
            "total_found": len(filtered),
            "query_processed": enhanced_query,
            "timestamp": datetime.now().isoformat()
        }
        return JSONResponse(content=response)
    except Exception as e:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
