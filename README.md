# SpaceMatch – AI-Powered Property Search

**SpaceMatch** is a prototype for an AI-powered property search platform that understands natural language and matches users to relevant real estate listings using vector similarity and filters.

> Built as part of a technical challenge for the Founding Engineer role at **SpaceMatch**.

---

## Features

- 🔍 Natural language search for Indian property listings
- 🤖 Semantic matching using `SentenceTransformers` + FAISS
- 🧠 Smart query enhancement and keyword expansion
- 🎯 Filtering support (price, property type, amenities, etc.)
- 💬 Chat-style frontend UI
- 🌐 Deployed backend & frontend

---

## 🧠 Tech Stack

| Layer     | Tech                               |
|-----------|------------------------------------|
| Backend   | FastAPI, FAISS, SentenceTransformers |
| Frontend  | HTML, CSS, Vanilla JS              |
| Model     | `all-MiniLM-L6-v2` from `sentence-transformers` |
| Deployment| Render.com (API), GitHub Pages (Frontend) |

---

## Dataset

- Synthetic dataset mimicking **Indian real estate**
- Fields: `title`, `description`, `monthly_rent`, `bedrooms`, `amenities`, etc.
- Stored as `.csv` and embedded using Sentence-BERT
- Indexed using FAISS (saved as `spacematch_index.faiss`)

---

## 🌐 Live Demo

> 🔹 **Frontend**:  
> https://jatinchandani.github.io/space-match-ui

> 🔹 **API (Render)**:  
> https://space-match-api.onrender.com/chat?query=studio%20in%20Pune

Try queries like:
- `2 BHK in Mumbai under ₹30000`
- `furnished flat with parking in Delhi`
- `studio in Pune with gym`
- `luxury apartment in Bangalore`

---

## How to Run Locally

# 1. Clone repo
git clone https://github.com/Jatinchandani/space-match-api.git
cd space-match-api

# 2. Set up environment
python -m venv env
source env/bin/activate      # or `.\env\Scripts\activate` on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run backend
uvicorn spacematch_api:app --reload

---

## AI + Vector Logic

* Uses `all-MiniLM-L6-v2` for sentence embeddings
* User query is **enhanced semantically** (e.g. "affordable" → "low cost, budget")
* Filter layer applies post-embedding (e.g. bedrooms, amenities)
* Returns top N matches + summary message

---

## Folder Structure

```
space-match-api/
│
├── spacematch_api.py            # FastAPI app with /chat and vector search logic
├── vector_store_loader.py       # Vector store: load, embed, index
├── spacematch_properties.csv    # Synthetic dataset
├── spacematch_index.faiss       # Saved FAISS index (auto-generated)
├── requirements.txt
└── README.md
```

---

## Loom Video

🎥 [Click to Watch – 3 min Prototype Walkthrough](#)

---

## Credits

Built with ❤️ by **Jatin Chandani**
As part of the **SpaceMatch Founding Engineer Challenge**.

---

## 📩 Contact

* GitHub: [github.com/Jatinchandani](https://github.com/Jatinchandani)
* Email: [jatinchandani18@gmail.com](mailto:jatinchandani18@gmail.com)
