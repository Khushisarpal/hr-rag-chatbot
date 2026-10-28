# Chatbot RH (RAG)

Assistant conversationnel qui répond aux questions RH à partir de documents internes (congés, télétravail, notes de frais, formation).

**Stack** : Python, FastAPI, LangChain (splitters, VectorStore, Embeddings), base vectorielle, interface web.

## Fonctionnement
1. Les documents `data/policies/*.md` sont découpés par titres puis par blocs (LangChain).
2. Chaque bloc est vectorisé et stocké dans une base vectorielle (`InMemoryVectorStore`).
3. `POST /chat` récupère les blocs les plus proches, puis génère la réponse avec les sources.
4. Sous le seuil de similarité, le bot refuse de répondre (pas d'hallucination).
5. Avec `ANTHROPIC_API_KEY`, la réponse est rédigée par un LLM. Sans clé, réponse extractive.

## Lancer
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload     # http://localhost:8000
pytest                            # tests de validation
```

## API
- `GET /health`
- `POST /chat` avec `{"question": "Combien de jours de télétravail ?"}`
  -> `{"answer": "...", "sources": [{"file", "section", "score"}], "grounded": true}`

## Déploiement
Dockerfile + `render.yaml` fournis (Render, plan gratuit). Ajouter ses propres documents dans `data/policies/`.
