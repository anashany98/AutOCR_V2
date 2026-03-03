import hashlib
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except (ImportError, OSError):  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)


class RAGManager:
    """Manages semantic and hybrid retrieval for chat context."""

    def __init__(self, index_dir: str, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "text_index.faiss"
        self.meta_path = self.index_dir / "text_meta.pkl"
        self.model_name = model_name
        self.model = None
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.db_manager = None
        self.ensure_loaded()

    def set_db_manager(self, db_manager):
        self.db_manager = db_manager

    def ensure_loaded(self):
        """Lazy load model and vector index."""
        if not SentenceTransformer or not faiss:
            logger.warning("RAG requires sentence-transformers and faiss.")
            return

        if self.model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self.model = SentenceTransformer(self.model_name, device="cuda:0")

        if self.index is None:
            if self.index_path.exists() and self.meta_path.exists():
                logger.info("Loading existing RAG index...")
                try:
                    self.index = faiss.read_index(str(self.index_path))
                    with open(self.meta_path, "rb") as f:
                        self.metadata = pickle.load(f)
                except Exception as exc:
                    logger.error("Failed to load RAG index: %s. Recreating.", exc)
                    self._create_new_index()
            else:
                self._create_new_index()

    def _create_new_index(self):
        if self.model is None:
            return
        dim = self.model.get_sentence_embedding_dimension()
        quantizer = faiss.IndexFlatL2(dim)
        nlist = 100  # ajustar a sqrt(num_documentos) cuando el índice crezca
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        self.index.nprobe = 10  # chunks candidatos por búsqueda, balance velocidad/precisión
        self.metadata = []

    def _train_index_if_needed(self, embeddings: np.ndarray):
        """Entrena el índice IVFFlat si es necesario antes de añadir vectores."""
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if len(embeddings) >= 100:
                self.index.train(embeddings.astype("float32"))
                logger.info(f"Índice IVFFlat entrenado con {len(embeddings)} vectores")
            else:
                # fallback a IndexFlatL2 si hay menos de 100 documentos
                logger.warning(f"Menor de 100 documentos ({len(embeddings)}), usando IndexFlatL2")
                dim = self.model.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatL2(dim)
                self.metadata = []

    def save_index(self):
        """Persist FAISS index and metadata."""
        os.makedirs(self.index_dir, exist_ok=True)
        if self.index:
            faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    @staticmethod
    def _normalize_hotel_ids(
        hotel_id: Optional[int] = None,
        hotel_ids: Optional[Iterable[int]] = None,
    ) -> List[int]:
        out: List[int] = []
        for h in hotel_ids or []:
            try:
                if h is not None:
                    out.append(int(h))
            except Exception:
                continue
        if hotel_id is not None:
            try:
                out.append(int(hotel_id))
            except Exception:
                pass
        return sorted(set(out))

    @staticmethod
    def _split_text_chunks(text: str, max_chunk_chars: int = 1500) -> List[str]:
        text = str(text or "").strip()
        if not text:
            return []
        raw_chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 20]
        chunks: List[str] = []
        for rc in raw_chunks:
            if len(rc) <= max_chunk_chars:
                chunks.append(rc)
            else:
                for i in range(0, len(rc), max_chunk_chars):
                    piece = rc[i : i + max_chunk_chars].strip()
                    if piece:
                        chunks.append(piece)
        if not chunks:
            chunks = [text[:max_chunk_chars]]
        return chunks

    @staticmethod
    def _clean_htmlish_snippet(value: str) -> str:
        text = re.sub(r"</?b>", "", str(value or ""), flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _safe_row_field(row: Any, index: int, key: str):
        if isinstance(row, (tuple, list)):
            return row[index]
        try:
            return row[key]
        except Exception:
            return row[index]

    @staticmethod
    def _fingerprint_metadata(
        *,
        filename: str,
        owner_id: Optional[int],
        hotel_id: Optional[int],
        md5_hash: Optional[str],
        text_len: int,
    ) -> str:
        payload = f"{filename}|{owner_id}|{hotel_id}|{md5_hash or ''}|{int(text_len)}"
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _fingerprint_content(text: str) -> str:
        return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()

    def add_document(
        self,
        doc_id: int,
        filename: str,
        text: str,
        db_manager=None,
        owner_id: Optional[int] = None,
        hotel_id: Optional[int] = None,
    ):
        """Chunk, embed and index one document."""
        self.ensure_loaded()
        if not self.model:
            return

        db = db_manager or self.db_manager
        chunks = self._split_text_chunks(text)
        if not chunks:
            return

        embeddings = self.model.encode(chunks)

        use_pgvector = False
        if db:
            pg_conf = (db.config or {}).get("postgresql", {})
            use_pgvector = bool(pg_conf.get("use_pgvector", False))

        if db and db.engine_type == "postgresql" and use_pgvector:
            with db.get_connection() as conn:
                cursor = db.get_cursor(conn)
                try:
                    cursor.execute(
                        f"DELETE FROM document_embeddings WHERE doc_id = {db.placeholder}",
                        (int(doc_id),),
                    )
                except Exception:
                    conn.rollback()
                    raise
                for i, chunk in enumerate(chunks):
                    emb = list(embeddings[i].astype("float32"))
                    cursor.execute(
                        "INSERT INTO document_embeddings (doc_id, embedding, chunk_text) VALUES (%s, %s, %s)",
                        (int(doc_id), emb, chunk),
                    )
                conn.commit()
            return

        if not self.index:
            return

        # Entrenar índice IVFFlat si es necesario
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if len(embeddings) >= 100:
                self.index.train(np.array(embeddings).astype("float32"))
                logger.info(f"Índice IVFFlat entrenado con {len(embeddings)} vectores")
            else:
                # fallback a IndexFlatL2 si hay menos de 100 documentos
                logger.warning(f"Menor de 100 documentos ({len(embeddings)}), usando IndexFlatL2")
                dim = self.model.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatL2(dim)
                self.metadata = []

        self.index.add(np.array(embeddings).astype("float32"))
        for chunk in chunks:
            self.metadata.append(
                {
                    "doc_id": int(doc_id),
                    "filename": filename,
                    "owner_id": owner_id,
                    "hotel_id": hotel_id,
                    "text": chunk,
                }
            )

    def _vector_search(
        self,
        query: str,
        *,
        k: int,
        db_manager,
        owner_id: Optional[int],
        hotel_ids: List[int],
    ) -> List[Dict[str, Any]]:
        db = db_manager or self.db_manager
        if not self.model:
            return []

        use_pgvector = False
        if db:
            pg_conf = (db.config or {}).get("postgresql", {})
            use_pgvector = bool(pg_conf.get("use_pgvector", False))

        if db and db.engine_type == "postgresql" and use_pgvector:
            vec = list(self.model.encode([query])[0].astype("float32"))
            with db.get_connection() as conn:
                cursor = db.get_cursor(conn)
                where = []
                params: List[Any] = [vec]
                if owner_id is not None:
                    where.append("d.owner_id = %s")
                    params.append(int(owner_id))
                if hotel_ids:
                    placeholders = ",".join(["%s"] * len(hotel_ids))
                    where.append(f"d.hotel_id IN ({placeholders})")
                    params.extend(hotel_ids)
                where_sql = f"WHERE {' AND '.join(where)}" if where else ""
                params.append(max(1, int(k)))
                sql = f"""
                    SELECT e.doc_id, d.filename, e.chunk_text as text, e.embedding <-> %s as distance
                    FROM document_embeddings e
                    JOIN documents d ON e.doc_id = d.id
                    {where_sql}
                    ORDER BY distance
                    LIMIT %s
                """
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall() or []
            out = []
            for row in rows:
                doc_id = self._safe_row_field(row, 0, "doc_id")
                filename = self._safe_row_field(row, 1, "filename")
                text = self._safe_row_field(row, 2, "text")
                distance = float(self._safe_row_field(row, 3, "distance") or 0.0)
                score = 1.0 / (1.0 + max(0.0, distance))
                out.append(
                    {
                        "doc_id": int(doc_id),
                        "filename": filename,
                        "text": text,
                        "score": score,
                        "_vector_score": score,
                    }
                )
            return out

        if not self.index or self.index.ntotal == 0:
            return []
        vec = self.model.encode([query])
        search_k = max(1, int(k * 5))
        D, I = self.index.search(np.array(vec).astype("float32"), search_k)
        out = []
        hotel_set = set(hotel_ids)
        for i, idx in enumerate(I[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx].copy()
            if owner_id is not None:
                doc_owner = item.get("owner_id")
                if doc_owner is None or int(doc_owner) != int(owner_id):
                    continue
            if hotel_set:
                doc_hotel = item.get("hotel_id")
                if doc_hotel is None or int(doc_hotel) not in hotel_set:
                    continue
            distance = float(D[0][i])
            score = 1.0 / (1.0 + max(0.0, distance))
            item["score"] = score
            item["_vector_score"] = score
            out.append(item)
            if len(out) >= k:
                break
        return out

    def _keyword_search(
        self,
        query: str,
        *,
        k: int,
        db_manager,
        owner_id: Optional[int],
        hotel_ids: List[int],
    ) -> List[Dict[str, Any]]:
        db = db_manager or self.db_manager
        if not db:
            return []
        try:
            rows = db.search_documents(query, limit=max(1, int(k)), owner_id=owner_id, hotel_ids=hotel_ids)
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        for rank, row in enumerate(rows):
            doc_id = self._safe_row_field(row, 0, "doc_id")
            filename = self._safe_row_field(row, 1, "filename")
            snippet = self._safe_row_field(row, 2, "snippet")
            raw_rank = self._safe_row_field(row, 3, "rank") if (
                isinstance(row, (tuple, list)) and len(row) > 3
            ) else None
            try:
                raw_rank_f = float(raw_rank)
            except Exception:
                raw_rank_f = float(rank + 1)
            lexical_score = 1.0 / (1.0 + abs(raw_rank_f))
            out.append(
                {
                    "doc_id": int(doc_id),
                    "filename": filename,
                    "text": self._clean_htmlish_snippet(snippet),
                    "score": lexical_score,
                    "_keyword_score": lexical_score,
                }
            )
        return out

    @staticmethod
    def _rrf_merge(
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        *,
        weight_vector: float = 0.7,
        weight_keyword: float = 0.3,
    ) -> List[Dict[str, Any]]:
        by_key: Dict[str, Dict[str, Any]] = {}
        rank_v = {f"{r.get('doc_id')}::{r.get('text')}": i for i, r in enumerate(vector_results)}
        rank_k = {f"{r.get('doc_id')}::{r.get('text')}": i for i, r in enumerate(keyword_results)}

        def _ingest(rows: List[Dict[str, Any]]):
            for item in rows:
                key = f"{item.get('doc_id')}::{item.get('text')}"
                if key not in by_key:
                    by_key[key] = dict(item)
                else:
                    if len(str(item.get("text") or "")) > len(str(by_key[key].get("text") or "")):
                        by_key[key]["text"] = item.get("text")

        _ingest(vector_results)
        _ingest(keyword_results)

        merged: List[Dict[str, Any]] = []
        for key, item in by_key.items():
            rv = rank_v.get(key)
            rk = rank_k.get(key)
            score_v = (1.0 / (rv + 1.0)) if rv is not None else 0.0
            score_k = (1.0 / (rk + 1.0)) if rk is not None else 0.0
            combined = (weight_vector * score_v) + (weight_keyword * score_k)
            item["score"] = float(combined)
            item["_vector_rank"] = rv
            item["_keyword_rank"] = rk
            merged.append(item)
        merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return merged

    @staticmethod
    def _lexical_overlap_score(query: str, text: str) -> float:
        q_terms = {t for t in re.findall(r"[A-Za-z0-9_]+", (query or "").lower()) if len(t) > 1}
        if not q_terms:
            return 0.0
        t_terms = {t for t in re.findall(r"[A-Za-z0-9_]+", (text or "").lower()) if len(t) > 1}
        if not t_terms:
            return 0.0
        return float(len(q_terms.intersection(t_terms)) / max(1, len(q_terms)))

    def search(
        self,
        query: str,
        k: int = 5,
        db_manager=None,
        owner_id: Optional[int] = None,
        hotel_id: Optional[int] = None,
        hotel_ids: Optional[List[int]] = None,
        hybrid: bool = True,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks (hybrid by default) with optional reranking."""
        self.ensure_loaded()
        if not self.model:
            return []

        db = db_manager or self.db_manager
        scoped_hotel_ids = self._normalize_hotel_ids(hotel_id=hotel_id, hotel_ids=hotel_ids)
        top_k = max(1, int(k))
        candidate_k = max(top_k * 3, top_k)

        vector_results = self._vector_search(
            query,
            k=candidate_k,
            db_manager=db,
            owner_id=owner_id,
            hotel_ids=scoped_hotel_ids,
        )

        if hybrid and db is not None:
            keyword_results = self._keyword_search(
                query,
                k=candidate_k,
                db_manager=db,
                owner_id=owner_id,
                hotel_ids=scoped_hotel_ids,
            )
            merged = self._rrf_merge(vector_results, keyword_results)
        else:
            merged = list(vector_results)

        if rerank:
            for item in merged:
                overlap = self._lexical_overlap_score(query, str(item.get("text") or ""))
                base = float(item.get("score", 0.0))
                item["score"] = (0.85 * base) + (0.15 * overlap)
            merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        return merged[:top_k]

    def _fetch_documents(
        self,
        db_manager,
        *,
        doc_ids: Optional[Iterable[int]] = None,
        include_text: bool = False,
    ) -> List[Dict[str, Any]]:
        select_text = ", o.text" if include_text else ", LENGTH(COALESCE(o.text, '')) as text_len"
        sql = f"""
            SELECT d.id, d.filename, d.md5_hash, d.owner_id, d.hotel_id {select_text}
            FROM documents d
            JOIN ocr_texts o ON d.id = o.id_doc
            WHERE o.text IS NOT NULL AND LENGTH(COALESCE(o.text, '')) > 0
        """
        params: List[Any] = []
        ids = []
        for d in doc_ids or []:
            try:
                ids.append(int(d))
            except Exception:
                continue
        if ids:
            placeholders = ",".join([db_manager.placeholder] * len(ids))
            sql += f" AND d.id IN ({placeholders})"
            params.extend(ids)
        sql += " ORDER BY d.id ASC"

        with db_manager.get_connection() as conn:
            cursor = db_manager.get_cursor(conn)
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall() or []

        out: List[Dict[str, Any]] = []
        for row in rows:
            item = {
                "doc_id": int(self._safe_row_field(row, 0, "id")),
                "filename": str(self._safe_row_field(row, 1, "filename") or ""),
                "md5_hash": str(self._safe_row_field(row, 2, "md5_hash") or ""),
                "owner_id": self._safe_row_field(row, 3, "owner_id"),
                "hotel_id": self._safe_row_field(row, 4, "hotel_id"),
            }
            if include_text:
                item["text"] = str(self._safe_row_field(row, 5, "text") or "")
                item["text_len"] = len(item["text"])
            else:
                try:
                    item["text_len"] = int(self._safe_row_field(row, 5, "text_len") or 0)
                except Exception:
                    item["text_len"] = 0
            out.append(item)
        return out

    def rebuild(self, db_manager, incremental: bool = True):
        """
        Re-index documents from DB.

        Incremental mode behavior:
        - PostgreSQL + pgvector: updates changed/new docs in-place and removes deleted docs.
        - Local FAISS mode: appends new docs; if updates/deletes are detected, falls back to full rebuild.
        """
        self.ensure_loaded()
        if not self.model:
            return

        state_supported = all(
            hasattr(db_manager, name)
            for name in (
                "get_rag_index_state",
                "upsert_rag_index_state_entries",
                "delete_rag_index_state_not_in",
            )
        )
        if not state_supported:
            incremental = False

        meta_docs = self._fetch_documents(db_manager, include_text=False)
        current_ids = {int(d["doc_id"]) for d in meta_docs}

        old_state = db_manager.get_rag_index_state() if state_supported else {}
        stale_ids = sorted(set(old_state.keys()) - current_ids)

        meta_hash_by_doc: Dict[int, str] = {}
        changed_ids: List[int] = []
        new_ids: List[int] = []
        for doc in meta_docs:
            doc_id = int(doc["doc_id"])
            meta_hash = self._fingerprint_metadata(
                filename=str(doc.get("filename") or ""),
                owner_id=doc.get("owner_id"),
                hotel_id=doc.get("hotel_id"),
                md5_hash=str(doc.get("md5_hash") or ""),
                text_len=int(doc.get("text_len") or 0),
            )
            meta_hash_by_doc[doc_id] = meta_hash
            prev = old_state.get(doc_id)
            if not prev:
                new_ids.append(doc_id)
                continue
            if str(prev.get("metadata_hash") or "") != meta_hash:
                changed_ids.append(doc_id)

        pg_conf = (db_manager.config or {}).get("postgresql", {})
        use_pgvector = bool(pg_conf.get("use_pgvector", False))
        pgvector_mode = bool(db_manager.engine_type == "postgresql" and use_pgvector)

        logger.info(
            "RAG rebuild planning: total=%d new=%d changed=%d stale=%d incremental=%s pgvector=%s",
            len(meta_docs),
            len(new_ids),
            len(changed_ids),
            len(stale_ids),
            bool(incremental),
            bool(pgvector_mode),
        )

        if pgvector_mode:
            self._rebuild_pgvector(
                db_manager=db_manager,
                all_meta=meta_docs,
                new_ids=new_ids,
                changed_ids=changed_ids,
                stale_ids=stale_ids,
                meta_hash_by_doc=meta_hash_by_doc,
                incremental=bool(incremental),
            )
            return

        self._rebuild_faiss(
            db_manager=db_manager,
            all_meta=meta_docs,
            new_ids=new_ids,
            changed_ids=changed_ids,
            stale_ids=stale_ids,
            meta_hash_by_doc=meta_hash_by_doc,
            incremental=bool(incremental),
        )

    def _rebuild_pgvector(
        self,
        *,
        db_manager,
        all_meta: List[Dict[str, Any]],
        new_ids: List[int],
        changed_ids: List[int],
        stale_ids: List[int],
        meta_hash_by_doc: Dict[int, str],
        incremental: bool,
    ):
        ids_to_index = sorted({int(d["doc_id"]) for d in all_meta}) if not incremental else sorted(set(new_ids + changed_ids))

        with db_manager.get_connection() as conn:
            cursor = db_manager.get_cursor(conn)
            if not incremental:
                cursor.execute("DELETE FROM document_embeddings")
            elif stale_ids:
                placeholders = ",".join([db_manager.placeholder] * len(stale_ids))
                cursor.execute(
                    f"DELETE FROM document_embeddings WHERE doc_id IN ({placeholders})",
                    tuple(stale_ids),
                )
            conn.commit()

        if ids_to_index:
            docs = self._fetch_documents(db_manager, doc_ids=ids_to_index, include_text=True)
            for doc in docs:
                self.add_document(
                    doc_id=int(doc["doc_id"]),
                    filename=str(doc.get("filename") or ""),
                    text=str(doc.get("text") or ""),
                    db_manager=db_manager,
                    owner_id=doc.get("owner_id"),
                    hotel_id=doc.get("hotel_id"),
                )

            state_entries = []
            for doc in docs:
                doc_id = int(doc["doc_id"])
                state_entries.append(
                    {
                        "doc_id": doc_id,
                        "content_hash": self._fingerprint_content(str(doc.get("text") or "")),
                        "metadata_hash": meta_hash_by_doc.get(doc_id, ""),
                    }
                )
            db_manager.upsert_rag_index_state_entries(state_entries)

        db_manager.delete_rag_index_state_not_in([int(d["doc_id"]) for d in all_meta])
        logger.info("RAG pgvector rebuild complete. indexed=%d", len(ids_to_index))

    def _rebuild_faiss(
        self,
        *,
        db_manager,
        all_meta: List[Dict[str, Any]],
        new_ids: List[int],
        changed_ids: List[int],
        stale_ids: List[int],
        meta_hash_by_doc: Dict[int, str],
        incremental: bool,
    ):
        needs_full = (not incremental) or bool(changed_ids) or bool(stale_ids)
        if needs_full:
            self._create_new_index()
            docs = self._fetch_documents(db_manager, include_text=True)
            logger.info("Rebuilding full FAISS index for %d docs", len(docs))
            for doc in docs:
                self.add_document(
                    doc_id=int(doc["doc_id"]),
                    filename=str(doc.get("filename") or ""),
                    text=str(doc.get("text") or ""),
                    db_manager=db_manager,
                    owner_id=doc.get("owner_id"),
                    hotel_id=doc.get("hotel_id"),
                )
            self.save_index()

            state_entries = []
            for doc in docs:
                doc_id = int(doc["doc_id"])
                state_entries.append(
                    {
                        "doc_id": doc_id,
                        "content_hash": self._fingerprint_content(str(doc.get("text") or "")),
                        "metadata_hash": meta_hash_by_doc.get(doc_id, ""),
                    }
                )
            db_manager.upsert_rag_index_state_entries(state_entries)
            db_manager.delete_rag_index_state_not_in([int(d["doc_id"]) for d in all_meta])
            logger.info("RAG full FAISS rebuild complete.")
            return

        if not new_ids:
            logger.info("RAG incremental FAISS rebuild: no new documents.")
            return

        docs = self._fetch_documents(db_manager, doc_ids=new_ids, include_text=True)
        logger.info("RAG incremental FAISS rebuild: indexing %d new docs", len(docs))
        for doc in docs:
            self.add_document(
                doc_id=int(doc["doc_id"]),
                filename=str(doc.get("filename") or ""),
                text=str(doc.get("text") or ""),
                db_manager=db_manager,
                owner_id=doc.get("owner_id"),
                hotel_id=doc.get("hotel_id"),
            )
        self.save_index()

        state_entries = []
        for doc in docs:
            doc_id = int(doc["doc_id"])
            state_entries.append(
                {
                    "doc_id": doc_id,
                    "content_hash": self._fingerprint_content(str(doc.get("text") or "")),
                    "metadata_hash": meta_hash_by_doc.get(doc_id, ""),
                }
            )
        db_manager.upsert_rag_index_state_entries(state_entries)
        logger.info("RAG incremental FAISS append complete.")
