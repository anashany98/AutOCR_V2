"""
Sistema Anti-Duplicados de Documentos.

Implementa un sistema de 3 capas para detectar documentos duplicados:
1. Message-ID (para emails)
2. SHA256 hash (para contenido exacto)
3. Similitud semántica con pgvector (>95%)
"""
import os
import logging
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DeduplicationManager:
    """
    Gestor de anti-duplicados de documentos.
    
    Tres capas de detección:
    1. Message-ID: Para correos electrónicos
    2. SHA256: Para contenido binario exacto
    3. pgvector: Para similitud semántica >95%
    """
    
    def __init__(self):
        self._db = None
    
    def _get_db(self):
        """Obtiene la instancia de DBManager."""
        if self._db is None:
            from modules.db_manager import DBManager
            self._db = DBManager.get_instance()
        return self._db
    
    # -------------------------------------------------------------------------
    # Capa 1: Message-ID (para emails)
    # -------------------------------------------------------------------------
    
    def check_message_id(self, message_id: str) -> Optional[str]:
        """
        Verifica si un Message-ID ya existe.
        
        Args:
            message_id: ID del mensaje de email
            
        Returns:
            ID del documento duplicado si existe, None si no
        """
        db = self._get_db()
        
        query = """
            SELECT id FROM email_message_ids 
            WHERE message_id = %s
            LIMIT 1
        """
        
        try:
            result = db.fetch_one(query, (message_id,))
            return str(result[0]) if result else None
        except Exception as e:
            logger.error(f"Error checking Message-ID: {e}")
            return None
    
    def register_message_id(self, message_id: str, document_id: str) -> bool:
        """
        Registra un Message-ID para evitar duplicados.
        
        Args:
            message_id: ID del mensaje
            document_id: ID del documento asociado
            
        Returns:
            True si se registró correctamente
        """
        db = self._get_db()
        
        query = """
            INSERT INTO email_message_ids (message_id, document_id, created_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (message_id) DO NOTHING
        """
        
        try:
            db.execute(query, (message_id, document_id))
            return True
        except Exception as e:
            logger.error(f"Error registering Message-ID: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Capa 2: SHA256 Hash
    # -------------------------------------------------------------------------
    
    def calculate_sha256(self, file_path: str) -> Optional[str]:
        """
        Calcula el hash SHA256 de un archivo.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            Hash SHA256 en hexadecimal o None si hay error
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating SHA256: {e}")
            return None
    
    def check_sha256(self, sha256_hash: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """
        Verifica si un hash SHA256 ya existe para el tenant.
        
        Args:
            sha256_hash: Hash SHA256
            tenant_id: ID del tenant
            
        Returns:
            Datos del documento duplicado si existe
        """
        db = self._get_db()
        
        query = """
            SELECT id, filename, file_hash, created_at
            FROM documents 
            WHERE file_hash = %s AND tenant_id = %s
            LIMIT 1
        """
        
        try:
            result = db.fetch_one(query, (sha256_hash, tenant_id))
            if result:
                return {
                    "document_id": str(result[0]),
                    "filename": result[1],
                    "file_hash": result[2],
                    "created_at": result[3].isoformat() if result[3] else None
                }
        except Exception as e:
            logger.error(f"Error checking SHA256: {e}")
        return None
    
    # -------------------------------------------------------------------------
    # Capa 3: Similitud Semántica con pgvector
    # -------------------------------------------------------------------------
    
    def check_semantic_similarity(self, text: str, tenant_id: str, 
                                  threshold: float = 0.95) -> List[Dict[str, Any]]:
        """
        Verifica similitud semántica usando pgvector.
        
        Args:
            text: Texto a comparar
            tenant_id: ID del tenant
            threshold: Umbral de similitud (0-1), default 0.95
            
        Returns:
            Lista de documentos similares
        """
        db = self._get_db()
        
        # First check if pgvector is available
        try:
            # Get the embedding for the text
            from modules.embedding_step import get_embeddings
            
            # Get embedding for the new text
            embedding = get_embeddings([text])
            if not embedding or len(embedding) == 0:
                return []
            
            embedding_vec = embedding[0]
            
            # Format for pgvector query
            embedding_str = "[" + ",".join(str(x) for x in embedding_vec) + "]"
            
            query = f"""
                SELECT id, filename, extracted_text, 
                       (embedding <=> '{embedding_str}') as similarity
                FROM documents 
                WHERE tenant_id = %s 
                    AND embedding IS NOT NULL
                    AND (embedding <=> '{embedding_str}') > {threshold}
                ORDER BY similarity DESC
                LIMIT 5
            """
            
            results = db.fetch_all(query, (tenant_id,))
            
            similar_docs = []
            for row in results:
                similar_docs.append({
                    "document_id": str(row[0]),
                    "filename": row[1],
                    "extracted_text": row[2][:200] + "..." if row[2] and len(row[2]) > 200 else row[2],
                    "similarity": float(row[3])
                })
            
            return similar_docs
            
        except Exception as e:
            logger.warning(f"pgvector similarity check failed: {e}")
            return []
    
    # -------------------------------------------------------------------------
    # Sistema completo de verificación
    # -------------------------------------------------------------------------
    
    def check_duplicates(self, file_path: str = None, 
                       message_id: str = None,
                       text: str = None,
                       tenant_id: str = None) -> Dict[str, Any]:
        """
        Sistema completo de verificación de duplicados.
        
        Verifica las 3 capas en orden:
        1. Message-ID
        2. SHA256
        3. Similitud semántica
        
        Args:
            file_path: Ruta al archivo (para SHA256)
            message_id: ID del mensaje (para emails)
            text: Texto para similitud semántica
            tenant_id: ID del tenant
            
        Returns:
            Diccionario con resultados de cada capa
        """
        result = {
            "is_duplicate": False,
            "message_id_match": None,
            "sha256_match": None,
            "semantic_matches": [],
            "primary_match": None
        }
        
        # Capa 1: Message-ID
        if message_id:
            msg_match = self.check_message_id(message_id)
            if msg_match:
                result["is_duplicate"] = True
                result["message_id_match"] = msg_match
                result["primary_match"] = {
                    "type": "message_id",
                    "document_id": msg_match
                }
                return result
        
        # Capa 2: SHA256
        if file_path and tenant_id:
            sha256_hash = self.calculate_sha256(file_path)
            if sha256_hash:
                sha_match = self.check_sha256(sha256_hash, tenant_id)
                if sha_match:
                    result["is_duplicate"] = True
                    result["sha256_match"] = sha_match
                    result["primary_match"] = {
                        "type": "sha256",
                        "document_id": sha_match["document_id"]
                    }
                    return result
        
        # Capa 3: Similitud semántica
        if text and tenant_id:
            semantic_matches = self.check_semantic_similarity(text, tenant_id)
            if semantic_matches:
                result["is_duplicate"] = True
                result["semantic_matches"] = semantic_matches
                result["primary_match"] = {
                    "type": "semantic",
                    "document_id": semantic_matches[0]["document_id"],
                    "similarity": semantic_matches[0]["similarity"]
                }
        
        return result
    
    def register_document(self, document_id: str, file_path: str = None,
                        message_id: str = None, text: str = None) -> bool:
        """
        Registra un documento en el sistema de deduplicación.
        
        Args:
            document_id: ID del documento
            file_path: Ruta al archivo (para SHA256)
            message_id: ID del mensaje (para emails)
            text: Texto para embedding
            
        Returns:
            True si se registró correctamente
        """
        db = self._get_db()
        
        try:
            # Register SHA256
            if file_path:
                sha256_hash = self.calculate_sha256(file_path)
                if sha256_hash:
                    query = """
                        UPDATE documents SET file_hash = %s, updated_at = NOW()
                        WHERE id = %s
                    """
                    db.execute(query, (sha256_hash, document_id))
            
            # Register Message-ID
            if message_id:
                self.register_message_id(message_id, document_id)
            
            # Register for semantic similarity (embedding)
            if text:
                try:
                    from modules.embedding_step import get_embeddings
                    
                    embedding = get_embeddings([text])
                    if embedding and len(embedding) > 0:
                        embedding_str = "[" + ",".join(str(x) for x in embedding[0]) + "]"
                        
                        query = f"""
                            UPDATE documents SET embedding = '{embedding_str}', updated_at = NOW()
                            WHERE id = %s
                        """
                        db.execute(query, (document_id,))
                except Exception as e:
                    logger.warning(f"Failed to create embedding for deduplication: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering document for deduplication: {e}")
            return False


# Singleton instance
_dedup_manager: Optional[DeduplicationManager] = None


def get_dedup_manager() -> DeduplicationManager:
    """Obtiene la instancia singleton del gestor de deduplicación."""
    global _dedup_manager
    if _dedup_manager is None:
        _dedup_manager = DeduplicationManager()
    return _dedup_manager


# Telegram notification for duplicates
async def notify_duplicate_detected(document_id: str, duplicate_info: Dict[str, Any],
                                    tenant_id: str):
    """
    Envía notificación por Telegram cuando se detecta un duplicado.
    
    Args:
        document_id: ID del nuevo documento
        duplicate_info: Información del duplicado detectado
        tenant_id: ID del tenant
    """
    from modules.telegram_gestores_db import get_telegram_gestores_db
    
    gestores_db = get_telegram_gestores_db()
    gestores = gestores_db.get_active_gestores(tenant_id)
    
    if not gestores:
        return
    
    # Build message
    match_type = duplicate_info.get("primary_match", {}).get("type", "unknown")
    
    message = f"⚠️ *DUPLICADO DETECTADO*\\n\\n"
    
    if match_type == "message_id":
        message += "Tipo: Email duplicado (Message-ID)\\n"
    elif match_type == "sha256":
        message += "Tipo: Archivo idéntico (SHA256)\\n"
    elif match_type == "semantic":
        similarity = duplicate_info.get("primary_match", {}).get("similarity", 0)
        message += f"Tipo: Similitud semántica ({similarity*100:.1f}%)\\n"
    
    message += f"\\nNuevo documento ID: `{document_id}`\\n"
    
    if duplicate_info.get("message_id_match"):
        message += f"Duplicado ID: `{duplicate_info['message_id_match']}`"
    elif duplicate_info.get("sha256_match"):
        message += f"Duplicado: {duplicate_info['sha256_match'].get('filename', 'N/A')}"
    elif duplicate_info.get("semantic_matches"):
        match = duplicate_info["semantic_matches"][0]
        message += f"Similar: {match.get('filename', 'N/A')} ({match.get('similarity', 0)*100:.1f}%)"
    
    # Send to all gestores
    for gestor in gestores:
        if not gestor.notify_alerts:
            continue
        
        try:
            from telegram import Bot
            import os
            
            token = os.environ.get("TELEGRAM_BOT_TOKEN")
            if token:
                bot = Bot(token=token)
                await bot.send_message(
                    chat_id=gestor.telegram_id,
                    text=message,
                    parse_mode="Markdown"
                )
        except Exception as e:
            logger.error(f"Error sending duplicate notification: {e}")
