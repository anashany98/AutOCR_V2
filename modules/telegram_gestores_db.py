"""
Telegram Gestores Database Manager.

Gestiona la tabla telegram_gestores en PostgreSQL para autenticar
los 20 gestores через Telegram.
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TelegramGestor:
    """Representa un gestor vinculado a Telegram."""
    id: Optional[str] = None
    telegram_id: int = 0
    username: str = ""
    first_name: str = ""
    last_name: str = ""
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    hotel_id: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    verified_at: Optional[datetime] = None
    notify_invoices: bool = True
    notify_expiry: bool = True
    notify_alerts: bool = True
    language: str = "es"
    last_command: Optional[str] = None
    last_seen_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TelegramGestoresDB:
    """Gestor de usuarios Telegram en PostgreSQL."""
    
    def __init__(self, db=None):
        """
        Inicializa el gestor de Telegram.
        
        Args:
            db: Instancia de DBManager. Si es None, se obtaindrá automáticamente.
        """
        self._db = db
    
    def _get_db(self):
        """Obtiene la instancia de DBManager."""
        if self._db is None:
            from modules.db_manager import DBManager
            self._db = DBManager.get_instance()
        return self._db
    
    def get_by_telegram_id(self, telegram_id: int) -> Optional[TelegramGestor]:
        """
        Busca un gestor por telegram_id.
        
        Args:
            telegram_id: ID de Telegram del usuario
            
        Returns:
            TelegramGestor si existe, None si no se encuentra
        """
        db = self._get_db()
        
        query = """
            SELECT id, telegram_id, username, first_name, last_name,
                   user_id, tenant_id, hotel_id, is_active, is_verified,
                   verified_at, notify_invoices, notify_expiry, notify_alerts,
                   language, last_command, last_seen_at, created_at, updated_at
            FROM telegram_gestores
            WHERE telegram_id = %s
        """
        
        try:
            result = db.fetch_one(query, (telegram_id,))
            if result:
                return TelegramGestor(
                    id=str(result[0]),
                    telegram_id=result[1],
                    username=result[2] or "",
                    first_name=result[3] or "",
                    last_name=result[4] or "",
                    user_id=str(result[5]) if result[5] else None,
                    tenant_id=str(result[6]) if result[6] else None,
                    hotel_id=str(result[7]) if result[7] else None,
                    is_active=result[8],
                    is_verified=result[9],
                    verified_at=result[10],
                    notify_invoices=result[11],
                    notify_expiry=result[12],
                    notify_alerts=result[13],
                    language=result[14] or "es",
                    last_command=result[15],
                    last_seen_at=result[16],
                    created_at=result[17],
                    updated_at=result[18]
                )
        except Exception as e:
            logger.error(f"Error fetching telegram gestor: {e}")
        
        return None
    
    def create(self, telegram_id: int, username: str, first_name: str,
               last_name: str = "") -> Optional[TelegramGestor]:
        """
        Crea un nuevo gestor de Telegram.
        
        Args:
            telegram_id: ID de Telegram
            username: Username de Telegram
            first_name: Nombre del usuario
            last_name: Apellido del usuario
            
        Returns:
            TelegramGestor creado o None si hay error
        """
        db = self._get_db()
        
        query = """
            INSERT INTO telegram_gestores 
                (telegram_id, username, first_name, last_name, is_active, is_verified)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (telegram_id) DO UPDATE SET
                username = EXCLUDED.username,
                first_name = EXCLUDED.first_name,
                last_name = EXCLUDED.last_name,
                updated_at = NOW()
            RETURNING id, telegram_id, username, first_name, last_name,
                     user_id, tenant_id, hotel_id, is_active, is_verified,
                     verified_at, notify_invoices, notify_expiry, notify_alerts,
                     language, last_command, last_seen_at, created_at, updated_at
        """
        
        try:
            result = db.fetch_one(query, (
                telegram_id, username, first_name, last_name, True, False
            ))
            
            if result:
                return TelegramGestor(
                    id=str(result[0]),
                    telegram_id=result[1],
                    username=result[2] or "",
                    first_name=result[3] or "",
                    last_name=result[4] or "",
                    user_id=str(result[5]) if result[5] else None,
                    tenant_id=str(result[6]) if result[6] else None,
                    hotel_id=str(result[7]) if result[7] else None,
                    is_active=result[8],
                    is_verified=result[9],
                    verified_at=result[10],
                    notify_invoices=result[11],
                    notify_expiry=result[12],
                    notify_alerts=result[13],
                    language=result[14] or "es",
                    last_command=result[15],
                    last_seen_at=result[16],
                    created_at=result[17],
                    updated_at=result[18]
                )
        except Exception as e:
            logger.error(f"Error creating telegram gestor: {e}")
        
        return None
    
    def link_to_autoocr(self, telegram_id: int, user_id: str, 
                       tenant_id: str, hotel_id: str = None) -> bool:
        """
        Vincula un usuario de Telegram a un usuario de AutoOCR.
        
        Args:
            telegram_id: ID de Telegram
            user_id: ID del usuario en AutoOCR
            tenant_id: ID del tenant
            hotel_id: ID del hotel (opcional)
            
        Returns:
            True si se vinculó correctamente
        """
        db = self._get_db()
        
        query = """
            UPDATE telegram_gestores
            SET user_id = %s,
                tenant_id = %s,
                hotel_id = %s,
                is_verified = TRUE,
                verified_at = NOW(),
                updated_at = NOW()
            WHERE telegram_id = %s
        """
        
        try:
            db.execute(query, (user_id, tenant_id, hotel_id, telegram_id))
            return True
        except Exception as e:
            logger.error(f"Error linking telegram gestor: {e}")
            return False
    
    def update_last_seen(self, telegram_id: int, command: str = None) -> bool:
        """
        Actualiza la última vez que se vio al usuario.
        
        Args:
            telegram_id: ID de Telegram
            command: Último comando ejecutado
            
        Returns:
            True si se actualizó correctamente
        """
        db = self._get_db()
        
        query = """
            UPDATE telegram_gestores
            SET last_seen_at = NOW(),
                last_command = %s,
                updated_at = NOW()
            WHERE telegram_id = %s
        """
        
        try:
            db.execute(query, (command, telegram_id))
            return True
        except Exception as e:
            logger.error(f"Error updating last_seen: {e}")
            return False
    
    def get_active_gestores(self, tenant_id: str = None) -> List[TelegramGestor]:
        """
        Obtiene todos los gestores activos.
        
        Args:
            tenant_id: Filtrar por tenant (opcional)
            
        Returns:
            Lista de gestores activos
        """
        db = self._get_db()
        
        if tenant_id:
            query = """
                SELECT id, telegram_id, username, first_name, last_name,
                       user_id, tenant_id, hotel_id, is_active, is_verified,
                       verified_at, notify_invoices, notify_expiry, notify_alerts,
                       language, last_command, last_seen_at, created_at, updated_at
                FROM telegram_gestores
                WHERE is_active = TRUE AND tenant_id = %s
                ORDER BY first_name
            """
            results = db.fetch_all(query, (tenant_id,))
        else:
            query = """
                SELECT id, telegram_id, username, first_name, last_name,
                       user_id, tenant_id, hotel_id, is_active, is_verified,
                       verified_at, notify_invoices, notify_expiry, notify_alerts,
                       language, last_command, last_seen_at, created_at, updated_at
                FROM telegram_gestores
                WHERE is_active = TRUE
                ORDER BY first_name
            """
            results = db.fetch_all(query)
        
        gestores = []
        for result in results:
            gestores.append(TelegramGestor(
                id=str(result[0]),
                telegram_id=result[1],
                username=result[2] or "",
                first_name=result[3] or "",
                last_name=result[4] or "",
                user_id=str(result[5]) if result[5] else None,
                tenant_id=str(result[6]) if result[6] else None,
                hotel_id=str(result[7]) if result[7] else None,
                is_active=result[8],
                is_verified=result[9],
                verified_at=result[10],
                notify_invoices=result[11],
                notify_expiry=result[12],
                notify_alerts=result[13],
                language=result[14] or "es",
                last_command=result[15],
                last_seen_at=result[16],
                created_at=result[17],
                updated_at=result[18]
            ))
        
        return gestores
    
    def log_command(self, telegram_id: int, command: str, 
                   args: str = None, result: str = None,
                   success: bool = True, tenant_id: str = None,
                   user_id: str = None) -> bool:
        """
        Registra un comando ejecutado por el usuario.
        
        Args:
            telegram_id: ID de Telegram
            command: Comando ejecutado
            args: Argumentos del comando
            result: Resultado del comando
            success: Si el comando fue exitoso
            tenant_id: ID del tenant
            user_id: ID del usuario en AutoOCR
            
        Returns:
            True si se registró correctamente
        """
        db = self._get_db()
        
        query = """
            INSERT INTO telegram_command_log 
                (telegram_id, command, args, result, success, tenant_id, user_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        try:
            db.execute(query, (telegram_id, command, args, result, success, tenant_id, user_id))
            return True
        except Exception as e:
            logger.error(f"Error logging command: {e}")
            return False


# Singleton instance
_gestores_db: Optional[TelegramGestoresDB] = None


def get_telegram_gestores_db() -> TelegramGestoresDB:
    """Obtiene la instancia singleton de TelegramGestoresDB."""
    global _gestores_db
    if _gestores_db is None:
        _gestores_db = TelegramGestoresDB()
    return _gestores_db
