"""
Outlook Email Importer using Microsoft Graph API.

Importa correos de Outlook/Exchange usando Microsoft Graph API,
soporta buzones centralizados y buzones de gestores individuales.
"""
import os
import time
import logging
import hashlib
import threading
import re
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OutlookEmail:
    """Representa un correo de Outlook."""
    message_id: str
    subject: str
    sender: str
    sender_email: str
    recipients: List[str]
    received_at: datetime
    has_attachments: bool
    attachments: List[Dict[str, Any]]
    body_preview: str


class OutlookGraphImporter:
    """
    Importador de correo de Outlook usando Microsoft Graph API.
    
    Características:
    - Autenticación con Azure AD (OAuth2)
    - Soporte para buzón centralizado (facturas@empresa.com)
    - Soporte para buzones de gestores individuales
    - Deduplicación por Message-ID
    - Movimiento a carpeta "AutOCR Procesados"
    """
    
    def __init__(self, config: dict, input_folder: str):
        """
        Inicializa el importador de Outlook.
        
        Args:
            config: Configuración con tenant_id, client_id, client_secret
            input_folder: Carpeta donde guardar los archivos descargados
        """
        self.config = config
        self.input_folder = input_folder
        self.logger = logging.getLogger("OutlookGraphImporter")
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Azure AD configuration
        self.tenant_id = config.get("tenant_id")
        self.client_id = config.get("client_id")
        self.client_secret = config.get("client_secret")
        self.scopes = config.get("scopes", ["https://graph.microsoft.com/.default"])
        
        # Email configuration
        self.central_mailbox = config.get("central_mailbox")  # facturas@empresa.com
        self.gestor_mailboxes = config.get("gestor_mailboxes", [])  # Lista de correos de gestores
        self.processed_folder = config.get("processed_folder", "AutOCR Procesados")
        self.allowed_extensions = set(config.get("allowed_extensions", [".pdf", ".jpg", ".jpeg", ".png", ".tiff"]))
        
        # Token de acceso
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        
        # Asegurar que existe la carpeta de entrada
        os.makedirs(input_folder, exist_ok=True)
    
    def start(self):
        """Inicia el importador de correo."""
        if self.running:
            return
        
        if not self.config.get("enabled", False):
            self.logger.info("Outlook Importer está deshabilitado en configuración.")
            return
        
        if not self.tenant_id or not self.client_id or not self.client_secret:
            self.logger.warning("Falta configuración de Azure AD (tenant_id, client_id, client_secret).")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.logger.info("Outlook Graph Importer iniciado.")
    
    def stop(self):
        """Detiene el importador de correo."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self.logger.info("Outlook Graph Importer detenido.")
    
    def _run_loop(self):
        """Bucle principal de verificación de correo."""
        check_interval = self.config.get("check_interval", 900)  # 15 minutos por defecto
        while self.running:
            try:
                self._check_all_mailboxes()
            except Exception as e:
                self.logger.error(f"Error en bucle de verificación: {e}")
            
            # Sleep en intervalos pequeños para permitir parada rápida
            for _ in range(check_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def check_now(self):
        """Trigger manual para verificar correos inmediatamente."""
        self.logger.info("Verificación manual de correos iniciada.")
        self._check_all_mailboxes()
    
    def _check_all_mailboxes(self):
        """Verifica todos los buzones configurados."""
        # Primero, verificar buzón centralizado
        if self.central_mailbox:
            self.logger.info(f"Verificando buzón centralizado: {self.central_mailbox}")
            self._check_mailbox(self.central_mailbox)
        
        # Luego, verificar buzones de gestores
        for mailbox in self.gestor_mailboxes:
            self.logger.info(f"Verificando buzón de gestor: {mailbox}")
            self._check_mailbox(mailbox)
    
    def _check_mailbox(self, mailbox: str):
        """Verifica un buzón específico."""
        try:
            # Obtener token de acceso
            if not self._ensure_valid_token():
                self.logger.error("No se pudo obtener token de acceso")
                return
            
            # Buscar correos no leídos
            emails = self._get_unread_emails(mailbox)
            
            for email in emails:
                self._process_email(email, mailbox)
                
        except Exception as e:
            self.logger.error(f"Error al verificar buzón {mailbox}: {e}")
    
    def _ensure_valid_token(self) -> bool:
        """
        Obtiene un token de acceso válido.
        
        Returns:
            True si se obtuvo un token válido
        """
        if self._access_token and self._token_expires_at:
            if datetime.now() < self._token_expires_at - timedelta(minutes=5):
                return True
        
        # Obtener nuevo token
        try:
            import msal
        except ImportError:
            self.logger.error("Se requiere msal: pip install msal")
            return False
        
        authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        
        # Crear aplicación cliente confidencial
        app = msal.ConfidentialClientApplication(
            client_id=self.client_id,
            authority=authority,
            client_credential=self.client_secret
        )
        
        # Obtener token
        result = app.acquire_token_for_client(scopes=self.scopes)
        
        if "access_token" in result:
            self._access_token = result["access_token"]
            expires_in = result.get("expires_in", 3600)
            self._token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            self.logger.debug("Token de acceso obtenido correctamente")
            return True
        else:
            error = result.get("error", "Unknown")
            error_description = result.get("error_description", "")
            self.logger.error(f"Error al obtener token: {error} - {error_description}")
            return False
    
    def _get_unread_emails(self, mailbox: str) -> List[OutlookEmail]:
        """Obtiene correos no leídos del buzón."""
        import requests
        
        # Usar la API de Microsoft Graph
        # Para buzones compartidos, necesitamos usar /users/{mailbox}
        user_id = mailbox
        
        url = f"https://graph.microsoft.com/v1.0/users/{user_id}/messages"
        
        params = {
            "$filter": "isRead eq false",
            "$top": 50,
            "$select": "id,subject,from,toRecipients,ccRecipients,receivedDateTime,hasAttachments,bodyPreview",
            "$orderby": "receivedDateTime desc"
        }
        
        headers = {
            "Authorization": f"Bearer {self._access_token}"
        }
        
        emails = []
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                for msg in data.get("value", []):
                    email = OutlookEmail(
                        message_id=msg.get("id", ""),
                        subject=msg.get("subject", "(Sin asunto)"),
                        sender=msg.get("from", {}).get("emailAddress", {}).get("name", ""),
                        sender_email=msg.get("from", {}).get("emailAddress", {}).get("address", ""),
                        recipients=[r.get("emailAddress", {}).get("address", "") 
                                  for r in msg.get("toRecipients", [])],
                        received_at=datetime.fromisoformat(
                            msg.get("receivedDateTime", "").replace("Z", "+00:00")
                        ),
                        has_attachments=msg.get("hasAttachments", False),
                        attachments=[],
                        body_preview=msg.get("bodyPreview", "")[:200]
                    )
                    emails.append(email)
                    
            elif response.status_code == 403:
                self.logger.error(f"Permisos insuficientes para acceder a {mailbox}")
            elif response.status_code == 404:
                self.logger.error(f"Buzón no encontrado: {mailbox}")
            else:
                self.logger.error(f"Error de Graph API: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error al obtener correos: {e}")
        
        return emails
    
    def _process_email(self, email: OutlookEmail, mailbox: str):
        """Procesa un correo individual."""
        self.logger.info(f"Procesando correo de {email.sender_email}: {email.subject}")
        
        # Verificar deduplicación por Message-ID
        if self._is_duplicate_message_id(email.message_id):
            self.logger.info(f"Correo duplicado (Message-ID): {email.message_id}")
            return
        
        # Si tiene adjuntos, descargarlos
        if email.has_attachments:
            self._download_attachments(email, mailbox)
        
        # Marcar como leído
        self._mark_as_read(email.message_id, mailbox)
        
        # Mover a carpeta procesados
        self._move_to_processed_folder(email.message_id, mailbox)
        
        # Registrar en base de datos
        self._log_email_processed(email, mailbox)
    
    def _download_attachments(self, email: OutlookEmail, mailbox: str):
        """Descarga los adjuntos del correo."""
        import requests
        
        user_id = mailbox
        url = f"https://graph.microsoft.com/v1.0/users/{user_id}/messages/{email.message_id}/attachments"
        
        headers = {
            "Authorization": f"Bearer {self._access_token}"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                for attachment in data.get("value", []):
                    # Solo procesar archivos
                    if "@odata.type" not in attachment or "#microsoft.graph.fileAttachment" not in attachment.get("@odata.type", ""):
                        continue
                    
                    filename = attachment.get("name", "unknown")
                    ext = os.path.splitext(filename)[1].lower()
                    
                    if ext not in self.allowed_extensions:
                        self.logger.debug(f"Extensión no permitida: {ext}")
                        continue
                    
                    # Decodificar contenido
                    content_bytes = None
                    if "contentBytes" in attachment:
                        import base64
                        content_bytes = base64.b64decode(attachment["contentBytes"])
                    
                    if content_bytes:
                        # Generar nombre de archivo seguro
                        timestamp = int(time.time())
                        safe_subject = re.sub(r'[<>:"/\|?*]', '', email.subject)[:20]
                        new_filename = f"Email_{timestamp}_{safe_subject}_{filename}"
                        filepath = os.path.join(self.input_folder, new_filename)
                        
                        with open(filepath, "wb") as f:
                            f.write(content_bytes)
                        
                        self.logger.info(f"Adjunto guardado: {new_filename}")
                        
                        # Registrar adjunto
                        email.attachments.append({
                            "filename": new_filename,
                            "original_name": filename,
                            "size": len(content_bytes)
                        })
                        
        except Exception as e:
            self.logger.error(f"Error al descargar adjuntos: {e}")
    
    def _is_duplicate_message_id(self, message_id: str) -> bool:
        """Verifica si el Message-ID ya fue procesado."""
        from modules.db_manager import DBManager
        
        db = DBManager.get_instance()
        
        query = """
            SELECT 1 FROM email_message_ids 
            WHERE message_id = %s
            LIMIT 1
        """
        
        try:
            result = db.fetch_one(query, (message_id,))
            return result is not None
        except Exception as e:
            self.logger.error(f"Error al verificar Message-ID: {e}")
            return False
    
    def _mark_as_read(self, message_id: str, mailbox: str):
        """Marca el correo como leído."""
        import requests
        
        user_id = mailbox
        url = f"https://graph.microsoft.com/v1.0/users/{user_id}/messages/{message_id}"
        
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "isRead": True
        }
        
        try:
            requests.patch(url, headers=headers, json=data, timeout=10)
        except Exception as e:
            self.logger.error(f"Error al marcar como leído: {e}")
    
    def _move_to_processed_folder(self, message_id: str, mailbox: str):
        """Mueve el correo a la carpeta de procesados."""
        import requests
        
        # Primero, obtener el ID de la carpeta de destino
        folder_id = self._get_or_create_processed_folder(mailbox)
        
        if not folder_id:
            self.logger.warning("No se pudo obtener/crear carpeta de procesados")
            return
        
        user_id = mailbox
        url = f"https://graph.microsoft.com/v1.0/users/{user_id}/messages/{message_id}/move"
        
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "destinationId": folder_id
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                self.logger.debug(f"Correo movido a carpeta procesados")
            else:
                self.logger.warning(f"Error al mover correo: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Error al mover correo: {e}")
    
    def _get_or_create_processed_folder(self, mailbox: str) -> Optional[str]:
        """Obtiene o crea la carpeta de correos procesados."""
        import requests
        
        user_id = mailbox
        
        # Primero buscar si existe la carpeta
        url = f"https://graph.microsoft.com/v1.0/users/{user_id}/mailFolders"
        
        headers = {
            "Authorization": f"Bearer {self._access_token}"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Buscar carpeta existente
                for folder in data.get("value", []):
                    if folder.get("displayName") == self.processed_folder:
                        return folder.get("id")
                
                # Crear carpeta si no existe
                create_url = f"https://graph.microsoft.com/v1.0/users/{user_id}/mailFolders"
                
                new_folder = {
                    "displayName": self.processed_folder
                }
                
                create_response = requests.post(create_url, headers=headers, json=new_folder, timeout=10)
                
                if create_response.status_code == 201:
                    return create_response.json().get("id")
                    
        except Exception as e:
            self.logger.error(f"Error al obtener/crear carpeta: {e}")
        
        return None
    
    def _log_email_processed(self, email: OutlookEmail, mailbox: str):
        """Registra el correo procesado en la base de datos."""
        from modules.db_manager import DBManager
        
        db = DBManager.get_instance()
        
        query = """
            INSERT INTO email_message_ids 
                (message_id, subject, sender_email, mailbox, processed_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (message_id) DO NOTHING
        """
        
        try:
            db.execute(query, (
                email.message_id,
                email.subject,
                email.sender_email,
                mailbox
            ))
        except Exception as e:
            self.logger.error(f"Error al registrar correo: {e}")


# Instancia global
_outlook_importer: Optional[OutlookGraphImporter] = None


def get_outlook_importer(config: dict = None, input_folder: str = None) -> Optional[OutlookGraphImporter]:
    """
    Obtiene la instancia global del importador de Outlook.
    
    Args:
        config: Configuración (opcional, usa config.yaml si no se proporciona)
        input_folder: Carpeta de entrada (opcional)
        
    Returns:
        Instancia de OutlookGraphImporter o None
    """
    global _outlook_importer
    
    if _outlook_importer is None:
        if config is None:
            # Cargar configuración de config.yaml
            import yaml
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
            
            try:
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                    config = full_config.get('outlook_importer', {})
            except Exception as e:
                logger.error(f"Error al cargar configuración: {e}")
                return None
        
        if input_folder is None:
            input_folder = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'input', 'emails'
            )
        
        _outlook_importer = OutlookGraphImporter(config, input_folder)
    
    return _outlook_importer


def start_outlook_importer():
    """Inicia el importador de Outlook."""
    importer = get_outlook_importer()
    if importer:
        importer.start()


def stop_outlook_importer():
    """Detiene el importador de Outlook."""
    global _outlook_importer
    if _outlook_importer:
        _outlook_importer.stop()
        _outlook_importer = None
