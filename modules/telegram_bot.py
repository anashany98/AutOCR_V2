"""
Telegram Bot integration for AutoOCR.

Provides a bot interface for managers to query documents and get AI assistance.
"""
import os
import json
import logging
import hashlib
import hmac
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

# Try to import python-telegram-bot
try:
    from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, ConversationHandler, CallbackQueryHandler
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import our PostgreSQL-based auth
from modules.telegram_gestores_db import get_telegram_gestores_db, TelegramGestoresDB

# Chat V2 API client (for querying documents)
CHAT_V2_API_URL = os.environ.get('CHAT_V2_API_URL', 'http://localhost:5000/api/v2/chat/query')


@dataclass
class TelegramUser:
    """Represents a Telegram user linked to AutoOCR."""
    telegram_id: int
    username: str
    first_name: str
    last_name: str = ""
    autoocr_user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    is_authenticated: bool = False


class TelegramAuth:
    """Handles authentication between Telegram and AutoOCR using PostgreSQL."""
    
    def __init__(self):
        self._db = get_telegram_gestores_db()
    
    def register_user(self, telegram_id: int, username: str, 
                    first_name: str, last_name: str = "") -> Optional[TelegramUser]:
        """Register or update a Telegram user in PostgreSQL."""
        # Try to get existing user
        gestor = self._db.get_by_telegram_id(telegram_id)
        
        if gestor:
            # Update existing
            # Note: This returns the updated gestor but we return TelegramUser for compatibility
            self._db.update_last_seen(telegram_id, "start")
            return TelegramUser(
                telegram_id=gestor.telegram_id,
                username=gestor.username,
                first_name=gestor.first_name,
                last_name=gestor.last_name,
                autoocr_user_id=gestor.user_id,
                tenant_id=gestor.tenant_id,
                is_authenticated=gestor.is_verified
            )
        else:
            # Create new user
            new_gestor = self._db.create(telegram_id, username, first_name, last_name)
            if new_gestor:
                return TelegramUser(
                    telegram_id=new_gestor.telegram_id,
                    username=new_gestor.username,
                    first_name=new_gestor.first_name,
                    last_name=new_gestor.last_name,
                    autoocr_user_id=new_gestor.user_id,
                    tenant_id=new_gestor.tenant_id,
                    is_authenticated=False
                )
        return None
    
    def link_to_autoocr(self, telegram_id: int, user_id: str, tenant_id: str, hotel_id: str = None) -> bool:
        """Link a Telegram user to an AutoOCR user in PostgreSQL."""
        success = self._db.link_to_autoocr(telegram_id, user_id, tenant_id, hotel_id)
        if success:
            self._db.log_command(telegram_id, "login", success=True, tenant_id=tenant_id, user_id=user_id)
        return success
    
    def get_user(self, telegram_id: int) -> Optional[TelegramUser]:
        """Get a Telegram user by ID from PostgreSQL."""
        gestor = self._db.get_by_telegram_id(telegram_id)
        if gestor:
            return TelegramUser(
                telegram_id=gestor.telegram_id,
                username=gestor.username,
                first_name=gestor.first_name,
                last_name=gestor.last_name,
                autoocr_user_id=gestor.user_id,
                tenant_id=gestor.tenant_id,
                is_authenticated=gestor.is_verified
            )
        return None
    
    def authenticate(self, telegram_id: int, user_id: str, tenant_id: str, hotel_id: str = None) -> bool:
        """Authenticate a user with AutoOCR credentials."""
        gestor = self._db.get_by_telegram_id(telegram_id)
        if gestor:
            success = self._db.link_to_autoocr(telegram_id, user_id, tenant_id, hotel_id)
            if success:
                self._db.log_command(telegram_id, "authenticate", success=True, 
                                   tenant_id=tenant_id, user_id=user_id)
            return success
        return False
    
    def update_last_command(self, telegram_id: int, command: str) -> bool:
        """Update the last command executed by a user."""
        return self._db.update_last_seen(telegram_id, command)


class TelegramCommands:
    """Handles Telegram bot commands."""
    
    # Command definitions
    START = "start"
    HELP = "help"
    SEARCH = "buscar"
    UPLOAD = "subir"
    MYDOCS = "misdocs"
    STATS = "stats"
    AUTH = "login"
    LOGOUT = "logout"
    
    @staticmethod
    def get_commands() -> List[BotCommand]:
        """Get list of available bot commands."""
        return [
            BotCommand("start", "Iniciar el bot"),
            BotCommand("buscar", "Buscar documentos"),
            BotCommand("subir", "Subir documento para OCR"),
            BotCommand("misdocs", "Ver mis documentos"),
            BotCommand("stats", "EstadÃ­sticas"),
            BotCommand("login", "Vincular cuenta AutoOCR"),
            BotCommand("logout", "Desvincular cuenta"),
            BotCommand("ayuda", "Mostrar ayuda"),
        ]


class TelegramBot:
    """Main Telegram bot handler for AutoOCR."""
    
    def __init__(self, token: str, webhook_url: Optional[str] = None):
        if not TELEGRAM_AVAILABLE:
            raise RuntimeError("python-telegram-bot not installed")
        
        self.token = token
        self.webhook_url = webhook_url
        self.auth = TelegramAuth()
        self.application: Optional[Application] = None
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user = update.effective_user
        if not user:
            return
        
        # Register user
        self.auth.register_user(
            telegram_id=user.id,
            username=user.username or "",
            first_name=user.first_name or "",
            last_name=user.last_name or ""
        )
        
        welcome_message = (
            f"ðŸ‘‹ Â¡Hola {user.first_name}!\n\n"
            "Bienvenido al bot de AutoOCR.\n\n"
            "Puedes:\n"
            "â€¢ /buscar <texto> - Buscar documentos\n"
            "â€¢ /subir - Subir documento para OCR\n"
            "â€¢ /misdocs - Ver tus documentos\n"
            "â€¢ /login - Vincular tu cuenta AutoOCR\n"
            "â€¢ /ayuda - Ver todos los comandos\n\n"
            "Usa /login para vincular tu cuenta."
        )
        
        await update.message.reply_text(welcome_message)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help and /ayuda commands."""
        help_text = (
            "ðŸ“– *Comandos disponibles*\n\n"
            "â€¢ /start - Iniciar el bot\n"
            "â€¢ /buscar <texto> - Buscar documentos en el sistema\n"
            "â€¢ /subir - Subir un documento para procesar con OCR\n"
            "â€¢ /misdocs - Ver tus documentos recientes\n"
            "â€¢ /stats - Ver estadÃ­sticas del sistema\n"
            "â€¢ /login - Vincular cuenta de AutoOCR\n"
            "â€¢ /logout - Desvincular cuenta\n"
            "â€¢ /ayuda - Mostrar esta ayuda\n\n"
            "_Para buscar documentos necesitas estar autenticado con /login_"
        )
        
        await update.message.reply_text(help_text, parse_mode="Markdown")
    
    async def search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /buscar command."""
        user = update.effective_user
        if not user:
            return
        
        telegram_user = self.auth.get_user(user.id)
        
        # Check authentication
        if not telegram_user or not telegram_user.is_authenticated:
            await update.message.reply_text(
                "âš ï¸ Necesitas vincular tu cuenta primero.\n"
                "Usa /login para autenticarte."
            )
            return
        
        # Get search query
        query = " ".join(context.args)
        if not query:
            await update.message.reply_text(
                "ðŸ” *Uso:* /buscar <texto a buscar>\n\n"
                "_Ejemplo: /buscar facturas enero 2025_",
                parse_mode="Markdown"
            )
            return
        
        await update.message.reply_text(f"ðŸ” Buscando: {query}...")
        
        try:
            # Call the chat API
            result = await self._search_documents(
                query=query,
                tenant_id=telegram_user.tenant_id,
                user_id=telegram_user.autoocr_user_id
            )
            
            if result.get("answer"):
                await update.message.reply_text(
                    result["answer"],
                    parse_mode="Markdown"
                )
                
                # Show sources if available
                if result.get("sources"):
                    sources_text = "\n".join([
                        f"â€¢ {s.get('filename', 'Documento')}"
                        for s in result["sources"][:5]
                    ])
                    await update.message.reply_text(
                        f"ðŸ“„ *Fuentes:*\n{sources_text}",
                        parse_mode="Markdown"
                    )
            else:
                await update.message.reply_text(
                    "â“ No encontrÃ© documentos relacionados con tu bÃºsqueda."
                )
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            await update.message.reply_text(
                f"âŒ Error al buscar: {str(e)}"
            )
    
    async def mydocs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /misdocs command."""
        user = update.effective_user
        if not user:
            return
        
        telegram_user = self.auth.get_user(user.id)
        
        if not telegram_user or not telegram_user.is_authenticated:
            await update.message.reply_text(
                "âš ï¸ Necesitas vincular tu cuenta primero.\nUsa /login"
            )
            return
        
        try:
            # Get recent documents
            docs = await self._get_recent_documents(
                tenant_id=telegram_user.tenant_id,
                user_id=telegram_user.autoocr_user_id
            )
            
            if docs:
                docs_text = "ðŸ“„ *Tus documentos recientes:*\n\n"
                for doc in docs[:10]:
                    filename = doc.get("filename", "Sin nombre")
                    doc_type = doc.get("document_type", "Desconocido")
                    date = doc.get("created_at", "")[:10]
                    docs_text += f"â€¢ {filename} ({doc_type}) - {date}\n"
                
                await update.message.reply_text(docs_text, parse_mode="Markdown")
            else:
                await update.message.reply_text(
                    "ðŸ“­ No tienes documentos procesados aÃºn."
                )
                
        except Exception as e:
            logger.error(f"Get docs error: {e}")
            await update.message.reply_text(f"âŒ Error: {str(e)}")
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""
        user = update.effective_user
        if not user:
            return
        
        telegram_user = self.auth.get_user(user.id)
        
        if not telegram_user or not telegram_user.is_authenticated:
            await update.message.reply_text(
                "âš ï¸ Necesitas vincular tu cuenta primero.\nUsa /login"
            )
            return
        
        try:
            stats = await self._get_stats(
                tenant_id=telegram_user.tenant_id
            )
            
            stats_text = (
                f"ðŸ“Š *EstadÃ­sticas*\n\n"
                f"â€¢ Total documentos: {stats.get('total_docs', 0)}\n"
                f"â€¢ Procesados: {stats.get('processed', 0)}\n"
                f"â€¢ Pendientes: {stats.get('pending', 0)}\n"
                f"â€¢ Fallidos: {stats.get('failed', 0)}"
            )
            
            await update.message.reply_text(stats_text, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Stats error: {e}")
            await update.message.reply_text(f"âŒ Error: {str(e)}")
    
    async def login_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /login command - link Telegram to AutoOCR account.
        
        This command starts a conversation to collect user credentials.
        Format: /login <user_id> <tenant_id> [hotel_id]
        Example: /login juan.gomez empresa1
        """
        user = update.effective_user
        if not user:
            return
        
        # Parse arguments
        args = context.args
        
        if not args or len(args) < 2:
            # Show inline keyboard with login instructions
            keyboard = [
                [InlineKeyboardButton("ðŸ“‹ CÃ³mo obtener credenciales", callback_data="help_login")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                "âš ï¸ *Uso incorrecto*\n\n"
                "Usa el comando asÃ­:\n"
                "`/login <usuario> <empresa>`\n\n"
                "Ejemplo: `/login juan.gomez empresa1`\n\n"
                "El usuario y empresa deben existir en AutoOCR.",
                parse_mode="Markdown",
                reply_markup=reply_markup
            )
            return
        
        user_id = args[0]
        tenant_id = args[1]
        hotel_id = args[2] if len(args) > 2 else None
        
        # Register user first
        self.auth.register_user(
            telegram_id=user.id,
            username=user.username or "",
            first_name=user.first_name or "",
            last_name=user.last_name or ""
        )
        
        # Link to AutoOCR
        success = self.auth.link_to_autoocr(
            telegram_id=user.id,
            user_id=user_id,
            tenant_id=tenant_id,
            hotel_id=hotel_id
        )
        
        if success:
            await update.message.reply_text(
                f"âœ… Â¡Cuenta vinculada correctamente!\n\n"
                f"ðŸ‘¤ Usuario: {user_id}\n"
                f"ðŸ¢ Empresa: {tenant_id}\n"
                f"ðŸ†” Tu Telegram: {user.id}\n\n"
                "Ya puedes buscar documentos con /buscar"
            )
        else:
            await update.message.reply_text(
                "âŒ Error al vincular cuenta.\n"
                "Verifica que el usuario y empresa existan en AutoOCR."
            )
    
    async def logout_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /logout command."""
        user = update.effective_user
        if not user:
            return
        
        telegram_user = self.auth.get_user(user.id)
        if telegram_user:
            telegram_user.is_authenticated = False
            telegram_user.autoocr_user_id = None
            telegram_user.tenant_id = None
            
            # Update in database
            db = get_telegram_gestores_db()
            db.update_last_seen(user.id, "logout")
        
        await update.message.reply_text(
            "ðŸ‘‹ Cuenta desvinculada. Hasta luego!"
        )
    
    async def menu_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /menu command - show inline keyboard menu."""
        user = update.effective_user
        if not user:
            return
        
        telegram_user = self.auth.get_user(user.id)
        
        # Build main menu keyboard
        keyboard = [
            [InlineKeyboardButton("ðŸ” Buscar documentos", callback_data="menu_search")],
            [InlineKeyboardButton("ðŸ“Ž Subir documento", callback_data="menu_upload")],
            [InlineKeyboardButton("ðŸ“‹ Mis documentos", callback_data="menu_mydocs")],
            [InlineKeyboardButton("ðŸ“Š EstadÃ­sticas", callback_data="menu_stats")],
        ]
        
        # Add login/logout based on auth state
        if telegram_user and telegram_user.is_authenticated:
            keyboard.append([InlineKeyboardButton("ðŸšª Cerrar sesiÃ³n", callback_data="menu_logout")])
        else:
            keyboard.append([InlineKeyboardButton("ðŸ” Iniciar sesiÃ³n", callback_data="menu_login")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "ðŸ“± *MenÃº Principal*\n\n"
            "Selecciona una opciÃ³n:",
            parse_mode="Markdown",
            reply_markup=reply_markup
        )
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards."""
        query = update.callback_query
        if not query:
            return
        
        # Answer the callback to stop loading animation
        await query.answer()
        
        user = update.effective_user
        telegram_user = self.auth.get_user(user.id) if user else None
        
        callback_data = query.data
        
        if callback_data == "menu_search":
            await query.edit_message_text(
                "ðŸ” *Buscar Documentos*\n\n"
                "Usa el comando:\n`/buscar <texto>`\n\n"
                "Ejemplo: `/buscar facturas enero`",
                parse_mode="Markdown"
            )
        elif callback_data == "menu_upload":
            if not telegram_user or not telegram_user.is_authenticated:
                await query.edit_message_text(
                    "âš ï¸ Necesitas iniciar sesiÃ³n primero.\n"
                    "Usa /login <usuario> <empresa>"
                )
            else:
                await query.edit_message_text(
                    "ðŸ“Ž *Subir Documento*\n\n"
                    "EnvÃ­ame el documento que deseas procesar.\n"
                    "Formatos: PDF, JPG, PNG, TIFF"
                )
        elif callback_data == "menu_mydocs":
            await query.edit_message_text(
                "ðŸ“‹ *Mis Documentos*\n\n"
                "Usa el comando:\n`/misdocs`"
            )
        elif callback_data == "menu_stats":
            await query.edit_message_text(
                "ðŸ“Š *EstadÃ­sticas*\n\n"
                "Usa el comando:\n`/stats`"
            )
        elif callback_data == "menu_login":
            await query.edit_message_text(
                "ðŸ” *Iniciar SesiÃ³n*\n\n"
                "Usa el comando:\n`/login <usuario> <empresa>`\n\n"
                "Ejemplo: `/login juan.gomez empresa1`"
            )
        elif callback_data == "menu_logout":
            if telegram_user:
                telegram_user.is_authenticated = False
                telegram_user.autoocr_user_id = None
                telegram_user.tenant_id = None
            await query.edit_message_text(
                "ðŸšª SesiÃ³n cerrada. Â¡Hasta luego!"
            )
        elif callback_data == "help_login":
            await query.edit_message_text(
                "ðŸ“‹ *CÃ³mo obtener credenciales*\n\n"
                "1. Consulta con tu administrador el usuario y empresa de AutoOCR\n\n"
                "2. Usa el comando:\n`/login <usuario> <empresa>`\n\n"
                "3. Â¡Listo! Ya puedes usar el bot"
            )
        elif callback_data.startswith("search_"):
            doc_id = callback_data.replace("search_", "")
            await query.edit_message_text(
                f"ðŸ” Para buscar en este documento, usa:\n`/buscar <texto>`",
                parse_mode="Markdown"
            )
        elif callback_data.startswith("details_"):
            doc_id = callback_data.replace("details_", "")
            await query.edit_message_text(
                f"ðŸ“‹ ID del documento: `{doc_id}`\n\n"
                "Usa `/buscar` para mÃ¡s detalles.",
                parse_mode="Markdown"
            )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages - treat as search query."""
        user = update.effective_user
        if not user or not update.message:
            return
        
        text = update.message.text
        
        # Ignore commands
        if text.startswith("/"):
            return
        
        telegram_user = self.auth.get_user(user.id)
        
        if not telegram_user or not telegram_user.is_authenticated:
            await update.message.reply_text(
                "âš ï¸ Usa /login para vincular tu cuenta primero."
            )
            return
        
        # Treat as search query
        await update.message.reply_text(f"ðŸ” Buscando: {text}...")
        
        try:
            result = await self._search_documents(
                query=text,
                tenant_id=telegram_user.tenant_id,
                user_id=telegram_user.autoocr_user_id
            )
            
            if result.get("answer"):
                await update.message.reply_text(result["answer"])
            else:
                await update.message.reply_text(
                    "â“ No encontrÃ© informaciÃ³n relacionada."
                )
                
        except Exception as e:
            logger.error(f"Message handler error: {e}")
            await update.message.reply_text(f"âŒ Error: {str(e)}")
    
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document uploads from Telegram.
        
        Downloads the document from Telegram and sends it to the OCR pipeline.
        """
        user = update.effective_user
        if not user or not update.message:
            return
        
        telegram_user = self.auth.get_user(user.id)
        
        if not telegram_user or not telegram_user.is_authenticated:
            await update.message.reply_text(
                "âš ï¸ Usa /login para vincular tu cuenta primero."
            )
            return
        
        # Check for document
        document = update.message.document
        if not document:
            await update.message.reply_text(
                "â“ No detectÃ© ningÃºn documento."
            )
            return
        
        # Validate file type
        allowed_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
        file_name = document.file_name or ""
        ext = os.path.splitext(file_name)[1].lower()
        
        if ext not in allowed_extensions:
            await update.message.reply_text(
                f"âŒ Tipo de archivo no soportado: {ext}\n"
                f"Formatos permitidos: {', '.join(allowed_extensions)}"
            )
            return
        
        # Check file size (max 20MB)
        if document.file_size and document.file_size > 20 * 1024 * 1024:
            await update.message.reply_text(
                "âŒ El archivo es demasiado grande. MÃ¡ximo 20MB."
            )
            return
        
        await update.message.reply_text(
            f"ðŸ“„ *{document.file_name}*\n"
            f"â³ Descargando y procesando con OCR...",
            parse_mode="Markdown"
        )
        
        try:
            # Download file from Telegram
            file = await context.bot.get_file(document.file_id)
            
            # Create temp directory
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp', 'telegram_uploads')
            os.makedirs(temp_dir, exist_ok=True)
            
            # SECURE: Sanitize filename to prevent path traversal attacks
            # Also fix: use chunked reading to avoid memory issues with large files
            safe_filename = os.path.basename(document.file_name)  # Only get basename, remove any path components
            if not safe_filename or safe_filename.startswith('.'):
                safe_filename = f"document_{document.file_id}"
            
            # Save file locally
            file_path = os.path.join(temp_dir, f"{user.id}_{document.file_id}_{safe_filename}")
            await file.download_to_drive(file_path)
            
            # Process with OCR pipeline
            result = await self._process_ocr_document(
                file_path=file_path,
                tenant_id=telegram_user.tenant_id,
                user_id=telegram_user.autoocr_user_id or str(user.id),
                original_filename=document.file_name
            )
            
            if result.get('success'):
                # Create inline keyboard with actions
                keyboard = [
                    [InlineKeyboardButton("ðŸ” Buscar en documento", callback_data=f"search_{result.get('doc_id', '')}")],
                    [InlineKeyboardButton("ðŸ“‹ Ver detalles", callback_data=f"details_{result.get('doc_id', '')}")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(
                    f"âœ… *Documento procesado correctamente*\n\n"
                    f"ðŸ“ *Resultado:*\n{result.get('extracted_text', 'Sin texto extraÃ­do')[:500]}...\n\n"
                    f"ðŸ†” ID: `{result.get('doc_id', 'N/A')}`",
                    parse_mode="Markdown",
                    reply_markup=reply_markup
                )
            else:
                await update.message.reply_text(
                    f"âš ï¸ Documento procesado con advertencias:\n{result.get('message', 'Error desconocido')}"
                )
            
            # Cleanup temp file
            try:
                os.remove(file_path)
            except:
                pass
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            await update.message.reply_text(
                f"âŒ Error al procesar: {str(e)}"
            )
    
    async def _process_ocr_document(self, file_path: str, tenant_id: str,
                                   user_id: str, original_filename: str) -> Dict[str, Any]:
        """Queue a document for OCR processing."""
        del tenant_id, original_filename
        try:
            from modules.tasks import process_document_task

            options: Dict[str, Any] = {
                "delete_original": False,
                "ocr_enabled": True,
                "classification_enabled": True,
                "doc_type": "other",
            }
            try:
                options["owner_id"] = int(user_id)
            except Exception:
                options["owner_id"] = user_id

            task = process_document_task(file_path, options)
            task_id = str(getattr(task, "id", "") or "")

            return {
                "success": True,
                "doc_id": None,
                "task_id": task_id or None,
                "message": "Documento en cola para procesamiento",
                "extracted_text": "El documento se esta procesando en segundo plano.",
            }
        except Exception as e:
            logger.error(f"Error in _process_ocr_document: {e}")
            return {
                "success": False,
                "message": str(e),
            }

    async def _search_documents(self, query: str, tenant_id: str, 
                               user_id: str) -> Dict[str, Any]:
        """Search documents using the Chat V2 API (RAG).
        
        Calls POST /api/v2/chat/query with the search query.
        
        Args:
            query: Search query text
            tenant_id: Tenant ID for the query
            user_id: User ID making the query
            
        Returns:
            Dict with 'answer' and 'sources' keys
        """
        import aiohttp
        
        # Get user's hotel_id from database
        db = get_telegram_gestores_db()
        gestor = db.get_by_telegram_id(int(user_id.replace('telegram_', '') if user_id.startswith('telegram_') else user_id))
        
        # For now, use telegram_id if user_id is not a number
        try:
            telegram_id = int(user_id) if user_id.isdigit() else None
        except (ValueError, AttributeError):
            telegram_id = None
        
        if telegram_id:
            gestor = db.get_by_telegram_id(telegram_id)
        
        hotel_ids = [gestor.hotel_id] if gestor and gestor.hotel_id else []
        
        # Build request payload for Chat V2 API
        payload = {
            "query": query,
            "tenant_id": tenant_id,
            "hotel_ids": hotel_ids,
            "user_id": user_id,
            "stream": False,
            "include_sources": True
        }
        
        # Get API URL from config or use default
        api_url = os.environ.get('CHAT_V2_API_URL', 'http://localhost:5000/api/v2/chat/query')
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "answer": data.get("response", data.get("answer", "No se encontrÃ³ informaciÃ³n.")),
                            "sources": data.get("sources", data.get("documents", []))
                        }
                    else:
                        logger.error(f"Chat V2 API error: {response.status}")
                        return {
                            "answer": f"Error al buscar documentos (cÃ³digo {response.status})",
                            "sources": []
                        }
        except aiohttp.ClientError as e:
            logger.error(f"Error calling Chat V2 API: {e}")
            return {
                "answer": f"Error de conexiÃ³n: {str(e)}",
                "sources": []
            }
        except Exception as e:
            logger.error(f"Unexpected error in _search_documents: {e}")
            return {
                "answer": f"Error inesperado: {str(e)}",
                "sources": []
            }
    
    async def _get_recent_documents(self, tenant_id: str, 
                                    user_id: str) -> List[Dict]:
        """Get recent documents for user from the database.
        
        Args:
            tenant_id: Tenant ID
            user_id: User ID
            
        Returns:
            List of recent documents
        """
        from modules.db_manager import DBManager
        
        db = DBManager.get_instance()
        
        query = """
            SELECT id, filename, file_path, status, created_at, extracted_text 
            FROM documents 
            WHERE tenant_id = %s 
            ORDER BY created_at DESC 
            LIMIT 10
        """
        
        try:
            results = db.fetch_all(query, (tenant_id,))
            return [
                {
                    "id": str(row[0]),
                    "filename": row[1],
                    "file_path": row[2],
                    "status": row[3],
                    "created_at": row[4].isoformat() if row[4] else None,
                    "text_preview": (row[5][:100] + "...") if row[5] and len(row[5]) > 100 else row[5]
                }
                for row in results
            ]
        except Exception as e:
            logger.error(f"Error fetching recent documents: {e}")
            return []
    
    async def _get_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get statistics for tenant from database.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            Dict with statistics
        """
        from modules.db_manager import DBManager
        
        db = DBManager.get_instance()
        
        query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as processed,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM documents 
            WHERE tenant_id = %s
        """
        
        try:
            result = db.fetch_one(query, (tenant_id,))
            if result:
                return {
                    "total_docs": result[0] or 0,
                    "processed": result[1] or 0,
                    "pending": result[2] or 0,
                    "failed": result[3] or 0
                }
        except Exception as e:
            logger.error(f"Error fetching stats: {e}")
        
        return {
            "total_docs": 0,
            "processed": 0,
            "pending": 0,
            "failed": 0
        }
    
    def setup_handlers(self, application: Application):
        """Setup all command and message handlers."""
        # Commands
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("ayuda", self.help_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("buscar", self.search_command))
        application.add_handler(CommandHandler("subir", self.upload_command))
        application.add_handler(CommandHandler("misdocs", self.mydocs_command))
        application.add_handler(CommandHandler("stats", self.stats_command))
        application.add_handler(CommandHandler("login", self.login_command))
        application.add_handler(CommandHandler("logout", self.logout_command))
        application.add_handler(CommandHandler("menu", self.menu_command))
        
        # Callback query handler for inline keyboards
        application.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Message handlers
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
        application.add_handler(
            MessageHandler(filters.Document.ALL, self.handle_document)
        )
    
    async def upload_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /subir command - prompt for document upload."""
        user = update.effective_user
        if not user:
            return
        
        telegram_user = self.auth.get_user(user.id)
        
        if not telegram_user or not telegram_user.is_authenticated:
            await update.message.reply_text(
                "âš ï¸ Usa /login para vincular tu cuenta primero."
            )
            return
        
        await update.message.reply_text(
            "ðŸ“Ž Por favor, envÃ­ame el documento que deseas procesar.\n"
            "Formatos soportados: PDF, JPG, PNG, TIFF"
        )


# Global bot instance
_telegram_bot: Optional[TelegramBot] = None


def get_telegram_bot(token: Optional[str] = None, 
                    webhook_url: Optional[str] = None) -> Optional[TelegramBot]:
    """Get or create the global Telegram bot instance."""
    global _telegram_bot
    
    if not TELEGRAM_AVAILABLE:
        logger.warning("python-telegram-bot not installed")
        return None
    
    if _telegram_bot is None:
        if token is None:
            token = os.environ.get("TELEGRAM_BOT_TOKEN")
        
        if token:
            _telegram_bot = TelegramBot(token, webhook_url)
    
    return _telegram_bot


def verify_webhook(token: str, data: str, signature: str) -> bool:
    """Verify Telegram webhook signature."""
    secret = hashlib.sha256(token.encode()).digest()
    expected = hmac.new(secret, data.encode()).hexdigest()
    return hmac.compare_digest(expected, signature)

