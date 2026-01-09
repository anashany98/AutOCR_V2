import os
import sys
import threading
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# Dependencies
from modules.db_manager import DBManager
from modules.logger_manager import setup_logger
from modules.file_utils import ensure_directories
from postbatch_processor import PipelineComponents, initialise_pipeline
from modules.classifier import DocumentClassifier
from modules.rag_manager import RAGManager
from modules.tool_manager import ToolManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configuration Path
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Thread-local storage for config caching within request context if needed
local = threading.local()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def load_configuration(reload: bool = False) -> Dict[str, Any]:
    """Load configuration from YAML. If reload=True or not cached, reads from disk."""
    if reload or getattr(local, "config", None) is None:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
                local.config = yaml.safe_load(handle) or {}
        else:
            local.config = {}
    return local.config

def save_configuration(config: Dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    local.config = config
    
    # Invalidate expensive resources if config changes (naive approach)
    # real invalidation might require more logic, but this matches original app.py intent
    if hasattr(local, "pipeline"):
        delattr(local, "pipeline")

# --------------------------------------------------------------------------- #
# Singletons
# --------------------------------------------------------------------------- #

_db_instance: Optional[DBManager] = None
_db_lock = threading.Lock()

_pipeline_instance: Optional[PipelineComponents] = None
_pipeline_lock = threading.Lock()

_classifier_instance: Optional[DocumentClassifier] = None
_classifier_lock = threading.Lock()

_rag_instance: Optional[RAGManager] = None
_rag_lock = threading.Lock()

_tool_instance: Optional[ToolManager] = None
_tool_lock = threading.Lock()

def get_db() -> DBManager:
    """Get database manager singleton with thread-safe initialization."""
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                config = load_configuration()
                _db_instance = DBManager(config)
    return _db_instance

def get_logger():
    # Keep logger in local or simplify
    if getattr(local, "logger", None) is None:
        config = load_configuration()
        app_conf = config.get("app", {})
        log_level = app_conf.get("log_level", "INFO")
        log_dir = PROJECT_ROOT / "web_app" / "logs"
        ensure_directories(str(log_dir))
        log_path = log_dir / "web_app.log"
        local.logger = setup_logger(str(log_path), level=log_level, db_manager=get_db())
    return local.logger

def get_pipeline() -> PipelineComponents:
    """Get pipeline components singleton with thread-safe initialization."""
    global _pipeline_instance
    if _pipeline_instance is None:
        with _pipeline_lock:
            if _pipeline_instance is None:
                config = load_configuration()
                _pipeline_instance = initialise_pipeline(config, str(PROJECT_ROOT), get_logger())
    return _pipeline_instance

def get_classifier() -> Optional[DocumentClassifier]:
    """Get document classifier singleton with thread-safe initialization."""
    global _classifier_instance
    if _classifier_instance is None:
        with _classifier_lock:
            if _classifier_instance is None:
                config = load_configuration()
                post_conf = config.get("postbatch", {})
                if post_conf.get("classification_enabled", True):
                    model_path = PROJECT_ROOT / "data" / "models" / "classifier.pkl"
                    _classifier_instance = DocumentClassifier(model_path=str(model_path))
                else:
                    _classifier_instance = None
    return _classifier_instance

def get_rag_manager() -> Optional[RAGManager]:
    """Get RAG manager singleton with thread-safe initialization."""
    global _rag_instance
    if _rag_instance is None:
        with _rag_lock:
            if _rag_instance is None:
                try:
                    rag_dir = PROJECT_ROOT / "data" / "rag_index"
                    _rag_instance = RAGManager(str(rag_dir))
                except Exception as e:
                    get_logger().error(f"Failed to load RAG Manager: {e}")
                    _rag_instance = None
    return _rag_instance

def get_tool_manager() -> ToolManager:
    """Get tool manager singleton with thread-safe initialization."""
    global _tool_instance
    if _tool_instance is None:
        with _tool_lock:
            if _tool_instance is None:
                pipeline = get_pipeline()
                vision = pipeline.vision_manager if pipeline else None
                _tool_instance = ToolManager(get_db(), str(PROJECT_ROOT), vision_manager=vision)
    return _tool_instance
def reload_classifier():
    """Force reload of the classifier instance."""
    global _classifier_instance
    with _classifier_lock:
        _classifier_instance = None
        get_classifier() # Re-initialize immediately

_llm_instance: Optional["LLMClient"] = None
_llm_lock = threading.Lock()

def get_llm_client():
    """Get LLM Client singleton with thread-safe initialization."""
    global _llm_instance
    if _llm_instance is None:
        with _llm_lock:
            if _llm_instance is None:
                try:
                    from modules.llm_client import LLMClient
                    config = load_configuration()
                    llm_conf = config.get("llm", {})
                    _llm_instance = LLMClient(llm_conf, get_logger())
                except Exception as e:
                    get_logger().error(f"Failed to load LLM Client: {e}")
                    _llm_instance = None
    return _llm_instance
