import logging
import torch
import gc
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ResourceOrchestrator:
    """
    Manages heavy AI models to prevent Out-Of-Memory (OOM) errors on shared GPUs.
    Ensures only one heavy model (Florence-2, SAM, SD, PaddleVL) is active at a time
    if VRAM is tight.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ResourceOrchestrator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.active_model_name: Optional[str] = None
            self.models: Dict[str, Any] = {}
            self.initialized = True
            self.vram_threshold_gb = 2.0 # Keep at least 2GB free

    def register_model(self, name: str, model_instance: Any):
        """Register a model instance for management."""
        self.models[name] = model_instance

    def request_model(self, name: str):
        """
        Requests a model to be active. 
        Unloads other heavy models if necessary.
        """
        if self.active_model_name == name:
            return

        logger.info(f"🔄 Orchestrator: Switching active model to '{name}'...")
        
        # Unload current active model if different
        if self.active_model_name and self.active_model_name in self.models:
            self._unload_model(self.active_model_name)

        # Clear cache
        self.clear_vram()
        
        self.active_model_name = name
        logger.info(f"✅ Orchestrator: '{name}' is now active.")

    def _unload_model(self, name: str):
        """Moves model to CPU or deletes reference to free VRAM."""
        model_obj = self.models.get(name)
        if not model_obj:
            return

        logger.info(f"📥 Orchestrator: Relieving '{name}' from VRAM...")
        try:
            # Check if it has a custom unload method
            if hasattr(model_obj, 'unload'):
                model_obj.unload()
            elif hasattr(model_obj, 'to'):
                model_obj.to("cpu")
            elif hasattr(model_obj, 'model') and hasattr(model_obj.model, 'to'):
                model_obj.model.to("cpu")
            
            self.clear_vram()
        except Exception as e:
            logger.warning(f"Orchestrator: Failed to unload '{name}': {e}")

    def clear_vram(self):
        """Force garbage collection and CUDA cache clearing."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.debug(f"🧹 VRAM Cleared. Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

    def get_vram_status(self) -> Dict[str, float]:
        """Provides current VRAM stats."""
        status = {"available": False, "allocated_gb": 0.0, "reserved_gb": 0.0}
        if torch.cuda.is_available():
            status["available"] = True
            status["allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
            status["reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
            status["total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        return status
