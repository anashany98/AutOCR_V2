"""
OpenAPI/Swagger Documentation for AutoOCR API.

This module provides API documentation using Flasgger (Swagger UI).
"""
import os
from flask import Blueprint, jsonify, current_app

# Try to import Flasgger
try:
    from flasgger import Swagger
    FLASK_SWAGGER_AVAILABLE = True
except ImportError:
    FLASK_SWAGGER_AVAILABLE = False

api_docs_bp = Blueprint("api_docs", __name__, url_prefix="/api/docs")

# OpenAPI Specification
API_SPEC = {
    "openapi": "3.0.0",
    "info": {
        "title": "AutoOCR API",
        "description": """
## Sistema de OCR Inteligente para Hoteles

AutoOCR es un sistema de procesamiento de documentos automatizado con las siguientes características:

- **OCR**: Extracción de texto de imágenes y PDFs
- **Chat**: Asistencia IA para buscar en documentos
- **Clasificación**: Clasificación automática de documentos
- **Visión**: Análisis de imágenes con IA
- **Telegram**: Bot de Telegram para gestión de documentos

### Autenticación

La API usa autenticación basada en sesiones de Flask-Login. 
Para endpoints protegidos, primero haga login en `/auth/login`.

### Rate Limiting

- 200 requests por minuto
- 10 requests por segundo
""",
        "version": "2.0.0",
        "contact": {
            "name": "AutoOCR Support",
            "email": "soporte@autocr.example.com"
        }
    },
    "servers": [
        {
            "url": "http://localhost:5000",
            "description": "Servidor de desarrollo"
        },
        {
            "url": "https://autocr.example.com",
            "description": "Servidor de producción"
        }
    ],
    "tags": [
        {"name": "Health", "description": "Endpoints de salud del sistema"},
        {"name": "Auth", "description": "Autenticación y usuarios"},
        {"name": "Documents", "description": "Gestión de documentos"},
        {"name": "OCR", "description": "Procesamiento OCR"},
        {"name": "Chat", "description": "Asistente de chat IA"},
        {"name": "Vision", "description": "Análisis de visión por IA"},
        {"name": "Admin", "description": "Endpoints de administración"}
    ],
    "paths": {
        "/health": {
            "get": {
                "tags": ["Health"],
                "summary": "Health check público",
                "description": "Endpoint para health checks de load balancers y orquestación (Kubernetes). No requiere autenticación.",
                "responses": {
                    "200": {
                        "description": "Sistema saludable",
                        "content": {
                            "application/json": {
                                "example": {
                                    "status": "healthy",
                                    "checks": {
                                        "database": "ok",
                                        "redis": "ok",
                                        "storage": "ok"
                                    }
                                }
                            }
                        }
                    },
                    "503": {
                        "description": "Sistema no saludable"
                    }
                }
            }
        },
        "/api/status": {
            "get": {
                "tags": ["Health"],
                "summary": "Estado detallado del sistema",
                "description": "Requiere autenticación. Devuelve estado detallado de web, worker y base de datos.",
                "security": [{"cookieAuth": []}],
                "responses": {
                    "200": {
                        "description": "Estado del sistema",
                        "content": {
                            "application/json": {
                                "example": {
                                    "web": "online",
                                    "worker": "online",
                                    "database": "online"
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autenticado"
                    }
                }
            }
        },
        "/auth/login": {
            "post": {
                "tags": ["Auth"],
                "summary": "Iniciar sesión",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/x-www-form-urlencoded": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "username": {"type": "string"},
                                    "password": {"type": "string", "format": "password"}
                                },
                                "required": ["username", "password"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Login exitoso"
                    },
                    "401": {
                        "description": "Credenciales inválidas"
                    }
                }
            }
        },
        "/auth/logout": {
            "post": {
                "tags": ["Auth"],
                "summary": "Cerrar sesión",
                "security": [{"cookieAuth": []}],
                "responses": {
                    "200": {
                        "description": "Logout exitoso"
                    }
                }
            }
        },
        "/api/documents": {
            "get": {
                "tags": ["Documents"],
                "summary": "Listar documentos",
                "description": "Lista documentos con paginación. Requiere autenticación.",
                "security": [{"cookieAuth": []}],
                "parameters": [
                    {"name": "hotel_id", "in": "query", "schema": {"type": "integer"}},
                    {"name": "page", "in": "query", "schema": {"type": "integer", "default": 1}},
                    {"name": "per_page", "in": "query", "schema": {"type": "integer", "default": 50}}
                ],
                "responses": {
                    "200": {
                        "description": "Lista de documentos"
                    }
                }
            },
            "post": {
                "tags": ["Documents"],
                "summary": "Subir documento",
                "description": "Sube un archivo para procesamiento OCR. Requiere autenticación.",
                "security": [{"cookieAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "file": {"type": "string", "format": "binary"},
                                    "hotel_id": {"type": "integer"}
                                },
                                "required": ["file"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Documento procesado"
                    }
                }
            }
        },
        "/api/v2/chat/query": {
            "post": {
                "tags": ["Chat"],
                "summary": "Consultar documentos con IA",
                "description": "Envía una consulta al asistente de IA. Requiere autenticación.",
                "security": [{"cookieAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "session_id": {"type": "string"},
                                    "hotel_id": {"type": "integer"},
                                    "doc_id": {"type": "integer"}
                                },
                                "required": ["query"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Respuesta del asistente"
                    }
                }
            }
        },
        "/api/vision/analyze": {
            "post": {
                "tags": ["Vision"],
                "summary": "Analizar imagen con IA",
                "description": "Analiza una imagen para detectar objetos, colores y materiales. Requiere autenticación.",
                "security": [{"cookieAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "file": {"type": "string", "format": "binary"}
                                },
                                "required": ["file"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Análisis completado"
                    }
                }
            }
        }
    },
    "components": {
        "securitySchemes": {
            "cookieAuth": {
                "type": "apiKey",
                "in": "cookie",
                "name": "session"
            },
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "description": "Token JWT (futuro)"
            }
        }
    }
}


@api_docs_bp.route("/openapi.json")
def openapi_json():
    """Serve OpenAPI spec as JSON."""
    return jsonify(API_SPEC)


def init_swagger(app):
    """Initialize Flask Swagger UI."""
    if not FLASK_SWAGGER_AVAILABLE:
        app.logger.warning("Flasgger not installed. API docs will not be available.")
        return None
    
    swagger = Swagger(app, template=API_SPEC)
    return swagger
