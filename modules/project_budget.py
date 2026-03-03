"""
Gestión de Proyectos y Presupuestos.

Módulo para vincular documentos a proyectos, seguimiento de presupuestos,
y alertas de gasto.
"""
import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProjectBudget:
    """Presupuesto de un proyecto."""
    project_id: str
    project_name: str
    budget_amount: float
    budget_currency: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    alert_threshold_percent: int
    spent_amount: float
    document_count: int


class ProjectBudgetManager:
    """
    Gestor de presupuestos por proyecto.
    
    Permite:
    - Vincular documentos a proyectos
    - Consultar gasto real vs presupuesto
    - Alertas al superar umbral
    """
    
    def __init__(self):
        self._db = None
    
    def _get_db(self):
        """Obtiene la instancia de DBManager."""
        if self._db is None:
            from modules.db_manager import DBManager
            self._db = DBManager.get_instance()
        return self._db
    
    def link_document_to_project(self, document_id: str, project_id: str) -> bool:
        """
        Vincula un documento a un proyecto.
        
        Args:
            document_id: ID del documento
            project_id: ID del proyecto
            
        Returns:
            True si se vinculó correctamente
        """
        db = self._get_db()
        
        query = """
            UPDATE documents
            SET project_id = %s, updated_at = NOW()
            WHERE id = %s
        """
        
        try:
            db.execute(query, (project_id, document_id))
            return True
        except Exception as e:
            logger.error(f"Error linking document to project: {e}")
            return False
    
    def get_project_budget(self, project_id: str) -> Optional[ProjectBudget]:
        """
        Obtiene el presupuesto de un proyecto.
        
        Args:
            project_id: ID del proyecto
            
        Returns:
            ProjectBudget con los datos del proyecto
        """
        db = self._get_db()
        
        # Query to get project with spending calculation
        query = """
            SELECT 
                p.id as project_id,
                p.name as project_name,
                p.budget_amount,
                p.budget_currency,
                p.start_date,
                p.end_date,
                p.alert_threshold_percent,
                COALESCE(SUM(d.total_amount), 0) as spent_amount,
                COUNT(d.id) as document_count
            FROM projects p
            LEFT JOIN documents d ON d.project_id = p.id AND d.status = 'completed'
            WHERE p.id = %s
            GROUP BY p.id
        """
        
        try:
            result = db.fetch_one(query, (project_id,))
            
            if result:
                return ProjectBudget(
                    project_id=str(result[0]),
                    project_name=result[1] or "",
                    budget_amount=float(result[2]) if result[2] else 0,
                    budget_currency=result[3] or "EUR",
                    start_date=result[4],
                    end_date=result[5],
                    alert_threshold_percent=int(result[6]) if result[6] else 80,
                    spent_amount=float(result[7]) if result[7] else 0,
                    document_count=int(result[8]) if result[8] else 0
                )
        except Exception as e:
            logger.error(f"Error getting project budget: {e}")
        
        return None
    
    def get_projects_budget_summary(self, tenant_id: str) -> List[ProjectBudget]:
        """
        Obtiene el resumen de presupuestos de todos los proyectos.
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Lista de ProjectBudget
        """
        db = self._get_db()
        
        query = """
            SELECT 
                p.id as project_id,
                p.name as project_name,
                p.budget_amount,
                p.budget_currency,
                p.start_date,
                p.end_date,
                p.alert_threshold_percent,
                COALESCE(SUM(d.total_amount), 0) as spent_amount,
                COUNT(d.id) as document_count
            FROM projects p
            LEFT JOIN documents d ON d.project_id = p.id AND d.status = 'completed'
            WHERE p.tenant_id = %s
            GROUP BY p.id
            ORDER BY p.name
        """
        
        budgets = []
        
        try:
            results = db.fetch_all(query, (tenant_id,))
            
            for result in results:
                budgets.append(ProjectBudget(
                    project_id=str(result[0]),
                    project_name=result[1] or "",
                    budget_amount=float(result[2]) if result[2] else 0,
                    budget_currency=result[3] or "EUR",
                    start_date=result[4],
                    end_date=result[5],
                    alert_threshold_percent=int(result[6]) if result[6] else 80,
                    spent_amount=float(result[7]) if result[7] else 0,
                    document_count=int(result[8]) if result[8] else 0
                ))
        except Exception as e:
            logger.error(f"Error getting projects budget summary: {e}")
        
        return budgets
    
    def check_budget_alerts(self, tenant_id: str) -> List[Dict[str, Any]]:
        """
        Verifica proyectos que han superado el umbral de presupuesto.
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Lista de alertas de presupuesto
        """
        projects = self.get_projects_budget_summary(tenant_id)
        alerts = []
        
        for project in projects:
            if project.budget_amount > 0:
                percent_used = (project.spent_amount / project.budget_amount) * 100
                
                if percent_used >= project.alert_threshold_percent:
                    alerts.append({
                        "project_id": project.project_id,
                        "project_name": project.project_name,
                        "budget_amount": project.budget_amount,
                        "spent_amount": project.spent_amount,
                        "percent_used": round(percent_used, 2),
                        "threshold_percent": project.alert_threshold_percent,
                        "is_over_budget": percent_used > 100,
                        "currency": project.budget_currency
                    })
        
        return alerts
    
    def get_spending_by_vendor(self, tenant_id: str, 
                              start_date: datetime = None,
                              end_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Obtiene el gasto agrupado por proveedor.
        
        Args:
            tenant_id: ID del tenant
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            
        Returns:
            Lista de gasto por proveedor
        """
        db = self._get_db()
        
        query = """
            SELECT 
                vendor_name,
                vendor_nif,
                COUNT(*) as document_count,
                SUM(total_amount) as total_spent,
                MAX(created_at) as last_invoice_date
            FROM documents
            WHERE tenant_id = %s
                AND status = 'completed'
                AND total_amount IS NOT NULL
        """
        
        params = [tenant_id]
        
        if start_date:
            query += " AND created_at >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND created_at <= %s"
            params.append(end_date)
        
        query += " GROUP BY vendor_name, vendor_nif ORDER BY total_spent DESC"
        
        results = []
        
        try:
            db_results = db.fetch_all(query, tuple(params))
            
            for row in db_results:
                results.append({
                    "vendor_name": row[0] or "Sin proveedor",
                    "vendor_nif": row[1],
                    "document_count": row[2],
                    "total_spent": float(row[3]) if row[3] else 0,
                    "last_invoice_date": row[4].isoformat() if row[4] else None
                })
        except Exception as e:
            logger.error(f"Error getting spending by vendor: {e}")
        
        return results
    
    def get_spending_by_month(self, tenant_id: str, 
                             year: int = None) -> List[Dict[str, Any]]:
        """
        Obtiene el gasto agrupado por mes.
        
        Args:
            tenant_id: ID del tenant
            year: Año (opcional, usa el actual si no se especifica)
            
        Returns:
            Lista de gasto por mes
        """
        db = self._get_db()
        
        if year is None:
            year = datetime.now().year
        
        query = """
            SELECT 
                EXTRACT(MONTH FROM created_at) as month,
                COUNT(*) as document_count,
                SUM(total_amount) as total_spent
            FROM documents
            WHERE tenant_id = %s
                AND status = 'completed'
                AND total_amount IS NOT NULL
                AND EXTRACT(YEAR FROM created_at) = %s
            GROUP BY EXTRACT(MONTH FROM created_at)
            ORDER BY month
        """
        
        results = []
        
        try:
            db_results = db.fetch_all(query, (tenant_id, year))
            
            for row in db_results:
                results.append({
                    "month": int(row[0]),
                    "document_count": row[1],
                    "total_spent": float(row[2]) if row[2] else 0
                })
        except Exception as e:
            logger.error(f"Error getting spending by month: {e}")
        
        return results
    
    def get_spending_by_category(self, tenant_id: str) -> List[Dict[str, Any]]:
        """
        Obtiene el gasto agrupado por categoría/tipo de documento.
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Lista de gasto por categoría
        """
        db = self._get_db()
        
        query = """
            SELECT 
                doc_type,
                COUNT(*) as document_count,
                SUM(total_amount) as total_spent
            FROM documents
            WHERE tenant_id = %s
                AND status = 'completed'
                AND total_amount IS NOT NULL
            GROUP BY doc_type
            ORDER BY total_spent DESC
        """
        
        results = []
        
        try:
            db_results = db.fetch_all(query, (tenant_id,))
            
            for row in db_results:
                results.append({
                    "category": row[0] or "Sin categoría",
                    "document_count": row[1],
                    "total_spent": float(row[2]) if row[2] else 0
                })
        except Exception as e:
            logger.error(f"Error getting spending by category: {e}")
        
        return results
    
    def update_project_budget(self, project_id: str, budget_amount: float,
                            currency: str = "EUR", 
                            alert_threshold: int = 80,
                            start_date: datetime = None,
                            end_date: datetime = None) -> bool:
        """
        Actualiza el presupuesto de un proyecto.
        
        Args:
            project_id: ID del proyecto
            budget_amount: Monto del presupuesto
            currency: Moneda
            alert_threshold: Porcentaje de alerta
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            True si se actualizó correctamente
        """
        db = self._get_db()
        
        query = """
            UPDATE projects
            SET budget_amount = %s,
                budget_currency = %s,
                alert_threshold_percent = %s,
                start_date = %s,
                end_date = %s,
                updated_at = NOW()
            WHERE id = %s
        """
        
        try:
            db.execute(query, (
                budget_amount, currency, alert_threshold,
                start_date, end_date, project_id
            ))
            return True
        except Exception as e:
            logger.error(f"Error updating project budget: {e}")
            return False


# Singleton instance
_budget_manager: Optional[ProjectBudgetManager] = None


def get_budget_manager() -> ProjectBudgetManager:
    """Obtiene la instancia singleton del gestor de presupuestos."""
    global _budget_manager
    if _budget_manager is None:
        _budget_manager = ProjectBudgetManager()
    return _budget_manager
