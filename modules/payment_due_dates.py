"""
Gestión de Vencimientos de Pagos.

Módulo para extraer fechas de vencimiento de facturas,
seguimiento de pagos pendientes y alertas.
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PaymentDue:
    """Representa un pago pendiente."""
    document_id: str
    invoice_number: str
    vendor_name: str
    total_amount: float
    currency: str
    invoice_date: datetime
    due_date: datetime
    days_until_due: int
    payment_status: str


class PaymentDueManager:
    """
    Gestor de fechas de vencimiento de pagos.
    
    Permite:
    - Extraer fecha de vencimiento del OCR
    - Consultar pagos pendientes
    - Alertas de vencimiento
    - Resumen semanal
    """
    
    def __init__(self):
        self._db = None
    
    def _get_db(self):
        """Obtiene la instancia de DBManager."""
        if self._db is None:
            from modules.db_manager import DBManager
            self._db = DBManager.get_instance()
        return self._db
    
    def get_pending_payments(self, tenant_id: str) -> List[PaymentDue]:
        """
        Obtiene todos los pagos pendientes.
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Lista de pagos pendientes
        """
        db = self._get_db()
        
        query = """
            SELECT 
                id,
                invoice_number,
                vendor_name,
                total_amount,
                currency,
                invoice_date,
                due_date,
                payment_status
            FROM documents
            WHERE tenant_id = %s
                AND due_date IS NOT NULL
                AND payment_status IN ('pending', 'partial')
            ORDER BY due_date ASC
        """
        
        payments = []
        
        try:
            results = db.fetch_all(query, (tenant_id,))
            
            for row in results:
                due_date = row[6]
                days_until_due = 0
                
                if due_date:
                    delta = due_date.date() - datetime.now().date()
                    days_until_due = delta.days
                
                payments.append(PaymentDue(
                    document_id=str(row[0]),
                    invoice_number=row[1] or "",
                    vendor_name=row[2] or "",
                    total_amount=float(row[3]) if row[3] else 0,
                    currency=row[4] or "EUR",
                    invoice_date=row[5],
                    due_date=due_date,
                    days_until_due=days_until_due,
                    payment_status=row[7] or "pending"
                ))
        except Exception as e:
            logger.error(f"Error getting pending payments: {e}")
        
        return payments
    
    def get_upcoming_payments(self, tenant_id: str, days: int = 7) -> List[PaymentDue]:
        """
        Obtiene pagos que vencen en los próximos N días.
        
        Args:
            tenant_id: ID del tenant
            days: Número de días hacia adelante
            
        Returns:
            Lista de pagos próximos a vencer
        """
        db = self._get_db()
        
        today = datetime.now().date()
        future_date = today + timedelta(days=days)
        
        query = """
            SELECT 
                id,
                invoice_number,
                vendor_name,
                total_amount,
                currency,
                invoice_date,
                due_date,
                payment_status
            FROM documents
            WHERE tenant_id = %s
                AND due_date IS NOT NULL
                AND payment_status IN ('pending', 'partial')
                AND due_date::date <= %s
                AND due_date::date >= %s
            ORDER BY due_date ASC
        """
        
        payments = []
        
        try:
            results = db.fetch_all(query, (tenant_id, future_date))
            
            for row in results:
                due_date = row[6]
                days_until_due = 0
                
                if due_date:
                    delta = due_date.date() - today
                    days_until_due = delta.days
                
                payments.append(PaymentDue(
                    document_id=str(row[0]),
                    invoice_number=row[1] or "",
                    vendor_name=row[2] or "",
                    total_amount=float(row[3]) if row[3] else 0,
                    currency=row[4] or "EUR",
                    invoice_date=row[5],
                    due_date=due_date,
                    days_until_due=days_until_due,
                    payment_status=row[7] or "pending"
                ))
        except Exception as e:
            logger.error(f"Error getting upcoming payments: {e}")
        
        return payments
    
    def get_overdue_payments(self, tenant_id: str) -> List[PaymentDue]:
        """
        Obtiene pagos vencidos.
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Lista de pagos vencidos
        """
        db = self._get_db()
        
        today = datetime.now().date()
        
        query = """
            SELECT 
                id,
                invoice_number,
                vendor_name,
                total_amount,
                currency,
                invoice_date,
                due_date,
                payment_status
            FROM documents
            WHERE tenant_id = %s
                AND due_date IS NOT NULL
                AND payment_status IN ('pending', 'partial')
                AND due_date::date < %s
            ORDER BY due_date ASC
        """
        
        payments = []
        
        try:
            results = db.fetch_all(query, (tenant_id, today))
            
            for row in results:
                due_date = row[6]
                days_until_due = 0
                
                if due_date:
                    delta = today - due_date.date()
                    days_until_due = -delta.days  # Negativo = vencido
                
                payments.append(PaymentDue(
                    document_id=str(row[0]),
                    invoice_number=row[1] or "",
                    vendor_name=row[2] or "",
                    total_amount=float(row[3]) if row[3] else 0,
                    currency=row[4] or "EUR",
                    invoice_date=row[5],
                    due_date=due_date,
                    days_until_due=days_until_due,
                    payment_status=row[7] or "pending"
                ))
        except Exception as e:
            logger.error(f"Error getting overdue payments: {e}")
        
        return payments
    
    def update_payment_status(self, document_id: str, status: str) -> bool:
        """
        Actualiza el estado de pago de un documento.
        
        Args:
            document_id: ID del documento
            status: Nuevo estado (pending, partial, paid)
            
        Returns:
            True si se actualizó correctamente
        """
        db = self._get_db()
        
        query = """
            UPDATE documents
            SET payment_status = %s, updated_at = NOW()
            WHERE id = %s
        """
        
        try:
            db.execute(query, (status, document_id))
            return True
        except Exception as e:
            logger.error(f"Error updating payment status: {e}")
            return False
    
    def get_weekly_summary(self, tenant_id: str) -> Dict[str, Any]:
        """
        Obtiene el resumen semanal de pagos.
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Diccionario con el resumen
        """
        today = datetime.now().date()
        
        # This week's payments (Mon-Sun)
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)
        
        # Next week's payments
        next_week_start = week_end + timedelta(days=1)
        next_week_end = next_week_start + timedelta(days=6)
        
        this_week = self.get_pending_payments_in_range(tenant_id, week_start, week_end)
        next_week = self.get_pending_payments_in_range(tenant_id, next_week_start, next_week_end)
        overdue = self.get_overdue_payments(tenant_id)
        
        this_week_total = sum(p.total_amount for p in this_week)
        next_week_total = sum(p.total_amount for p in next_week)
        overdue_total = sum(p.total_amount for p in overdue)
        
        return {
            "this_week": {
                "count": len(this_week),
                "total_amount": this_week_total,
                "payments": [self._payment_to_dict(p) for p in this_week]
            },
            "next_week": {
                "count": len(next_week),
                "total_amount": next_week_total,
                "payments": [self._payment_to_dict(p) for p in next_week]
            },
            "overdue": {
                "count": len(overdue),
                "total_amount": overdue_total,
                "payments": [self._payment_to_dict(p) for p in overdue]
            }
        }
    
    def get_pending_payments_in_range(self, tenant_id: str, 
                                     start_date, end_date) -> List[PaymentDue]:
        """Obtiene pagos en un rango de fechas."""
        db = self._get_db()
        
        query = """
            SELECT 
                id,
                invoice_number,
                vendor_name,
                total_amount,
                currency,
                invoice_date,
                due_date,
                payment_status
            FROM documents
            WHERE tenant_id = %s
                AND due_date IS NOT NULL
                AND payment_status IN ('pending', 'partial')
                AND due_date::date >= %s
                AND due_date::date <= %s
            ORDER BY due_date ASC
        """
        
        payments = []
        
        try:
            results = db.fetch_all(query, (tenant_id, start_date, end_date))
            
            for row in results:
                due_date = row[6]
                days_until_due = 0
                
                if due_date:
                    delta = due_date.date() - datetime.now().date()
                    days_until_due = delta.days
                
                payments.append(PaymentDue(
                    document_id=str(row[0]),
                    invoice_number=row[1] or "",
                    vendor_name=row[2] or "",
                    total_amount=float(row[3]) if row[3] else 0,
                    currency=row[4] or "EUR",
                    invoice_date=row[5],
                    due_date=due_date,
                    days_until_due=days_until_due,
                    payment_status=row[7] or "pending"
                ))
        except Exception as e:
            logger.error(f"Error getting payments in range: {e}")
        
        return payments
    
    def _payment_to_dict(self, payment: PaymentDue) -> Dict[str, Any]:
        """Convierte PaymentDue a diccionario."""
        return {
            "document_id": payment.document_id,
            "invoice_number": payment.invoice_number,
            "vendor_name": payment.vendor_name,
            "total_amount": payment.total_amount,
            "currency": payment.currency,
            "invoice_date": payment.invoice_date.isoformat() if payment.invoice_date else None,
            "due_date": payment.due_date.isoformat() if payment.due_date else None,
            "days_until_due": payment.days_until_due,
            "payment_status": payment.payment_status
        }


# Singleton instance
_due_manager: Optional[PaymentDueManager] = None


def get_due_manager() -> PaymentDueManager:
    """Obtiene la instancia singleton del gestor de vencimientos."""
    global _due_manager
    if _due_manager is None:
        _due_manager = PaymentDueManager()
    return _due_manager


# Telegram notification function
async def send_due_date_alerts(tenant_id: str):
    """
    Envía alertas de vencimiento a través de Telegram.
    
    Args:
        tenant_id: ID del tenant
    """
    from modules.telegram_gestores_db import get_telegram_gestores_db
    
    due_manager = get_due_manager()
    gestores_db = get_telegram_gestores_db()
    
    # Get active gestores for this tenant
    gestores = gestores_db.get_active_gestores(tenant_id)
    
    if not gestores:
        logger.info(f"No gestores found for tenant {tenant_id}")
        return
    
    # Get payments due in 7 days and 1 day
    upcoming_7 = due_manager.get_upcoming_payments(tenant_id, 7)
    upcoming_1 = due_manager.get_upcoming_payments(tenant_id, 1)
    overdue = due_manager.get_overdue_payments(tenant_id)
    
    for gestor in gestores:
        if not gestor.notify_expiry:
            continue
        
        message = ""
        
        # Overdue payments
        if overdue:
            message += "⚠️ *PAGOS VENCIDOS*\\n\\n"
            for p in overdue[:5]:  # Max 5
                message += f"• {p.vendor_name}: {p.total_amount:.2f}{p.currency} (venció hace {abs(p.days_until_due)} días)\\n"
            message += "\\n"
        
        # Due in 1 day
        if upcoming_1:
            message += "⏰ *VENCEN MAÑANA*\\n\\n"
            for p in upcoming_1:
                message += f"• {p.vendor_name}: {p.total_amount:.2f}{p.currency}\\n"
            message += "\\n"
        
        # Due in 7 days
        if upcoming_7:
            message += "📅 *VENCEN ESTA SEMANA*\\n\\n"
            for p in upcoming_7[:5]:  # Max 5
                message += f"• {p.vendor_name}: {p.total_amount:.2f}{p.currency} (en {p.days_until_due} días)\\n"
        
        if message:
            try:
                # Send via Telegram bot
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
                logger.error(f"Error sending Telegram alert: {e}")
