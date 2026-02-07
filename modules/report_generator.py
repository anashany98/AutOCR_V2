import os
import logging
from typing import Dict, Any, List
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generates professional PDF reports for design and architecture projects.
    """
    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        self.styles.add(ParagraphStyle(
            name='ProjectTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor("#2c3e50"),
            spaceAfter=20,
            alignment=1 # Center
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor("#3498db"),
            spaceBefore=15,
            spaceAfter=10
        ))

    def generate_project_report(self, project_data: Dict[str, Any], filename: str = None) -> str:
        """
        Creates a PDF report from project data.
        project_data: {
            "title": str,
            "original_image": path,
            "render_image": path,
            "furniture": List[str],
            "advice": str,
            "colors": List[str],
            "details": str
        }
        """
        if not filename:
             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
             filename = f"Report_{timestamp}.pdf"
        
        filepath = os.path.join(self.output_dir, filename)
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        story = []

        # 1. Header
        story.append(Paragraph(project_data.get("title", "Informe de Proyecto AutOCR"), self.styles['ProjectTitle']))
        story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

        # 2. Visual Comparison
        story.append(Paragraph("Visualización del Proyecto", self.styles['SectionHeader']))
        
        img_data = []
        if project_data.get("original_image") and os.path.exists(project_data["original_image"]):
            orig = Image(project_data["original_image"], width=3*inch, height=2.25*inch)
            img_data.append(orig)
        
        if project_data.get("render_image") and os.path.exists(project_data["render_image"]):
            rend = Image(project_data["render_image"], width=3*inch, height=2.25*inch)
            img_data.append(rend)
        
        if img_data:
            t = Table([img_data])
            story.append(t)
            story.append(Paragraph("<center><i>Izquierda: Original | Derecha: Propuesta AI</i></center>", self.styles['Italic']))

        story.append(Spacer(1, 0.3*inch))

        # 3. Furniture & Elements
        story.append(Paragraph("Elementos Detectados", self.styles['SectionHeader']))
        furn_list = project_data.get("furniture", [])
        if furn_list:
            for item in furn_list:
                story.append(Paragraph(f"• {item}", self.styles['Normal']))
        else:
            story.append(Paragraph("No se detectaron elementos específicos.", self.styles['Normal']))

        # 4. AI Advice
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Asesoría Técnica y Estética", self.styles['SectionHeader']))
        advice = project_data.get("advice", "Crea espacios equilibrados combinando texturas naturales.")
        story.append(Paragraph(advice, self.styles['Normal']))

        # 5. Color Palette
        palette = project_data.get("colors", [])
        if palette:
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph("Paleta de Colores Extraída", self.styles['SectionHeader']))
            color_boxes = []
            for c in palette:
                # Expecting hex or rgb
                try:
                    color_obj = colors.HexColor(c) if c.startswith('#') else colors.black
                    color_boxes.append(Table([['']], colWidths=[0.5*inch], rowHeights=[0.5*inch], 
                                            style=[('BACKGROUND', (0,0), (-1,-1), color_obj)]))
                except: continue
            
            if color_boxes:
                story.append(Table([color_boxes]))

        # Finalize
        try:
            doc.build(story)
            logger.info(f"Report generated: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to build PDF: {e}")
            return ""
