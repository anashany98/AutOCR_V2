import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from datetime import datetime

class ProposalManager:
    def __init__(self, logger=None):
        self.logger = logger
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        # Styles matching "Premium Decoration" theme
        self.styles.add(ParagraphStyle(
            name='CoverTitle',
            parent=self.styles['Title'],
            fontSize=32,
            leading=40,
            textColor=colors.HexColor('#D4AF37'), # Gold
            alignment=1, # Center
            spaceAfter=20
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#333333'),
            borderPadding=(0, 0, 10, 0),
            spaceAfter=15
        ))

    def generate_proposal(self, doc_filename, items, output_path):
        """
        Generates a PDF proposal.
        :param doc_filename: Name of the project/document.
        :param items: List of dicts with {'label': str, 'image_path': str, 'price': str}.
        :param output_path: Where to save the PDF.
        """
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=2*cm, leftMargin=2*cm,
                topMargin=2*cm, bottomMargin=2*cm
            )
            
            story = []
            
            # --- COVER PAGE ---
            story.append(Spacer(1, 4*cm))
            story.append(Paragraph("DOSSIER DE DECORACIÓN", self.styles['CoverTitle']))
            story.append(Spacer(1, 1*cm))
            story.append(Paragraph(f"Proyecto: {doc_filename}", self.styles['Heading3']))
            story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", self.styles['Normal']))
            story.append(Spacer(1, 8*cm))
            story.append(Paragraph("Estudio de Interiorismo", self.styles['Normal']))
            story.append(PageBreak())
            
            # --- CONTENT ---
            story.append(Paragraph("Elementos Detectados", self.styles['SectionHeader']))
            story.append(Spacer(1, 0.5*cm))

            # Table Data with Images
            table_data = [['Vista Previa', 'Descripción', 'Precio Est.']]
            
            for item in items:
                img_path = item.get('image_path')
                label = item.get('label', 'Elemento')
                
                # Resize image for thumbnail
                img_obj = None
                if img_path and os.path.exists(img_path):
                    try:
                        img_obj = Image(img_path)
                        img_obj.drawHeight = 3*cm
                        img_obj.drawWidth = 3*cm * (img_obj.imageWidth / img_obj.imageHeight)
                        # Limit max width
                        if img_obj.drawWidth > 4*cm:
                            ratio = 4*cm / img_obj.drawWidth
                            img_obj.drawWidth = 4*cm
                            img_obj.drawHeight = img_obj.drawHeight * ratio
                    except Exception:
                        img_obj = "Img Error"
                else:
                    img_obj = "No Imagen"

                table_data.append([
                    img_obj,
                    Paragraph(f"<b>{label}</b><br/><font size=9 color='#666'>Estilo: Detectado auto</font>", self.styles['Normal']),
                    "Consultar"
                ])

            # Table Style
            t = Table(table_data, colWidths=[5*cm, 8*cm, 3*cm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f0f0f0')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#333333')),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e0e0e0')),
            ]))
            
            story.append(t)
            
            # Build
            doc.build(story)
            if self.logger:
                self.logger.info(f"Proposal generated at {output_path}")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Proposal generation failed: {e}")
            return False
