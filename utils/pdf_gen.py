import re
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors

def create_pdf_report(markdown_text, image_bytes_list=None):
    """
    Generates a PDF buffer from markdown text and a list of image bytes.
    Uses ReportLab to manually construct the document flow.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    
    # Custom Styles
    title_style = styles["Heading1"]
    h2_style = styles["Heading2"]
    h3_style = styles["Heading3"]
    
    # Add Heading4 style if not present
    if 'Heading4' not in styles:
        styles.add(ParagraphStyle(name='Heading4', parent=styles['Heading3'], fontSize=11, spaceBefore=6, spaceAfter=2))
    h4_style = styles["Heading4"]

    normal_style = styles["BodyText"]
    
    # Improved Code Style for Preformatted blocks
    code_style = ParagraphStyle(
        'Code', 
        parent=styles['BodyText'], 
        fontName='Courier', 
        fontSize=8, 
        leading=10,
        backColor=colors.whitesmoke,
        borderPadding=5,
        spaceBefore=5,
        spaceAfter=5
    )

    story = []
    
    # --- 1. Markdown Parsing (Enhanced) ---
    lines = markdown_text.split('\n')
    
    in_code_block = False
    code_buffer = []

    for line in lines:
        stripped_line = line.strip()
        
        # --- Code Block Handling ---
        if stripped_line.startswith('```'):
            if in_code_block:
                # End of block
                block_content = "\n".join(code_buffer)
                story.append(Preformatted(block_content, code_style))
                story.append(Spacer(1, 10))
                in_code_block = False
                code_buffer = []
            else:
                # Start of block
                in_code_block = True
            continue
        
        if in_code_block:
            code_buffer.append(line) # Preserve original indentation
            continue

        # --- Normal Line Handling ---
        if not stripped_line:
            story.append(Spacer(1, 6))
            continue
            
        # Structure Detection
        text_content = stripped_line
        style = normal_style
        bullet = False
        
        if stripped_line.startswith('# '):
            style = title_style
            text_content = stripped_line[2:]
        elif stripped_line.startswith('## '):
            style = h2_style
            text_content = stripped_line[3:]
        elif stripped_line.startswith('### '):
            style = h3_style
            text_content = stripped_line[4:]
        elif stripped_line.startswith('#### '):
            style = h4_style
            text_content = stripped_line[5:]
        elif stripped_line.startswith('- ') or stripped_line.startswith('* '):
            # Bullet point start
            bullet = True
            text_content = stripped_line[2:]
        
        # Inline Formatting (Bold & Italics)
        # Bold: **text** -> <b>text</b>
        text_content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text_content)
        
        # Italics: *text* -> <i>text</i>
        # Use negative lookaround to ensure we don't match occurrences involved in **bold** 
        # (e.g., the middle part of **bold**) or unrelated asterisks
        text_content = re.sub(r'(?<!\*)\*(.*?)\*(?!\*)', r'<i>\1</i>', text_content)

        if bullet:
            # Add bullet character and spacing
            text_content = f"•  {text_content}"
            
        story.append(Paragraph(text_content, style))

    story.append(Spacer(1, 24))

    # --- 2. Add Images ---
    if image_bytes_list:
        story.append(Paragraph("Supporting Visual Evidence", h2_style))
        story.append(Spacer(1, 12))
        
        for img_bytes in image_bytes_list:
            try:
                # Create ReportLab Image from bytes
                img_io = io.BytesIO(img_bytes)
                im = RLImage(img_io)
                
                # Resize to fit width (approx 450-500 pts)
                avail_width = 460
                img_width = im.drawWidth
                img_height = im.drawHeight
                
                if img_width > avail_width:
                    ratio = avail_width / img_width
                    im.drawHeight = img_height * ratio
                    im.drawWidth = avail_width
                
                story.append(im)
                story.append(Spacer(1, 12))
            except Exception as e:
                story.append(Paragraph(f"[Error rendering image: {e}]", code_style))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
