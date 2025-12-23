import re
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors

def create_pdf_report(markdown_text, image_dict=None):
    """
    Generates a PDF buffer from markdown text and a dictionary of image bytes.
    Supports inline placement via tags like [[IMAGE:key]].
    """
    if image_dict is None:
        image_dict = {}

    # Copy dict to track used images (so we can append unused ones at the end)
    unused_images = image_dict.copy()

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
    
    # --- Helper to Add Image ---
    def add_image_to_story(img_key):
        if img_key in image_dict:
            try:
                img_bytes = image_dict[img_key]
                img_io = io.BytesIO(img_bytes)
                im = RLImage(img_io)
                
                # Resize to fit width
                avail_width = 460
                img_width = im.drawWidth
                img_height = im.drawHeight
                
                if img_width > avail_width:
                    ratio = avail_width / img_width
                    im.drawHeight = img_height * ratio
                    im.drawWidth = avail_width
                
                story.append(Spacer(1, 6))
                story.append(im)
                story.append(Spacer(1, 6))
                
                # Mark as used
                if img_key in unused_images:
                    del unused_images[img_key]
                    
            except Exception as e:
                story.append(Paragraph(f"[Error rendering image '{img_key}': {e}]", code_style))

    # --- 1. Markdown Parsing (Enhanced with Tags) ---
    lines = markdown_text.split('\n')
    
    in_code_block = False
    code_buffer = []

    for line in lines:
        stripped_line = line.strip()
        
        # --- Check for Image Tag ---
        # Regex for [[IMAGE:key]]
        img_match = re.search(r'\[\[IMAGE:(.*?)\]\]', line)
        if img_match:
            # If there's text before/after, we might want to split, but simplest is to just inject image
            # For now, let's assume the tag is on its own line or end of line.
            # We will effectively replace the tag line with the image.
            key = img_match.group(1).strip()
            add_image_to_story(key)
            continue # Skip adding the text line itself

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
        text_content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text_content)
        text_content = re.sub(r'(?<!\*)\*(.*?)\*(?!\*)', r'<i>\1</i>', text_content)

        if bullet:
            text_content = f"•  {text_content}"
            
        story.append(Paragraph(text_content, style))

    story.append(Spacer(1, 24))

    # --- 2. Add Remaining Images (Fallback) ---
    if unused_images:
        story.append(Paragraph("Appendix: Additional Visuals", h2_style))
        story.append(Spacer(1, 12))
        for key in list(unused_images.keys()):
             # We rely on the order of keys in the dict, which is insertion order in modern Python
             add_image_to_story(key)

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
