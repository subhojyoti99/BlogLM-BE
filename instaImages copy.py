import base64
from pathlib import Path
import re
from typing import Dict

from IPython.display import Image, display
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI


import google.generativeai as genai
import os

def generate_instagram_carousel(blog_content: str) -> str:
    """
    Converts blog content into a 5-slide Instagram carousel post and caption.
    """
    prompt = f"""
    You are an expert social media manager specializing in creating viral-worthy Instagram carousels.
    Your task is to convert the following blog content into a 5-slide informational carousel post and an accompanying caption. The tone should be engaging, educational, and easy to read on a mobile screen.

    ---
    BLOG CONTENT:
    {blog_content}
    ---

    **Instructions:**
    1.  **Distill the Core Message**: Identify the single most important message or journey in the blog.
    2.  **Structure the Carousel**: Create content for 5 distinct slides. Each slide's text must be concise and visually scannable.
    3.  **Write the Caption**: Create a separate, compelling caption that summarizes the carousel, asks a question, and includes a call-to-action and hashtags.

    **Output Format (Strictly follow this structure):**

    **Slide 1: Title/Hook**
    A powerful, scroll-stopping title or question. (Max 10-12 words)

    **Slide 2: The Problem / The "Before"**
    Briefly introduce the core problem or concept. Use 1-2 short sentences or a few bullet points.

    **Slide 3: The Solution / The "How-To"**
    Explain the main solution or provide the key steps. Use bullet points (•) for clarity. Keep it concise.

    **Slide 4: Key Takeaway / The "After"**
    Summarize the most important takeaway or the benefit of the solution. Make it impactful.

    **Slide 5: Call to Action (CTA)**
    Encourage engagement directly on the image. Use prompts like "Save for later," "Comment your thoughts below," or "Share this with a friend!"

    **Instagram Caption:**
    [Start with a hook (can be the same as Slide 1). Add 2-3 sentences providing context. Ask an engaging question to spark conversation. Finish with a CTA like "Read the full blog - link in bio!" and 5-7 relevant hashtags.]

    ---
    Return ONLY the formatted text for the slides and caption, with no extra explanations or introductory phrases.
    """

    try:
        # Using a model that's good at following structured instructions
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Instagram carousel generation failed: {e}")
        # Provide a structured fallback so the downstream process doesn't fail
        fallback_text = """**Slide 1: Title/Hook**
        A Quick Guide to Our Latest Topic!

        **Slide 2: The Problem / The "Before"**
        Discover the key challenge we're tackling in our new blog post.

        **Slide 3: The Solution / The "How-To"**
        • We break down the solution step-by-step.
        • Simple, actionable advice.

        **Slide 4: Key Takeaway / The "After"**
        Unlock a powerful new perspective.

        **Slide 5: Call to Action (CTA)**
        👇 Save this post & check the link in bio!

        **Instagram Caption:**
        Our new blog post is live! 🚀 We're breaking down a complex topic into easy-to-understand points. Swipe left to get the key insights!

        What are your thoughts on this? Let us know in the comments! 👇

        #NewPost #BlogUpdate #Learning #Insights #InstaGuide #Carousel
        """
        return fallback_text

    
from PIL import Image, ImageDraw, ImageFont
import textwrap
import uuid
import base64
import io

def generate_slide_image(content: str, slide_number: int) -> str:
    """
    Generate an image for an Instagram slide and return as base64
    """
    try:
        # Create image with Instagram dimensions (1080x1080 for square posts)
        width, height = 1080, 1080
        image = Image.new('RGB', (width, height), color='#FFFFFF')  # White background
        draw = ImageDraw.Draw(image)
        
        # Try to use a nice font (fallback to default if not available)
        try:
            font_large = ImageFont.truetype("arial.ttf", 60)
            font_medium = ImageFont.truetype("arial.ttf", 40)
            font_small = ImageFont.truetype("arial.ttf", 30)
        except:
            try:
                font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
                font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
                font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
                font_small = ImageFont.load_default()
        
        # Extract title and content
        lines = content.split('\n')
        title = ""
        body_content = []
        
        for line in lines:
            if line.strip() and not title:
                title = line.replace('**', '').replace('*', '').strip()
            elif line.strip():
                body_content.append(line.replace('**', '').replace('*', '').strip())
        
        # If no title found, use first meaningful line
        if not title and body_content:
            title = body_content[0]
            body_content = body_content[1:]
        
        # Draw title
        title_lines = textwrap.wrap(title, width=25)
        title_y = 100
        
        for line in title_lines:
            bbox = draw.textbbox((0, 0), line, font=font_large)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) // 2
            draw.text((x, title_y), line, fill='#000000', font=font_large)
            title_y += 70
        
        # Draw content
        content_y = title_y + 50
        for line in body_content:
            if line.strip():
                content_lines = textwrap.wrap(line, width=35)
                for content_line in content_lines:
                    bbox = draw.textbbox((0, 0), content_line, font=font_medium)
                    text_width = bbox[2] - bbox[0]
                    x = (width - text_width) // 2
                    draw.text((x, content_y), content_line, fill='#333333', font=font_medium)
                    content_y += 50
        
        # Add slide number at bottom
        slide_text = f"Slide {slide_number}"
        bbox = draw.textbbox((0, 0), slide_text, font=font_small)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, height - 80), slide_text, fill='#666666', font=font_small)
        
        # Convert to base64 instead of saving to file
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        data_url = f"data:image/png;base64,{image_base64}"
        
        print(f"✅ Generated slide image as base64")
        return data_url
        
    except Exception as e:
        print(f"❌ Error generating slide image: {e}")
        # Return a placeholder base64 image instead of filename
        return create_placeholder_base64_image(slide_number)



def extract_slides(carousel_text: str) -> Dict[str, str]:
    """
    Extract slides from carousel text with better parsing
    """
    slides = {}
    
    # Split by slide indicators
    slide_sections = re.split(r'\*\*Slide \d+:', carousel_text)
    
    for i, section in enumerate(slide_sections[1:], 1):  # Skip first empty section
        lines = section.strip().split('\n')
        if lines:
            title = lines[0].strip()
            content = '\n'.join(lines[1:]).strip()
            slides[title] = content
    
    # If the above method didn't work, try alternative parsing
    if not slides:
        # Try parsing by numbered slides
        slide_pattern = r'\*\*Slide (\d+): ([^*]+)(.*?)(?=\*\*Slide \d+:|$)'
        matches = re.findall(slide_pattern, carousel_text, re.DOTALL)
        
        for match in matches:
            slide_num, title, content = match
            slides[title.strip()] = content.strip()
    
    # If still no slides, create simple slides from the text
    if not slides:
        lines = carousel_text.split('\n')
        current_slide = 1
        current_content = []
        
        for line in lines:
            if line.strip() and not line.startswith('**Instagram Caption:**'):
                current_content.append(line.strip())
                # Create a new slide every 3-4 lines
                if len(current_content) >= 4:
                    slides[f"Slide {current_slide}"] = '\n'.join(current_content)
                    current_slide += 1
                    current_content = []
        
        # Add remaining content
        if current_content:
            slides[f"Slide {current_slide}"] = '\n'.join(current_content)
    
    return slides

def create_placeholder_base64_image(slide_number: int) -> str:
    """Create a placeholder image as base64 when image generation fails"""
    try:
        width, height = 1080, 1080
        image = Image.new('RGB', (width, height), color='#f0f0f0')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        text = f"Slide {slide_number}\n(Image generation failed)"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), text, fill='#666666', font=font)
        
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        print(f"❌ Error creating placeholder image: {e}")
        # Return a simple colored rectangle as fallback
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
