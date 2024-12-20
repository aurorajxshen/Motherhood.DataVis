from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

# Create a blank canvas
def create_canvas(width, height, background_color):
    return Image.new("RGB", (width, height), background_color)

# Draw text bubbles
def draw_text_bubbles(draw, text_list, positions, bubble_color, text_color, font):
    for text, pos in zip(text_list, positions):
        # Draw a rounded rectangle as a bubble
        x, y = pos
        text_size = font.getsize(text)
        padding = 10
        bubble_rect = [(x, y), (x + text_size[0] + 2 * padding, y + text_size[1] + 2 * padding)]
        draw.rounded_rectangle(bubble_rect, radius=20, fill=bubble_color)
        draw.text((x + padding, y + padding), text, fill=text_color, font=font)

# Main function to assemble the structure
def create_visual_structure():
    # Canvas size
    width, height = 800, 600
    canvas = create_canvas(width, height, background_color="darkblue")

    # Load fonts (ensure path to font file is correct)
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"  # Replace with an available font
    font = ImageFont.truetype(font_path, 20)
    
    draw = ImageDraw.Draw(canvas)
    
    # Central placeholder object
    central_text = "NEW JERSEY\nLICENSE PLATE\n2F354H\n06-30-23"
    center_pos = (width // 2 - 80, height // 2 - 60)
    draw.rectangle([center_pos, (center_pos[0] + 200, center_pos[1] + 120)], fill="white", outline="black", width=3)
    draw.text((center_pos[0] + 10, center_pos[1] + 10), central_text, fill="black", font=font)

    # Draw text bubbles
    text_list = [
        "We have a delivery. DM me",
        "Kareem gets it",
        "What’s the address?",
        "New job: 899 Carroll Street",
        "Send me the PDF",
        "Customer says he needs it ASAP"
    ]
    positions = [(50, 50), (650, 50), (100, 200), (600, 250), (50, 400), (600, 450)]
    draw_text_bubbles(draw, text_list, positions, bubble_color="lightblue", text_color="black", font=font)

    # Load placeholder for person image
    person_img_path = "person_placeholder.png"  # Replace with actual image
    try:
        person_img = Image.open(person_img_path).resize((200, 300))
        canvas.paste(person_img, (width // 2 - 100, height // 2 - 150), person_img)
    except FileNotFoundError:
        draw.rectangle([width // 2 - 100, height // 2 - 150, width // 2 + 100, height // 2 + 150],
                       fill="gray", outline="white", width=3)
        draw.text((width // 2 - 80, height // 2 - 30), "Person\nImage", fill="white", font=font)

    # Show final image
    plt.imshow(np.asarray(canvas))
    plt.axis('off')
    plt.show()

# Execute
create_visual_structure()
