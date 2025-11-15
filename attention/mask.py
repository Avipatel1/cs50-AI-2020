import sys
import tensorflow as tf
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Load TF model — MUST use from_pt=True
    model = TFBertForMaskedLM.from_pretrained(MODEL, from_pt=True)

    # Forward pass
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Return index of the [MASK] token.
    """
    input_ids = inputs["input_ids"][0].numpy()
    for i, token_id in enumerate(input_ids):
        if token_id == mask_token_id:
            return i
    return None


def get_color_for_attention_score(attention_score):
    """
    Convert attention score to grayscale color tuple (R, G, B).
    - attention_score ∈ [0, 1]
    """
    shade = int(attention_score * 255)
    return (shade, shade, shade)


def visualize_attentions(tokens, attentions):
    """
    Produce diagrams for every layer and every attention head.
    """
    for layer_index, layer_attentions in enumerate(attentions):
        # layer_attentions shape: (batch=1, num_heads, seq, seq)
        for head_index, head_attentions in enumerate(layer_attentions[0]):
            generate_diagram(
                layer_index + 1,
                head_index + 1,
                tokens,
                head_attentions.numpy()
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Draw a NxN grayscale attention map + axis labels.
    """
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw token labels
    for i, token in enumerate(tokens):

        # Vertical token text (rotated)
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90, expand=True)
        img.paste(token_image, mask=token_image)

        # Horizontal token text
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw grid cells
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            attention_val = float(attention_weights[i][j])
            color = get_color_for_attention_score(attention_val)

            x = PIXELS_PER_WORD + j * GRID_SIZE
            y = PIXELS_PER_WORD + i * GRID_SIZE
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    img.save(f"images/Attention_Layer{layer_number}_Head{head_number}.png")
    

if __name__ == "__main__":
    main()
