import numpy as np
import matplotlib.pyplot as plt


def create_triangular_qr(message, size=50):
    # Convert message to binary
    binary = ''.join(format(ord(c), '08b') for c in message)

    # Calculate the height of the equilateral triangle
    height = int(size * np.sqrt(3) / 2)

    # Create a blank canvas
    canvas = np.ones((height, size, 4), dtype=np.float32)  # RGBA
    canvas[:, :, 3] = 0  # Set alpha channel to transparent

    # Calculate the spacing between dots
    dot_spacing = size / (size - 1)

    # Create the triangular pattern with fine dots
    for i in range(size):
        y = int(i * np.sqrt(3) / 2 * dot_spacing)
        for j in range(size - i):
            x = int(i / 2 * dot_spacing + j * dot_spacing)

            if len(binary) > 0:
                color = (0, 0, 0, 1) if binary[0] == '1' else (1, 1, 1, 1)
                binary = binary[1:]
            else:
                color = (0, 0, 0, 1) if np.random.randint(0, 2) else (1, 1, 1, 1)

            canvas[y - 1:y + 2, x - 1:x + 2] = color

    return canvas


def display_triangular_qr(canvas):
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas)
    plt.axis('off')
    plt.show()


# Example usage
message = "Hello, World!"
qr_code = create_triangular_qr(message, size=50)
display_triangular_qr(qr_code)