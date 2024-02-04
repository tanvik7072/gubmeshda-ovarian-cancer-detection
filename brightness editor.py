from PIL import Image, ImageEnhance 
import random 
def adjust_random_brightness(image_path, output_path): 
    """ 
    Adjust the brightness of an image to a random factor between 0.5 and 1.5 and save the result. 
    Parameters: 
        -image_path: The file path of the input image. 
        - output_path: The file path for the output image. 
        """ 
        # Load the image from the given path 
image = Image.open(image_path) 
# Generate a random brightness factor between 0.5 and 1.5 
random_factor = random.uniform(0.5, 1.5) 
print(f"Adjusting brightness by a factor of {random_factor}") 

# Create a brightness enhancer and apply the random factor 
enhancer = ImageEnhance.Brightness(image) 
enhanced_image = enhancer.enhance(random_factor) 
# Save the enhanced image to the specified output path 
enhanced_image.save(output_path) 
print(f"Enhanced image saved to {output_path}") 
# Example usage: 
image_path = 'your_input_image.jpg' # Update this to your image's file path 
output_path = 'your_enhanced_image.jpg' # Update this to your desired output file path 
adjust_random_brightness(image_path, output_path)
