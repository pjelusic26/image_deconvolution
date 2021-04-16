import image_preprocessing
import cv2

# Read color and grayscale image
image_color, image_gray = image_preprocessing.image_read('3d_pokemon.png')

# Save image filters as objects
filter_sharpen = image_preprocessing.image_filter(filter = 'sharpen')
filter_laplacian = image_preprocessing.image_filter(filter = 'laplacian')

# Apply image filters and save processed images
output_sharpen = image_preprocessing.image_convolution(image_gray, filter_sharpen)
output_laplacian = image_preprocessing.image_convolution(image_gray, filter_laplacian)
output_wiener = image_preprocessing.filter_wiener(image_gray)

# Display output images
cv2.imshow("Original", image_gray)
# cv2.imshow(f"Sharpen", output_sharpen)
# cv2.imshow(f"Laplacian", output_laplacian)
cv2.imshow(f"Wiener", output_wiener)

cv2.waitKey(0)
cv2.destroyAllWindows()