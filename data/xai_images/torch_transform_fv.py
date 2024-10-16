import torchvision.transforms as transforms


# Function to convert RGB to BGR
def to_bgr(img):
    return img[[-1, -2, -3]]


# Important note: make sure that your transforms have resize and normalize!
# Transformation pipeline
transform = transforms.Compose(
    [
        transforms.Resize((112, 112)),  # Resize image to 112x112
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Lambda(to_bgr),  # Convert RGB to BGR
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        ),  # Normalize with BGR values
    ]
)
