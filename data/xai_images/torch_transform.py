import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),  # Resize image to 256x256
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(10),  # Randomly rotate the image by up to 10 degrees
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(  # Normalize image with mean and standard deviation
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)
