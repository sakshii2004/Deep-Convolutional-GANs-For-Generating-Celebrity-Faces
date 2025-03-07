# Deep Convolutional Generative Adversarial Network (DCGAN)

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) based on the paper **"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"** by Radford et al. (2015). The model is trained on the **CelebA Faces Dataset** to generate realistic human faces.

## Dataset Preprocessing
### 1. **Dataset Used**
- The **CelebA** dataset is used, which contains aligned and cropped celebrity faces.
- The dataset is stored in `/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/`.

### 2. **Preprocessing Steps**
- Images are resized to **64x64 pixels**.
- Center cropping is applied.
- Images are converted to tensors and normalized to **[-1, 1]**.

```python
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

### 3. **Custom Dataset Loader**
Since the dataset does not have class folders, a **custom dataset class** is used to load images from the directory:

```python
class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # No labels in CelebA, so return a dummy label
```

## Model Architecture
The DCGAN consists of two networks:
1. **Generator**: Converts a random noise vector into a realistic image.
2. **Discriminator**: Classifies images as real or fake.

### **Generator**
- Uses **transposed convolutions** to upsample noise into an image.
- **Batch normalization** is applied for stability.
- Uses **ReLU** activation except for the output, which uses **Tanh**.

### **Discriminator**
- Uses **strided convolutions** to downsample images.
- **LeakyReLU (0.2)** is used in all layers except the output.
- The final layer applies a **Sigmoid** activation to classify images.

## How to Train the Model
### **Prerequisites**
Ensure you have Python and the required libraries installed:
```bash
pip install torch torchvision numpy matplotlib
```

### **Training the Model**
Run the training script:
```python
python train.py
```
This will:
- Train the DCGAN model for 10 epochs.
- Save generated images in `fake_images_epoch_X.png`.
- Save the trained model weights.

### **Saving and Loading Models**
To save the trained models:
```python
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
```
To load them later:
```python
generator.load_state_dict(torch.load("generator.pth"))
discriminator.load_state_dict(torch.load("discriminator.pth"))
```

## Expected Outputs
- **Early epochs**: Generated images may appear noisy and unrealistic.
- **Later epochs**: Faces start forming with clearer features.
- **Final results**: The generator produces realistic celebrity faces.

## Deployment
1. Upload the project to a GitHub repository.
2. Include the dataset preprocessing, training script, and saved models.
3. Share the repository link in your submission.

---

## Example Output (Generated Faces)
Check the `fake_images_epoch_X.png` files in the output directory for sample images.

