from PIL import Image
import os

from google.colab import drive
drive.mount('/content/drive')

# Define the paths
input_folder = '/content/drive/MyDrive/dataset/train'
output_image_folder = '/content/drive/MyDrive/dataset/output/images'
output_mask_folder = '/content/drive/MyDrive/dataset/output/masks'

# Create directories if they don't exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# Function to separate the image and mask from a side-by-side combined image
def separate_image_and_mask(input_path, image_output_path, mask_output_path):
    # Load the combined image
    combined_image = Image.open(input_path)
    width, height = combined_image.size

    # Split the image vertically
    # Image on the left, mask on the right
    image_part = combined_image.crop((0, 0, width // 2, height))
    mask_part = combined_image.crop((width // 2, 0, width, height))

    # Save the separated image and mask
    image_part.save(image_output_path)
    mask_part.save(mask_output_path)

# Process each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):  # adjust based on file format
        input_path = os.path.join(input_folder, filename)

        # Define output paths for separated image and mask
        image_output_path = os.path.join(output_image_folder, f'image_{filename}')
        mask_output_path = os.path.join(output_mask_folder, f'mask_{filename}')

        # Call the function to separate and save the images
        separate_image_and_mask(input_path, image_output_path, mask_output_path)


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Image dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1  # Grayscale

def preprocess_image_mask(image_path, mask_path):
    # Load and resize the grayscale image and mask
    image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale")
    mask = load_img(mask_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale")

    # Convert to arrays and normalize to [0,1]
    image = img_to_array(image) / 255.0
    mask = img_to_array(mask) / 255.0

    return image, mask

# Data augmentation for grayscale images
data_gen_args = dict(rotation_range=30,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

def augment_data(image, mask):
    seed = np.random.randint(100)
    image_gen = image_datagen.flow(np.expand_dims(image, axis=0), seed=seed)
    mask_gen = mask_datagen.flow(np.expand_dims(mask, axis=0), seed=seed)

    return next(image_gen)[0], next(mask_gen)[0]

# Generator for creating training batches
def data_generator(image_folder, mask_folder, batch_size=8):
    image_files = sorted(os.listdir(image_folder))
    mask_files = sorted(os.listdir(mask_folder))

    while True:
        batch_images = []
        batch_masks = []

        for i in range(batch_size):
            idx = np.random.randint(len(image_files))
            image_path = os.path.join(image_folder, image_files[idx])
            mask_path = os.path.join(mask_folder, mask_files[idx])

            image, mask = preprocess_image_mask(image_path, mask_path)
            image, mask = augment_data(image, mask)

            batch_images.append(image)
            batch_masks.append(mask)

        yield np.array(batch_images), np.array(batch_masks)

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.models import Model

def unet_generator(input_shape=(256, 256, 1)):
    inputs = Input(shape=input_shape)

    # Downsampling layers with increased filter sizes
    down1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(inputs)
    down1 = LeakyReLU()(down1)

    down2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(down1)
    down2 = BatchNormalization()(down2)
    down2 = LeakyReLU()(down2)

    down3 = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(down2)
    down3 = BatchNormalization()(down3)
    down3 = LeakyReLU()(down3)

    down4 = Conv2D(512, (4, 4), strides=(2, 2), padding='same')(down3)
    down4 = BatchNormalization()(down4)
    down4 = LeakyReLU()(down4)

    bottleneck = Conv2D(1024, (4, 4), strides=(2, 2), padding='same')(down4)
    bottleneck = BatchNormalization()(bottleneck)
    bottleneck = LeakyReLU()(bottleneck)

    # Upsampling layers with increased filter sizes
    up1 = Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same')(bottleneck)
    up1 = BatchNormalization()(up1)
    up1 = Dropout(0.5)(up1)
    up1 = Concatenate()([up1, down4])

    up2 = Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same')(up1)
    up2 = BatchNormalization()(up2)
    up2 = Dropout(0.5)(up2)
    up2 = Concatenate()([up2, down3])

    up3 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(up2)
    up3 = BatchNormalization()(up3)
    up3 = Dropout(0.5)(up3)
    up3 = Concatenate()([up3, down2])

    up4 = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(up3)
    up4 = BatchNormalization()(up4)
    up4 = Dropout(0.5)(up4)
    up4 = Concatenate()([up4, down1])

    outputs = Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', activation='sigmoid')(up4)

    return Model(inputs, outputs)


def patchgan_discriminator(input_shape=(IMG_HEIGHT, IMG_WIDTH, 2)):  # Two channels
    inputs = Input(shape=input_shape)

    d1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(inputs)
    d1 = LeakyReLU()(d1)

    d2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = LeakyReLU()(d2)

    d3 = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = LeakyReLU()(d3)

    d4 = Conv2D(512, (4, 4), strides=(1, 1), padding='same')(d3)
    d4 = BatchNormalization()(d4)
    d4 = LeakyReLU()(d4)

    outputs = Conv2D(1, (4, 4), strides=(1, 1), padding='same')(d4)

    return Model(inputs, outputs)

# Loss weights
LAMBDA_RECONSTRUCTION = 10
LAMBDA_SPARSE = 0.1

def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def generator_loss(disc_generated_output, gen_output, target, lambda_bce=1.0, lambda_dice=1.0):
    # Binary Cross-Entropy loss (for adversarial training)
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_generated_output), disc_generated_output)
    
    # Reconstruction loss (mean absolute error)
    reconstruction_loss = tf.reduce_mean(tf.abs(target - gen_output))

    # Dice Loss
    dice = dice_loss(target, gen_output)

    # Combine losses
    return lambda_bce * bce_loss + lambda_dice * dice + reconstruction_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Create the generator and discriminator instances outside the train_step function
generator = unet_generator()
discriminator = patchgan_discriminator()

@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Now call the generator and discriminator instances
        generated_mask = generator(input_image, training=True)

        # Concatenate input_image with target and generated_mask for discriminator
        disc_real_output = discriminator(tf.concat([input_image, target], axis=-1), training=True)
        disc_generated_output = discriminator(tf.concat([input_image, generated_mask], axis=-1), training=True)

        gen_loss = generator_loss(disc_generated_output, generated_mask, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Training loop
EPOCHS = 50
STEPS_PER_EPOCH = 100

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for step in range(STEPS_PER_EPOCH):
        input_image, target = next(data_generator(output_image_folder, output_mask_folder, batch_size=8))
        gen_loss, disc_loss = train_step(input_image, target)

        if step % 10 == 0:
            print(f"Step {step}, Gen Loss: {gen_loss.numpy():.4f}, Disc Loss: {disc_loss.numpy():.4f}")
print("Training complete.")

def separate_image_and_mask_test(input_path, img_height=256, img_width=256):
    combined_image = Image.open(input_path)
    width, height = combined_image.size

    # Check if images are side-by-side (left-right)
    if width == img_width * 2 and height == img_height:
        # Split horizontally into left (image) and right (mask)
        image_part = combined_image.crop((0, 0, width // 2, height))
        mask_part = combined_image.crop((width // 2, 0, width, height))
    elif height == img_height * 2 and width == img_width:  # Top-bottom stacked images
        # Split vertically into top (image) and bottom (mask)
        image_part = combined_image.crop((0, 0, width, height // 2))
        mask_part = combined_image.crop((0, height // 2, width, height))
    else:
        raise ValueError("Unexpected dimensions for combined image and mask.")

    return image_part, mask_part

def separate_and_save_all(input_folder, image_output_folder, mask_output_folder):
    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(mask_output_folder, exist_ok=True)
    
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.png') or filename.endswith('.jpg'):  # or your specific image format
            input_path = os.path.join(input_folder, filename)
            image_part, mask_part = separate_image_and_mask_test(input_path)

            # Save the separated image and mask
            image_part.save(os.path.join(image_output_folder, filename))
            mask_part.save(os.path.join(mask_output_folder, filename))

# Define the paths
test_input_folder = '/content/drive/MyDrive/dataset/test'
test_output_image_folder = '/content/drive/MyDrive/dataset/output_test/images'
test_output_mask_folder = '/content/drive/MyDrive/dataset/output_test/masks'

val_input_folder = '/content/drive/MyDrive/dataset/val'
val_output_image_folder = '/content/drive/MyDrive/dataset/output_val/images'
val_output_mask_folder = '/content/drive/MyDrive/dataset/output_val/masks'
# Separate and save images for validation and test sets
separate_and_save_all(test_input_folder, test_output_image_folder, test_output_mask_folder)
separate_and_save_all(val_input_folder, val_output_image_folder, val_output_mask_folder)


def load_data(image_folder, mask_folder, img_height=256, img_width=256):
    images = []
    masks = []
    image_files = sorted(os.listdir(image_folder))
    mask_files = sorted(os.listdir(mask_folder))

    for img_file, mask_file in zip(image_files, mask_files):
        image_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(mask_folder, mask_file)
        
        # Preprocess each image and mask
        image, mask = preprocess_image_mask(image_path, mask_path)
        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Load validation and test data
val_images, val_masks = load_data(val_output_image_folder, val_output_mask_folder)
test_images, test_masks = load_data(test_output_image_folder, test_output_mask_folder)


def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def calculate_iou(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def calculate_metrics(y_true, y_pred):
    dice_scores = []
    iou_scores = []
    
    for true, pred in zip(y_true, y_pred):
        dice = dice_coefficient(true, pred)
        iou = calculate_iou(true, pred)
        dice_scores.append(dice)
        iou_scores.append(iou)
        
    return np.mean(dice_scores), np.mean(iou_scores)

# Predict masks for validation and test images
val_predictions = generator.predict(val_images)
test_predictions = generator.predict(test_images)

# Calculate metrics for validation and test sets
val_dice, val_iou = calculate_metrics(val_masks, val_predictions)
test_dice, test_iou = calculate_metrics(test_masks, test_predictions)

print(f"Validation Dice Coefficient: {val_dice:.4f}, IoU: {val_iou:.4f}")
print(f"Test Dice Coefficient: {test_dice:.4f}, IoU: {test_iou:.4f}")


import matplotlib.pyplot as plt

def display_predictions(images, true_masks, predicted_masks, num_samples=3):
    plt.figure(figsize=(15, num_samples * 5))
    
    for i in range(num_samples):
        idx = np.random.randint(0, len(images))
        
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.title("Original Image")
        plt.imshow(images[idx].squeeze(), cmap='gray')
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.title("Ground Truth Mask")
        plt.imshow(true_masks[idx].squeeze(), cmap='gray')
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.title("Predicted Mask")
        plt.imshow(predicted_masks[idx].squeeze(), cmap='gray')
        plt.axis('off')
    
    plt.show()

# Display predictions for validation set
display_predictions(val_images, val_masks, val_predictions)

# Display predictions for test set
display_predictions(test_images, test_masks, test_predictions)

