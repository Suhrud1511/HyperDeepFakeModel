# pip commands: !pip install numpy opencv-python-headless scikit-learn tensorflow matplotlib keras-tuner 



import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import TimeDistributed, GlobalAveragePooling2D
from tensorflow.keras.utils import Sequence
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband
import matplotlib.pyplot as plt
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Custom data generator class
class VideoDataGenerator(Sequence):
    """Custom data generator for loading videos and labels."""
    def __init__(self, video_paths, labels, batch_size=8, frames_per_video=30, frame_size=(64, 64), augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.frames_per_video = frames_per_video
        self.frame_size = frame_size
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_video_paths = [self.video_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]

        X, y = self.__data_generation(batch_video_paths, batch_labels)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.video_paths))
        np.random.shuffle(self.indexes)

    def __data_generation(self, batch_video_paths, batch_labels):
        X = np.empty((self.batch_size, self.frames_per_video, self.frame_size[0], self.frame_size[1], 3), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        for i, video_path in enumerate(batch_video_paths):
            frames = []
            cap = cv2.VideoCapture(video_path)
            while len(frames) < self.frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
                frame = frame / 255.0
                if self.augment:
                    frame = augment_frame(frame)
                frames.append(frame)
            cap.release()

            if len(frames) == self.frames_per_video:
                X[i,] = frames
                y[i] = batch_labels[i]

        return X, y

def augment_frame(frame):
   
    if np.random.rand() < 0.5:
        frame = np.fliplr(frame)
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((frame.shape[1] // 2, frame.shape[0] // 2), angle, 1)
    frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    return frame

# HyperModel class for creating and tuning the model
class DeepfakeHyperModel(HyperModel):
    def __init__(self, input_shape, frames_per_video):
        self.input_shape = input_shape
        self.frames_per_video = frames_per_video

    def build(self, hp):
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=self.input_shape)
        base_model.trainable = False

        inputs = layers.Input(shape=(self.frames_per_video,) + self.input_shape)
        x = TimeDistributed(base_model)(inputs)
        x = TimeDistributed(GlobalAveragePooling2D())(x)

        for i in range(hp.Int('conv_layers', 1, 3)):
            x = layers.Conv1D(filters=hp.Int(f'conv_{i}_filters', 32, 128, step=32),
                              kernel_size=hp.Int(f'conv_{i}_kernel_size', 3, 5),
                              activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_reg', 1e-5, 1e-2, sampling='log')))(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(2)(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu')(x)
        x = layers.Dropout(hp.Float('dropout', 0.3, 0.7, step=0.1))(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                      loss='binary_crossentropy', metrics=['accuracy'])

        return model

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Load video paths and labels
real_videos_path = '/content/drive/MyDrive/Deepfake_detection/Copy of Celeb-DF.zip (Unzipped Files)/Celeb-real'
fake_videos_path = '/content/drive/MyDrive/Deepfake_detection/Copy of Celeb-DF.zip (Unzipped Files)/Celeb-synthesis'

real_video_files = [os.path.join(real_videos_path, file) for file in os.listdir(real_videos_path)]
fake_video_files = [os.path.join(fake_videos_path, file) for file in os.listdir(fake_videos_path)]

video_paths = real_video_files + fake_video_files
labels = [0] * len(real_video_files) + [1] * len(fake_video_files)

X_train_paths, X_val_paths, y_train, y_val = train_test_split(video_paths, labels, test_size=0.2, random_state=42)

train_generator = VideoDataGenerator(X_train_paths, y_train, batch_size=8, augment=True)
val_generator = VideoDataGenerator(X_val_paths, y_val, batch_size=8)

# Define input shape based on frames
input_shape = (64, 64, 3)
frames_per_video = 30

# Create the hypermodel
hypermodel = DeepfakeHyperModel(input_shape=input_shape, frames_per_video=frames_per_video)

# Initialize the Hyperband tuner
tuner = Hyperband(hypermodel,
                  objective='val_accuracy',
                  max_epochs=10,  # Reduced epochs for initial testing
                  factor=3,
                  directory='hyperband_dir',
                  project_name='deepfake_detection')

# Search for the best hyperparameters
tuner.search(train_generator, validation_data=val_generator, epochs=10)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the optimal hyperparameters
model = hypermodel.build(best_hps)
model.summary()

# Train the model with the optimal hyperparameters
history = model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)])

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_generator)
print(f'Validation Accuracy: {val_accuracy:.2f}')

# Save the best model and weights
model.save('/content/drive/MyDrive/Deepfake_detection/best_model.h5')
model.save_weights('/content/drive/MyDrive/Deepfake_detection/best_model_weights.h5')

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualize some validation images with predictions
num_visualize = 5
fig, axs = plt.subplots(1, num_visualize, figsize=(15, 3))

for i in range(num_visualize):
    index = np.random.randint(0, len(X_val_paths))
    video_path = X_val_paths[index]
    label = y_val[index]

    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < frames_per_video:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (input_shape[1], input_shape[0]))
        frame = frame / 255.0
        frames.append(frame)
    cap.release()

    if len(frames) == frames_per_video:
        video = np.array(frames)
        prediction = model.predict(np.expand_dims(video, axis=0))[0, 0]

        axs[i].imshow(video[0])
        axs[i].axis('off')
        axs[i].set_title(f'Label: {label}\nPrediction: {prediction:.2f}')

plt.tight_layout()
plt.show()
