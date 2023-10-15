
# %%
import numpy as np
import librosa.display, os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)
    
def create_pngs_from_wavs(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        print('input_file',input_file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))
        create_spectrogram(input_file, output_file)

# %%
"""
Create PNG files containing spectrograms from all the WAV files in the "Sounds/background" directory.
"""

# %%
#create_pngs_from_wavs('Sounds/Laryngozele', 'Spectrograms/Laryngozele')

# %%
"""
Create PNG files containing spectrograms from all the WAV files in the "Sounds/chainsaw" directory.
"""

# %%
#create_pngs_from_wavs('Sounds/Normal', 'Spectrograms/Normal')

# %%
"""
Create PNG files containing spectrograms from all the WAV files in the "Sounds/engine" directory.
"""

# %%
#create_pngs_from_wavs('Sounds/Vox senilis', 'Spectrograms/Vox senilis')

# %%
"""
Create PNG files containing spectrograms from all the WAV files in the "Sounds/storm" directory.
"""



# %%
"""
Define two new helper functions for loading and displaying spectrograms and declare two Python lists — one to store spectrogram images, and another to store class labels.
"""

# %%
from keras.preprocessing import image

def load_images_from_path(path, label):
    images = []
    labels = []

    for file in os.listdir(path):
        images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
        labels.append((label))
        
    return images, labels

def show_images(images):
    fig, axes = plt.subplots(1, 8, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)
        
x = []
y = []

# %%
"""
Load the background spectrogram images, add them to the list named `x`, and label them with 0s.
"""

# %%
images, labels = load_images_from_path('Spectrograms/Laryngozele', 0)
show_images(images)
    
x += images
y += labels

# %%
"""
Load the chainsaw spectrogram images, add them to the list named `x`, and label them with 1s.
"""

# %%
images, labels = load_images_from_path('Spectrograms/Normal', 1)
show_images(images)
    
x += images
y += labels

# %%
"""
Load the engine spectrogram images, add them to the list named `x`, and label them with 2s.
"""

# %%
images, labels = load_images_from_path('Spectrograms/Vox senilis', 2)
show_images(images)
    
x += images
y += labels


# %%
"""
Split the images and labels into two datasets — one for training, and one for testing. 
"""

# %%
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=0)

x_train_norm = np.array(x_train) / 255
x_test_norm = np.array(x_test) / 255

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# %%
"""
## Build and train a CNN
"""

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %%
"""
Train the CNN and save the `history` object returned by `fit` in a local variable.
"""

# %%
hist = model.fit(x_train_norm, y_train_encoded, validation_data=(x_test_norm, y_test_encoded), batch_size=10, epochs=10)

# %%
"""
Plot the training and validation accuracy.
"""

# %%
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training Accuracy')
plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
#plt.plot()
plt.show()


# %%
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x_train_norm = preprocess_input(np.array(x_train))
x_test_norm = preprocess_input(np.array(x_test))

train_features = base_model.predict(x_train_norm)
test_features = base_model.predict(x_test_norm)

# %%
"""
Define a neural network to classify features extracted by `MobileNetV2`.
"""

# %%
model = Sequential()
model.add(Flatten(input_shape=train_features.shape[1:]))
model.add(Dense(1024, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()

y_predicted = model.predict(test_features)
mat = confusion_matrix(y_test_encoded.argmax(axis=1), y_predicted.argmax(axis=1))
class_labels = ['Laryngozele', 'Normal', 'Vox senilis']

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)

plt.xlabel('Predicted label')
plt.ylabel('Actual label')

# %%

# %%
"""
## Test with unrelated WAV files"""

# %%
create_spectrogram('Sounds/Laryngozele/1205-a_h-egg.wav', 'Spectrograms/sample1.png')

x = image.load_img('Spectrograms/sample1.png', target_size=(224, 224))
plt.xticks([])
plt.yticks([])
plt.imshow(x)

# %%
"""
Preprocess the spectrogram image, pass it to `MobileNetV2` for feature extraction, and classify the features.
"""

# %%
x = image.img_to_array(x)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

y = base_model.predict(x)
predictions = model.predict(y)

for i, label in enumerate(class_labels):
    print(f'{label}: {predictions[0][i]}')

# %%

