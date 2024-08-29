import os
import glob
import h5py
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Flatten
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import KFold
from keras.models import Model, Sequential
from PIL import Image

# ---------------------------------------------   DIRECTORIES    -------------------------------------------------------

DATA_DIR_TRAIN = r''
DATA_DIR_TEST = r''
ModVersion = r''

# ---------------------------------------------    PARAMETERS    -------------------------------------------------------

IM_WIDTH = IM_HEIGHT = 224


ID_FAST_MAP = {0: 'Real', 1: 'Spoof'}
GENDER_ID_MAP = dict((g, i) for i, g in ID_FAST_MAP.items())

ID_FQA_MAP = {0: 'Good', 1: 'Blur', 2: 'Bright', 3: 'Dark'}
RACE_ID_MAP = dict((r, i) for i, r in ID_FQA_MAP.items())


ID_ATTACK_MAP = {0: 'Real', 1: 'Replay', 2: 'Print', 3: 'Mask'}
ATTACK_ID_MAP = dict((gl, i) for i, gl in ID_ATTACK_MAP.items())


# ---------------------------------------------    FUNCTIONS    --------------------------------------------------------


class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_directory, model_base_name):
        self.model_directory = model_directory
        self.model_base_name = model_base_name

    def on_epoch_end(self, epoch, logs=None):
        model_json = self.model.to_json()
        model_json_file_name = f"{self.model_base_name}_epoch_{epoch+1:02d}.json"
        with open(os.path.join(self.model_directory, model_json_file_name), "w") as json_file:
            json_file.write(model_json)

        # Guarda el modelo en formato HDF5
        model_h5_file_name = f"{self.model_base_name}_epoch_{epoch+1:02d}.h5"
        model.save(os.path.join(self.model_directory, model_h5_file_name))

        print(f"Modelo JSON y HDF5 creados para {model_json_file_name} y {model_h5_file_name}")


def parse_filepath(filepath):
    try:
        path, filename = os.path.split(filepath)
        filename, ext = os.path.splitext(filename)
        fast, fqa, attack, _ = filename.split("_")
        return (ID_FAST_MAP[int(fast)], ID_FQA_MAP[int(fqa)],
                ID_ATTACK_MAP[int(attack)]
    except Exception as e:
        # print(filepath)
        return None, None, None, None


def get_data_generator(df, indices, for_training, batch_size=16):
    images, fast, fqa, attack = [], [], [], []
    while True:
        for i in indices:
            if i < 0 or i >= len(df):
                continue

            r = df.iloc[i]
            file = r['file']
            fast, fqa, attack = r['fast_id'], r['fqa_id'], r['attack_id']


            im = Image.open(file)
            im = im.resize((IM_WIDTH, IM_HEIGHT))
            im = np.array(im) / 255.0

            images.append(im)
            fast.append(to_categorical(fast, 2))
            fqa.append(to_categorical(fqa, len(ID_FQA_MAP)))
            attack.append(to_categorical(fqa, len(ID_ATTACK_MAP)))

            if len(images) >= batch_size:
                yield np.array(images), [np.array(fast), np.array(fqa),
                                         np.array(attack)]
                images, fast, fqa, attack = [], [], [], []
        if not for_training:
            break


# ------------------------------------   EXPLORATORY DATA ANALYSIS (EDA)  ----------------------------------------------

files_train = glob.glob(os.path.join(DATA_DIR_TRAIN, "*.jpg"))
attributes_train = list(map(parse_filepath, files_train))
df_train = pd.DataFrame(attributes_train)
df_train['file'] = files_train
df_train.columns = ['fast', 'fqa', 'attack']
df_train = df_train.dropna()
df_train.head()
df_train.describe()

files_test = glob.glob(os.path.join(DATA_DIR_TEST, "*.jpg"))
attributes_test = list(map(parse_filepath, files_test))

df_test = pd.DataFrame(attributes_test)
df_test['file'] = files_test
df_test.columns = ['fast', 'fqa', 'attack']
df_test = df_test.dropna()
df_test.head()
df_test.describe()



# -------------------------------------------    DATA PROCESSING   -----------------------------------------------------

train_idx = np.random.permutation(len(df_train))
test_idx = np.random.permutation(len(df_test))
max_age = 100

# TRAIN
df_train['fast_id'] = df_train['fast'].map(lambda fast: ID_FAST_MAP[fast])
df_train['fqa_id'] = df_train['fqa'].map(lambda fqa: ID_FQA_MAP[fqa])
df_train['attack_id'] = df_train['attack'].map(lambda glass: ID_ATTACK_MAP[attack])

# TEST
df_test['fast_id'] = df_test['fast'].map(lambda fast: ID_FAST_MAP[fast])
df_test['fqa_id'] = df_test['fqa'].map(lambda fqa: ID_FQA_MAP[fqa])
df_test['attack_id'] = df_test['attack'].map(lambda glass: ID_ATTACK_MAP[glass])

print("Data for TRAIN: ", len(train_idx))
print("Data for TEST: ", len(test_idx))
print("Max age: ", max_age)

# -------------------------------------------        MODEL        ------------------------------------------------------

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(units=128, activation='relu')(x)


Fast= Dense(units=len(ID_FAST_MAP), activation='softmax', name='Fast')(x)
Attack = Dense(units=len(ID_ATTACK_MAP), activation='sigmoid', name='Attack')(x)
Fqa = Dense(units=len(ID_FQA_MAP), activation='softmax', name='Fqa')(x)


model = Model(inputs=base_model.input, outputs=[Fast, Attack, Fqa])
model.compile(optimizer='rmsprop',
              loss={
                    'Fast': 'categorical_crossentropy',
                    'Attack': 'categorical_crossentropy',
                    'Fqa': 'categorical_crossentropy'
                    },
              loss_weights={'Fast': 2.,
                            'Attack': 1.5,
                            'Fqa': 1.,
                            },
              metrics={'Fast': 'accuracy',
                       'Attack': 'accuracy',
                       'Fqa': 'accuracy',
                       })

batch_size = 32
valid_batch_size = 32
epochs = 150
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True)
if not os.path.exists(ModVersion):
    os.makedirs(ModVersion)

save_model_callback = SaveModelCallback(model_directory=ModVersion, model_base_name='mobile_net')

fold_index = 0
for train_index, test_index in kf.split(df_train):
    fold_index += 1
    print(f"Fold {fold_index}/{num_folds}")
    print("Train Index:", train_index)
    print("Test Index:", test_index)

    train_data, test_data = df_train.iloc[train_index], df_train.iloc[test_index]
    train_gen = get_data_generator(train_data, train_index, for_training=True, batch_size=batch_size)
    test_gen = get_data_generator(test_data, test_index, for_training=False, batch_size=valid_batch_size)

    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_index) // batch_size,
        epochs=epochs,
        validation_data=test_gen,
        validation_steps=len(test_index) // valid_batch_size,
        callbacks=[save_model_callback]  # Pasa el callback al método fit()
    )

    test_loss, test_accuracy = model.evaluate(test_gen, steps=len(test_index) // valid_batch_size)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

model.save(os.path.join(ModVersion, 'final_model.h5'))