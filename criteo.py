import sys
import datetime

import tensorflow as tf
from absl import app

train_filename = "C:\\Users\\brand\\Downloads\\Criteo_x1\\train.csv"
val_filename = "C:\\Users\\brand\\Downloads\\Criteo_x1\\valid.csv"
test_filename = "C:\\Users\\brand\\Downloads\\Criteo_x1\\Criteo_x1"

learning_rate = 0.0002282433105027466
hidden_layer_dims = [768, 256, 128]
BATCH_SIZE = 512

num_train_steps = 150000
num_eval_steps = 8634

LABEL_FEATURE = "label"
LABEL_FEATURE_TYPE = tf.float32

NUMERIC_FEATURES = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13"]
NUMERIC_FEATURE_TYPES = [tf.float32] * len(NUMERIC_FEATURES)

CATEGORICAL_FEATURES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15",
                       "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"]
CATEGORICAL_FEATURE_TYPES = [tf.int32] * len(CATEGORICAL_FEATURES)
CATEGORICAL_FEATURE_EMBEDDING_DIMENSION = 20
NUM_BINS = 10000


def get_dataset(file_pattern, is_test):
   dataset = tf.data.experimental.make_csv_dataset(
       file_pattern=file_pattern,
       batch_size=BATCH_SIZE,
       column_names=[LABEL_FEATURE] + NUMERIC_FEATURES + CATEGORICAL_FEATURES,
       column_defaults=[LABEL_FEATURE_TYPE] + NUMERIC_FEATURE_TYPES + CATEGORICAL_FEATURE_TYPES,
       label_name=LABEL_FEATURE,
       header=True,
       num_epochs=1 if is_test else None,
       shuffle=True,
       num_parallel_reads=16)
   return dataset


class M(tf.keras.Model):

   def __init__(self, layer_dims, layer_activations):
       super(M, self).__init__()

       self.Es = {}
       self.Hs = {}

       for name in CATEGORICAL_FEATURES:
           self.Hs[name] = tf.keras.layers.Hashing(NUM_BINS)
           self.Es[name] = tf.keras.layers.Embedding(
               NUM_BINS, CATEGORICAL_FEATURE_EMBEDDING_DIMENSION)

       self.Ls = []
       for dim, activation in zip(layer_dims, layer_activations):
           self.Ls += [tf.keras.layers.Dense(dim, activation=activation)]

   @tf.function
   def call(self, inputs):
       outputs = []
       for name in NUMERIC_FEATURES:
           output = tf.reshape(inputs[name], [-1, 1])
           outputs.append(output)
       for name in CATEGORICAL_FEATURES:
           output = self.Hs[name](inputs[name])
           output = self.Es[name](output)
           outputs.append(output)
       outputs = tf.keras.layers.concatenate(outputs)
       for L in self.Ls:
           outputs = L(outputs)
       return tf.math.sigmoid(outputs)

def main(argv):
   del argv

   train_dataset = get_dataset(train_filename, False)
   # print("DATASET", next(train_dataset.take(1).as_numpy_iterator()))
   # return

   val_dataset = get_dataset(val_filename, False)
   test_dataset = get_dataset(test_filename, True)

   model = M(
       layer_dims=hidden_layer_dims + [1],
       layer_activations=["relu"] * len(hidden_layer_dims) + [None],
   )
   optimizer = tf.keras.optimizers.legacy.Adam(
       learning_rate=learning_rate, clipnorm=100
   )
   model.compile(
       optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
   )

   log_dir = "/tmp/crite_small/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
   print('log_dir', log_dir)
   tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
                                                         write_graph=True,
                                                         write_images=True,
                                                         update_freq='epoch',
                                                         profile_batch=2,
                                                         embeddings_freq=1)

   history = model.fit(
       x=train_dataset,
       epochs=150,
       steps_per_epoch=1000,
       verbose=2,
       # validation_data=eval_dataset,
       # validation_steps=100,
       callbacks=[tensorboard_callback]
   )

   results = model.evaluate(test_dataset)
   print('results', results)


if __name__ == "__main__":
   app.run(main)
