import tensorflow as tf


dataset = tf.data.Dataset.from_tensor_slices(([1.], [1.])).repeat(100).batch(10)

# Create a MirroredStrategy
strategy = tf.distribute.MirroredStrategy()


# ## Set up a multi-worker cluster on the python script
# os.environ['TF_CONFIG'] = json.dumps({
#     'cluster': {
#         'worker': ["worker0.example.com:12345", "worker1.example.com:23456"]
#     },
#     'task': {'type': 'worker', 'index': 0}
# })
# 
# strategy = tf.distribute.MultiWorkerMirroredStrategy()
# with strategy.scope():
#     model = create_model()
#     optimizer = tf.keras.optimizers.Adam()



# Open a strategy scope and build the model
with strategy.scope():
   model = tf.keras.models.Sequential([
   tf.keras.layers.Dense(64, activation='relu'),
   tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model using data parallelism across multiple GPUs
model.fit(dataset, epochs=10)
