import tensorflow as tf 
import matplotlib.pyplot as pyplot


# TENSORFLOW KERAS MODEL IMPLEMENTATION

# define model and training loop

num_epochs = 5


model_tf = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2d((2, 2)),
    tf.keras.layers.Conv2d(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2d((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model_tf.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                 metrics=['accuracy'])
model_tf.fit(train_loader, epochs=num_epochs)
test_loss, test_acc = model_tf.evaluate(test_loader)
print(f'Test Accuracy (TensorFlow): {test_acc * 100:.2f}%')

