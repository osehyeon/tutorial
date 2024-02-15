import tensorflow_datasets as tfds

dataset = tfds.load('imagenet2012', split='validation', as_supervised=True)
