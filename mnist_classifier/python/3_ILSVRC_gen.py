import tensorflow_datasets as tfds

dataset = tfds.load('imagenet2012', split='test', as_supervised=True)
