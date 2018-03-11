# Tensorflow-DatasetAPI
Simple Tensorflow DatasetAPI Tutorial for reading image

## Usage

### 1. `glob` images

```python
ex) trainA_dataset = glob('./dataset/{}/*.*'.format(dataset_name + '/trainA'))

trainA_dataset = ['./dataset/cat/trainA/a.jpg', 
                  './dataset/cat/trainA/b.png', 
                  './dataset/cat/trainA/c.jpeg', 
                  ...]
```

***

### 2. Use `from_tensor_slices`
```python
trainA = tf.data.Dataset.from_tensor_slices(trainA_dataset)
```

***

### 3. Use `map` for preprocessing
```python

    def image_processing(filename):
        x = tf.read_file(filename) # file read 
        x_decode = tf.image.decode_jpeg(x, channels=3) # for RGB

        # DO NOT USE decode_image
        # will be error

        img = tf.image.resize_images(x_decode, [256, 256])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        return img
        

trainA = trainA.map(image_processing, num_parallel_calls=8)
```

* If you want `data augmentation` too...
```python

class ImageData:

    def __init__(self, batch_size, load_size, channels, augment_flag):
        self.batch_size = batch_size
        self.load_size = load_size
        self.channels = channels
        self.augment_flag = augment_flag
        self.augment_size = load_size + (30 if load_size == 256 else 15)

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        
        # DO NOT USE decode_image
        # will be error
        
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag :
            p = random.random()
            if p > 0.5:
                img = self.augmentation(img)

        return img
        
    def augmentation(self, image):
        seed = random.randint(0, 2 ** 31 - 1)
    
        ori_image_shape = tf.shape(image)
        image = tf.image.random_flip_left_right(image, seed=seed)
        image = tf.image.resize_images(image, [self.augment_size, self.augment_size])
        image = tf.random_crop(image, ori_image_shape, seed=seed)
    
        return image
    
    
Image_Data_Class = ImageData(batch_size, img_size, img_ch, augment_flag)
trainA = trainA.map(Image_Data_Class.image_processing, num_parallel_calls=8)

```

* Personally recommend `num_parallel_calls` = `4 or 8`

***

### 4. Set `prefetch` & `batch_size`
```python

trainA = trainA.shuffle(buffer_size=10000).prefetch(buffer_size=batch_size).batch(batch_size).repeat()

```

* Personally recommend `prefetch_size` = `batch_size` or `small size`
* if `shuffle_size` is greater than the number of elements in the dataset, you get a uniform shuffle
* if `shuffle_size` is  1 then you get no shuffling at all.
* If the number of elements `N` in this dataset is not an exact multiple of batch_size, the final batch contain smaller tensors with shape `N % batch_size` in the batch dimension. 
* If your program depends on the batches having the same shape, consider using the `tf.contrib.data.batch_and_drop_remainder `transformation instead.

```python
trainA = trainA.shuffle(10000).prefetch(batch_size).apply(batch_and_drop_remainder(batch_size)).repeat()
```

***

### 5. Set `Iterator`
```python

trainA_iterator = trainA.make_initializable_iterator()

# DO NOT USE make_one_shot_iterator

trainA_init_op = trainA_iterator.initializer

data_A = trainA_iterator.get_next()
loss = network(data_A)
...

```

***

### 6. Run `Init operation`
```python

def train() :
    trainA_init_op.run() or sess.run(trainA_init_op)

    for epoch ...
        for iteration ...

```

***

### 7. See `Code`
* [Unsupervised Image to Image Translation Networks for DatasetAPI](https://github.com/taki0112/UNIT-Tensorflow)


## Author
Junho Kim
