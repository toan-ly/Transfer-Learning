# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 2024

@author: Toan Ly
"""

import tensorflow as tf
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import random
import sys

class DataGenerator:
    def __init__(self, train_files, val_files):
        self.train_files = train_files
        self.val_files = val_files

    def load_data(self, files):
        data = np.zeros((len(files), 224, 224, 3), dtype='float32')
        labels = np.zeros((len(files), 2), dtype='uint8')
        for i, file in enumerate(files):
            dat = sio.loadmat(file)
            data[i, :, :, :] = dat['im']
            labels[i, :] = tf.one_hot(dat['label'], 2)
        return data, labels

    def generate_dataset(self, files):
        data, labels = self.load_data(files)
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        return dataset.batch(1)

    def generate_train_dataset(self):
        train_files_to_shuffle = self.train_files[:]
        random.shuffle(train_files_to_shuffle)
        train_files = self.train_files + train_files_to_shuffle
        return self.generate_dataset(train_files)

    def generate_val_dataset(self):
        return self.generate_dataset(self.val_files)

        
class CustomModel:
    def __init__(self):
        self.mean_subtract_val = [[[[103.939, 116.779, 123.68]]]]
        self.data_aug = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomContrast(0.2),
        ])
        self.net_no_top = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        self.net_no_top.trainable = False # Freeze base model

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        net = self.data_aug(inputs)
        net = self.net_no_top(net)
        net = tf.keras.layers.GlobalAveragePooling2D()(net)
        outputs = tf.keras.layers.Dense(2, activation='softmax')(net)
        model = tf.keras.Model(inputs, outputs)
        return model

class Trainer:
    def __init__(self, model, train_ds, val_ds):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds

    def compile_model(self, lr=1e-4):
        optim = tf.keras.optimizers.Adam(learning_rate=lr)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy')]
        self.model.compile(optimizer=optim, loss=loss, metrics=metrics)

    def display_results(self, miscTxt=''):
        # Display train set
        fig = plt.figure(figsize=(25, 20))
        plt.suptitle('Train set ' + miscTxt)
        for i, dat in enumerate(self.train_ds.take(10)):
            # Get image
            im, label = dat[0], dat[1]

            # Only for training
            im = self.model.data_aug(im)

            plt.subplot(4, 5, i+1)
            plt.imshow(np.squeeze(np.uint8(im + self.model.mean_subtract_val)))

            if self.model is not None:
                pred_label = self.model.predict(im)
                if np.round(pred_label[0][0]) == label[0][0].numpy():
                    col = 'green'
                else:
                    col = 'red'
                plt.title(np.array_str(label[0].numpy(), precision=2)
                          + np.array_str(pred_label[0], precision=2),
                          color=col
                          )
            else:
                plt.title(str(label.numpy()))

        # Display val set
        for i, dat in enumerate(self.val_ds.take(20)):
            if i < 10:
                continue

            # Get image
            im = dat[0]
            label = dat[1]

            plt.subplot(4, 5, i+11-10)
            plt.imshow(np.squeeze(np.uint8(im + self.model.mean_subtract_val)))
            if self.model is not None:
                pred_label = self.model.predict(im)
                if np.round(pred_label[0][0]) == label[0][0].numpy():
                    col = 'green'
                else:
                    col = 'red'
                plt.title(np.array_str(label[0].numpy(), precision=2) 
                          + np.array_str(pred_label[0], precision=2),
                          color=col
                          )
            else:
                plt.title(str(label.numpy()))

            
        fig.text(0.5, 0.5, 'Val set', horizontalalignment='center')
        plt.show()
    
    def train(self, epochs, callbacks=[]):
        results = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks,
            workers=12,
            use_multiprocessing=True
        )
        return results

def main():
    data_folder = sys.path[0] + '/Data'
    train_files = glob.glob(data_folder + '/Train/*.mat')
    val_files = glob.glob(data_folder + '/Val/*.mat')

    data_generator = DataGenerator(train_files, val_files)
    train_ds = data_generator.generate_train_dataset()
    val_ds = data_generator.generate_val_dataset()

    custom_model = CustomModel()
    model = custom_model.build_model()

    trainer = Trainer(model, train_ds, val_ds)
    trainer.compile_model()

    # Define callbacks

    # Early stopping
    earlystopper = tf.keras.callbacks.EarlyStopping(patience=30, 
                                                    verbose=1,
                                                    monitor='val_loss',
                                                    mode='min'
                                                    )
    cur_date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    out_folder = sys.path[0] + '/Model'

    # Checkpointer to save model
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        out_folder+'/'+cur_date+'-TF'+tf.__version__+'-Net-CP{epoch:03d}-{loss:.3E}-{val_loss:.3E}'
        +'{accuracy:.3E}-{val_accuracy:.3E}.h5',
        verbose=1,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    # Tensorboard logs
    tboard_logs = out_folder + '/logs'
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tboard_logs,
                                                     higtogram_freq=1,
                                                     profile_batch=(2,5)
                                                     )
    
    class DisplayCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            trainer.display_results(miscTxt='Epoch '+str(epoch+1))
            print('\nSample Prediction after epoch {}\n'.format(epoch+1))

    
    val = input('Proceed? (y): ')
    if val != 'y':
        print('Quitting...')
    else:
        trainer.train(epochs=200, callbacks=[earlystopper, checkpointer, tboard_callback, DisplayCallback()])
    
if __name__ == '__main__':
    main()

            
        