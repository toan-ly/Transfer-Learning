import tensorflow as tf
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import random


mean_subtract_val = [[[[103.939, 116.779, 123.68]]]]

### Display
def display_results(train_ds, val_ds, model=None, miscTxt=''):
    
    # Display train set
    fig = plt.figure(figsize=(25,20))
    plt.suptitle('Train set ' + miscTxt)
    for i, dat in enumerate(train_ds.take(10)):
        # Get image
        im = dat[0]
        label = dat[1]
        
        # Only for training
        im = data_augmentation( im )
    
        plt.subplot(4, 5, i+1)
        # plt.imshow( np.squeeze( np.uint8(im - np.min(im)) ) )
        plt.imshow( np.squeeze( np.uint8(im + mean_subtract_val)))
        
        if model != None:
            predLabel = model.predict( im )
            
            if np.round(predLabel[0][0]) == label[0][0].numpy():
                col = 'green'
            else:
                col = 'red'
            plt.title( np.array_str(label[0].numpy(), precision=2) + \
                      np.array_str(predLabel[0], precision=2), color=col)
        else:
            plt.title( str(label.numpy()) )
        
    

    # Display val set
    for i, dat in enumerate(val_ds.take(20)):
        if i < 10:
            continue
        # Get image
        im = dat[0] 
        label = dat[1]
        
        plt.subplot(4, 5, i+11-10)
        # plt.imshow( np.squeeze( np.uint8(im - np.min(im)) ) )
        plt.imshow( np.squeeze( np.uint8(im + mean_subtract_val)))
        
        if model != None:
            predLabel = model.predict( im )
            
            if np.round(predLabel[0][0]) == label[0][0].numpy():
                col = 'green'
            else:
                col = 'red'
            plt.title( np.array_str(label[0].numpy(), precision=2) + \
                      np.array_str(predLabel[0], precision=2), color=col)
        else:
            plt.title( str(label.numpy()) )
        
    fig.text(0.5, 0.5, 'Val set', horizontalalignment='center')
    
    plt.show()
    
    
    
    
# Data augmentation
data_augmentation = tf.keras.Sequential(
    [
         tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
         tf.keras.layers.experimental.preprocessing.RandomContrast(0.2),
     ]
)



### NET
# Import Full Net
# net_orig = tf.keras.applications.VGG16()


# Load pre-trained model without classifier
net_no_top = tf.keras.applications.VGG16( include_top=False, weights='imagenet' )
net_no_top.trainable = False # freeze base model

# Build transfer-learning model
inputs = tf.keras.layers.Input( shape=(224, 224, 3) )
net = data_augmentation( inputs )
net = net_no_top(net) # frozen base model
net = tf.keras.layers.GlobalAveragePooling2D() (net)
outputs = tf.keras.layers.Dense(2, activation='softmax') (net)
model = tf.keras.Model( inputs, outputs )

# Model parameters
learn_rate = 1e-4
optim = tf.keras.optimizers.Adam(learning_rate=learn_rate)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy')]

model.compile(optimizer=optim, loss=loss, metrics=metrics )

model.summary()


### DATA
# Load data 
data_folder = r'/Users/toanne/Desktop/Riverain Tech/TransferLearning/Data'
data_folder2 = r'/Users/toanne/Desktop/Riverain Tech/TransferLearning/Data'


train_files = glob.glob(data_folder2 + r'/Train/*.mat') 
train_files_to_shuffle = glob.glob(data_folder + r'/Train/*.mat')

random.shuffle( train_files_to_shuffle ) # shuffle in place

train_files = train_files + train_files_to_shuffle

print(*train_files[:20], sep='\n')

num_train_files = len(train_files)
print('Num train files:', num_train_files)

val_files = glob.glob(data_folder2 + r'/Val/*.mat')
num_val_files = len(val_files)
print('Num val files:', num_val_files)

# Initialize data matrix
train_mat = np.zeros( (num_train_files, 224, 224, 3), dtype='float32' )
train_label = np.zeros( (num_train_files, 2), dtype='uint8' )

val_mat = np.zeros( (num_val_files, 224, 224, 3), dtype='float32' )
val_label = np.zeros( (num_val_files, 2), dtype='uint8' )


for i, file in enumerate(train_files):
    dat = sio.loadmat( file )
    train_mat[i, :, :, :] = dat['im']
    train_label[i, :] = tf.one_hot( dat['label'], 2 )
    
for i, file in enumerate(val_files):
    dat = sio.loadmat( file )
    val_mat[i, :, :, :] = dat['im']
    val_label[i, :] = tf.one_hot( dat['label'], 2 )

train_ds = tf.data.Dataset.from_tensor_slices( (train_mat, train_label) )
train_ds = train_ds.batch(1)

val_ds = tf.data.Dataset.from_tensor_slices( (val_mat, val_label) )
val_ds = val_ds.batch(1)


### Display once
display_results( train_ds, val_ds )

    
### Train Params
epoch_n = 200
monitoring = 'val_loss'
monitor_mode = 'min'
addiFmt = '-{accuracy:.3E}-{val_accuracy:.3E}'
out_folder = r'/Users/toanne/Desktop/Riverain Tech/TransferLearning/Model'


### Define callbacks 

# Early stopping
earlystopper = tf.keras.callbacks.EarlyStopping(patience=30, verbose=1, monitor=monitoring, mode=monitor_mode)

# Checkpointer to save model
cur_date = time.strftime("%Y-%m-%d", time.localtime(time.time()))
checkpointer = tf.keras.callbacks.ModelCheckpoint(out_folder+'/'+cur_date+'-TF'+tf.__version__\
                                +'-Net-CP{epoch:03d}-{loss:.3E}-{val_loss:.3E}'+addiFmt+'.h5',
                                verbose=1, 
                                save_best_only=True,
                                monitor=monitoring, mode=monitor_mode)
    
# Tensorboard logs
tboard_logs = out_folder+'/logs'
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = tboard_logs,
                                              histogram_freq = 1,
                                              profile_batch = (2,5) #profiles batch 2-5
                                              )

# Display Calback
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        
        display_results( train_ds, val_ds, model=self.model, miscTxt='Epoch '+str(epoch+1) )

        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))



### Check and Train
val = input("Proceed? (y): ")
if val != 'y':
    print('Quitting...')

else:
    #--------------------------------------------------------------------------
    # Train!
    results = model.fit(
        train_ds, 
        validation_data = val_ds, 
        epochs = epoch_n, 
        callbacks = [earlystopper, 
                      checkpointer, 
                      tboard_callback, 
                      DisplayCallback()],
        workers=12, use_multiprocessing=True)

