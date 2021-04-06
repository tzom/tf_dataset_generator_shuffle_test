import tensorflow as tf
import os
import numpy as np

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.keras.backend.set_floatx('float32')

n = 20
def odd_generator(n):
    some_numbers = range(0,n,2)
    for i in some_numbers:
        yield [1.],[i+1]

def even_generator(n):
    some_numbers = range(0,n,2)
    for i in some_numbers:
        yield [0.],[i]



training = True
buffer_size = n
batch_size = 4
steps_per_epoch = int(n/batch_size)

x = tf.data.Dataset.from_generator(odd_generator,args=[n],output_types=(tf.float32,tf.float32),output_shapes=((1),(1)))#.repeat(1)
y = tf.data.Dataset.from_generator(even_generator,args=[n],output_types=(tf.float32,tf.float32),output_shapes=((1),(1)))#.repeat(1)

ds = tf.compat.v1.data.experimental.sample_from_datasets([x,y]) 

if training:
    #pass
    #ds = ds.repeat(int(buffer_size/batch_size))
    ds = ds.shuffle(buffer_size=buffer_size)
#ds = ds.map(lambda y,x: tuple((int(y),int(x))),num_parallel_calls=AUTOTUNE)
ds = ds.batch(batch_size,drop_remainder=True) 

ds = ds.repeat(steps_per_epoch)
ds = ds.prefetch(buffer_size=AUTOTUNE)

#########################################################
#########################################################

x = tf.data.Dataset.from_generator(odd_generator,args=[n],output_types=(tf.float32,tf.float32),output_shapes=((1),(1)))#.repeat(1)
y = tf.data.Dataset.from_generator(even_generator,args=[n],output_types=(tf.float32,tf.float32),output_shapes=((1),(1)))#.repeat(1)

#v_ds = tf.compat.v1.data.experimental.sample_from_datasets([x,y]) 
v_ds = x.concatenate(y)

if training:
    pass
    #v_ds = v_ds.repeat(int(buffer_size/batch_size))
    #v_ds = v_ds.shuffle(buffer_size=buffer_size)
#v_ds = v_ds.map(lambda y,x: tuple((int(y),int(x))),num_parallel_calls=AUTOTUNE)
v_ds = v_ds.batch(batch_size,drop_remainder=False) 

v_ds = v_ds.repeat(steps_per_epoch)
#v_ds = v_ds.prefetch(buffer_size=AUTOTUNE)

#########################################################
#########################################################


for i,x in ds:
    print(x)

inp = tf.keras.layers.Input((1))
layer = tf.keras.layers.Dense(1,activation='sigmoid')(inp)
out = layer

model = tf.keras.Model(inputs=inp,outputs=out)

bce=tf.keras.losses.BinaryCrossentropy(from_logits=False)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.0e-5,clipnorm=1.0),loss='binary_crossentropy',metrics=['binary_accuracy','mse'],run_eagerly=True)

model.fit(ds,validation_data=v_ds,epochs=3)
#model.fit(ds)#,validation_data=v_ds,validation_steps=2,verbose=2)
