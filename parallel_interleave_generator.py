import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
N = 100


def generator(label,x):
    if x == 5:
        return
    yield label,x

# def dataset(n):
#   return tf.data.Dataset.from_generator(generator,output_types=tf.int32,args=n)

# def map_pass(x):
#     return x

ds = tf.data.Dataset.range(1, N).interleave(lambda x: tf.data.Dataset.from_generator(generator,output_types=(tf.int32,tf.int32),args=(1,x)), cycle_length=N,num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds = ds.shuffle(10*N)
ds = ds.repeat(2)

for i,x in enumerate(ds):
    print(x)
