def main(argv):
    import os
    os.environ['TF_GPU_THREAD_MODE'] = "gpu_private"
    DATA_DIR = "/scratch/project_2002659/heap-tcga-files"
    MODEL_DIR = "/projappl/project_2002659/cnn-predict/models"
    METADATA_FILE=f"{DATA_DIR}/10904_metadata.csv"
    PADDING_CHAR=b'P'

    VOCAB_LEN = 65
    SEQUENCE_LEN = 647

    WORD_FEATURE_DIM = 20
    DOC_FEATURE_DIM = 64

    BATCH_SIZE=64

    import gzip
    import tensorflow as tf
    from tensorflow import keras
    #keras.mixed_precision.set_global_policy("mixed_float16")
    import time
    import sklearn.metrics
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np

    def load_sequences_from_fqgzip(filename, length):
        reads=[]

        with open(filename, 'rb') as f:
            with gzip.open(f, 'rt') as g:
                while True:
                    g.readline()
                    code = list(g.readline().rstrip())
                    g.readline()
                    last = g.readline()

                    if last == '':
                        break

                    reads.extend(code)
        return [reads[i:i+length] for i in range(0, len(reads), length)]

#    def load_sequences_from_fqgzip(filename, length):
#        reads=[]
#
#        with open(filename, 'rb') as f:
#            with gzip.open(f, 'rt') as g:
#                while True:
#                    g.readline()
#                    code = list(g.readline().rstrip())
#                    g.readline()
#                    last = g.readline()
#
#                    if last == '':
#                        break
#
#                    reads.append(code)
#        sequences=[]
#        maxlen=len(max(reads, key=len))
#        x=[]
#        for s in reads:
#            x.extend(list(s))
#            if len(s)<maxlen:
#                sequences.extend([x[i:i+length] for i in range(0, len(x), length)])
#                x=[]
#        return sequences


    def prepare_generator_onefile(filename, length, padding_symbol = None):
        sequences = load_sequences_from_fqgzip(filename, length)
        def generator():
            for x in sequences:
                padding = [padding_symbol] * (length - len(x))
                yield(x+padding, False)
        return generator

#    def prepare_generator_onefile(filename, length, padding_symbol = None):
#        sequences = load_sequences_from_fqgzip(filename, length)
#        def generator():
#          for offset in range(0, length, 1):
#            for x in sequences:
#                x = x[offset:]
#                for i in range(0, len(x), length):
#                    v = x[i:i+length]
#                    padding = [padding_symbol] * (length - len(v))
#                    yield(v+padding, False)
#        return generator

    class OneHotEncoder:
        def __init__(self):
            #cf. A. Geron's book, p. 431
            self.vocabulary = [b'T', b'C', b'G', b'A', b'N', PADDING_CHAR] # 4 letters for TCGA, 1 for no data, 1 for padding
            indices = tf.range(len(self.vocabulary), dtype=tf.int64)
            table_init = tf.lookup.KeyValueTensorInitializer(self.vocabulary, indices)
            self.lookup_table = tf.lookup.StaticHashTable(table_init, -1)

        def __call__(self, data):
            indices=self.lookup_table.lookup(data)
            one_hot=tf.one_hot(indices, depth=len(self.vocabulary))
            return one_hot
    one_hot_encoder = OneHotEncoder()

    padding_mask = one_hot_encoder(tf.constant([PADDING_CHAR]))

    metadata = pd.read_csv(METADATA_FILE)

    cervix_endometrium=pd.DataFrame(metadata[(metadata["site_of_resection_or_biopsy"] == "Endometrium") | (metadata["site_of_resection_or_biopsy"] == "Cervix uteri")])
    filenames = [ name for name in cervix_endometrium["sample"]]
    #filenames = [ "sample_05483.fq.gz", "sample_00058.fq.gz", "sample_00064.fq.gz", "sample_00005.fq.gz", "hpv_viruses.fq.gz"] # zupełnie różne spodziewane wartości i wyniki
    #filenames = [ "hpv_viruses.fq.gz"]
    print("Samples with cervix_endometrium: (%d) "%len(filenames)) #, filenames)

    filenames = [ f"{DATA_DIR}/{name}" for name in filenames]
    #print("Filenames with cervix_endometrium (%d): "%len(filenames)) #, filenames)

    filenames = [ name for name in filenames if os.path.isfile(name)]
    print("Existing files with cervix_endometrium (%d): "%len(filenames))#, filenames)

    if len(argv) == 2:
        end = int(argv[1])
        filenames = filenames[:end]
    elif len(argv) == 3:
        start = int(argv[1])
        end = int(argv[2])
        filenames = filenames[start:end]

#    filenames = filenames[:300]
#    filenames = filenames[300:600]
#    filenames = filenames[600:]
    print("Filenames to process (%d): "%len(filenames), filenames, flush=True)

    model_cnn = keras.models.load_model(f"{MODEL_DIR}/cnn_model-2464956", compile=True)  # model_cnn.compile()
#    model_cnn.compile(run_eagerly=True)

    label_arrays=[]
    y_pred_arrays=[]
    with open('save_file.npz', 'wb') as save_file:
        for filename in filenames:
            dataset = tf.data.Dataset.from_generator(prepare_generator_onefile(filename, SEQUENCE_LEN, PADDING_CHAR),
                                                     output_signature=(tf.TensorSpec(shape=(None), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.int32)))\
                                                    .map(lambda subsequence_x, subsequence_y: (one_hot_encoder(subsequence_x), subsequence_y), num_parallel_calls=tf.data.AUTOTUNE)\
                                                    .batch(BATCH_SIZE, drop_remainder=False)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            prediction = model_cnn.predict(dataset, verbose=0)

            histogram, bin_edges = np.histogram(prediction, bins=10, range=(0,1))
            bhistogram, bin_edges = np.histogram(prediction, bins=2, range=(0,1))
            try:
                label = metadata[metadata["sample"] == os.path.basename(filename)]["positivity"].iloc[0]
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                label = None

            print(filename, label, histogram.argmax(), histogram, flush=True)
            print(filename, label, "binary", bhistogram.argmax(), bhistogram, bhistogram/sum(bhistogram), flush=True)


            filename_array = [filename]*len(prediction)
            label_array = [label]*len(prediction)
            y_pred = np.where(prediction > 0.5, True, False)

            try:
                print(sklearn.metrics.classification_report(label_array, y_pred))
                label_arrays.extend(label_array)
                y_pred_arrays.extend(y_pred)
            except Exception as err:
                print(f"Unexpected {err=}: {label=}")

            np.savez_compressed(save_file, filenames=filename_array, labels=label_array, predictions=prediction)
            save_file.flush()

    print("prediction completed")
    if len(label_arrays)>0:
        print(sklearn.metrics.classification_report(label_arrays, y_pred_arrays))

if __name__ == "__main__":
    from sys import argv
    main(argv)
