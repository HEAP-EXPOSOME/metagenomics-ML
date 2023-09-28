def code_to_run():
    import os
    os.environ['TF_GPU_THREAD_MODE'] = "gpu_private"

    DATA_DIR = "/scratch/project_2002659/heap-tcga-files"
    METADATA_FILE=f"{DATA_DIR}/10904_metadata.csv"
    HPV_FILE=f"{DATA_DIR}/hpv_viruses.fasta"
    PADDING_CHAR=b'P'

    SEQUENCE_LEN = 647 # 8192 # max([len(virus) for virus in hpv_viruses]) = 8104
    BATCH_SIZE=64

    DROPOUT=0.0
    FILTERS=1024*4
    KERNEL_SIZE=18

    import gzip
    import tensorflow as tf
    from tensorflow import keras
    #keras.mixed_precision.set_global_policy("mixed_float16")
    import time
    import sklearn.metrics
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from collections import Counter
    import random
    random.seed(42)
    tf.random.set_seed(42)
    def load_sequences_from_hpvfile(filename, length):
        viruses=[]
        virus=[]
        with open(filename) as f:
            while True:
                line = f.readline()
                if not line or line.startswith(">"):
                    viruses.append(virus)
                    if not line: break
                    virus=[]
                else:
                    virus += list(line.rstrip())

        sequences=[]
        for virus in viruses:
            sequences.extend([virus[i:i+length] for i in range(0, len(virus), length)])
        return sequences

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
#        sequences=[]
#        for offset in [0, length//2]:
#            sequences.extend([reads[i:i+length] for i in range(offset, len(reads), length)])
#        return sequences

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

    def prepare_datasets(sequences_positive, sequences_negative, length, padding_symbol):
        tensors=[]
        tensors.extend([(s, True) for s in sequences_positive])
        tensors.extend([(s, False) for s in sequences_negative])
        
        #print("In generator before SMOTE")
        #X_train=[]
        #y_train=[]
        #
        #X_train.extend([list(s) + ([padding_symbol] * (length - len(s))) for s in sequences_positive])
        #y_train.extend([True] * len(sequences_positive))
        #
        #X_train.extend([list(s) + ([padding_symbol] * (length - len(s))) for s in sequences_negative])
        #y_train.extend([True] * len(sequences_negative))
        #
        #X_train=[one_hot_encoder(x) for x in X_train]
        #print(f"Sequence count before resample: {len(X_train)}")
        #
        #from imblearn.over_sampling import SMOTE
        #oversample = SMOTE(sampling_strategy="minority", random_state=42)
        #X_train, y_train = oversample.fit_resample(X_train, y_train)
        #print(f"Sequence count after resample: {len(X_train)}")
        #tensors=zip(X_train, y_train)

        def generator():
            #random.shuffle(tensors)
            for x, label in tensors:
                padding = [padding_symbol] * (length - len(x))
                yield (x+padding, label)
        dataset = tf.data.Dataset.from_generator(generator, output_signature=(tf.TensorSpec(shape=(None), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.int32)))\
                        .map(lambda subsequence_x, subsequence_y: (one_hot_encoder(subsequence_x), subsequence_y), num_parallel_calls=tf.data.AUTOTUNE)\
                        .shuffle(1024*64, reshuffle_each_iteration=True)\
                        .batch(BATCH_SIZE, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

#    def extract_positive_and_negative_samples(data, num_of_positives, num_of_negatives, key_name='diamond_positivity'):
#        positives = data[data[key_name] == True][:num_of_positives]["sample"]
#        negatives = data[data[key_name] == False][:num_of_negatives]["sample"]
#
#        return positives, negatives

    positives=load_sequences_from_hpvfile(HPV_FILE, SEQUENCE_LEN)
    print(f"Total number of sequences from HPV file: {len(positives)}")

    #metadata filename changed
    metadata = pd.read_csv(METADATA_FILE)
    cervix=pd.DataFrame(metadata[(metadata["site_of_resection_or_biopsy"] == "Cervix uteri")])
    print(f"Available files with with cervix: {len(cervix)}")

    cervix=cervix[cervix["positivity"] == False]
    print(f"Available files without HPV with cervix: {len(cervix)}")

    sizes = [ os.stat(f"{DATA_DIR}/{name}").st_size / (1024 * 1024) if os.path.exists(f"{DATA_DIR}/{name}") else 1000 for name in cervix["sample"] ] 
    cervix["sizes"] = sizes
    cervix = cervix[cervix["sizes"] < 35]
    print(f"Available files without HPV with size < 35 MB: {len(cervix)}")

#    cervix_smallest=cervix.nsmallest(3, 'sizes')
#    print(f"Smallest file(s) with HPV: {len(cervix_smallest)}")
    
    negatives=[]
    cervix_smallest=cervix.sort_values("sizes")
    for fqgzip in cervix_smallest["sample"]:
        sequences=load_sequences_from_fqgzip(f"{DATA_DIR}/{fqgzip}", SEQUENCE_LEN)
        print(f"\t {fqgzip} -> {len(sequences)}")
        negatives.extend(sequences)
        if len(negatives)*2 >= len(positives): break

    if len(negatives)*2 >= len(positives):
        negatives=negatives[:len(positives)//2]
    

    print (f"Size of positives: {len(positives)}, size of negatives: {len(negatives)}")

#    positives_train,positives_eval,positives_test = positives,positives,positives
    positives_train, positives_eval_test = train_test_split(positives, test_size=0.05, random_state=42)
    positives_eval, positives_test = train_test_split(positives_eval_test, test_size=0.2, random_state=42)
    positives_eval_test, positives = None, None

    negatives_train, negatives_eval_test = train_test_split(negatives, test_size=0.05, random_state=42)
    negatives_eval, negatives_test = train_test_split(negatives_eval_test, test_size=0.2, random_state=42)
    negatives_eval_test, negatives = None, None

    positives_train = shuffle(positives_train, random_state=42)
    positives_eval = shuffle(positives_eval, random_state=42)
    positives_test = shuffle(positives_test, random_state=42)

    negatives_train = shuffle(negatives_train, random_state=42)
    negatives_eval = shuffle(negatives_eval, random_state=42)
    negatives_test = shuffle(negatives_test, random_state=42)

    train_dataset = prepare_datasets(positives_train, negatives_train, SEQUENCE_LEN, PADDING_CHAR)
    eval_dataset = prepare_datasets(positives_eval, negatives_eval, SEQUENCE_LEN, PADDING_CHAR)
    test_dataset = prepare_datasets(positives_test, negatives_eval, SEQUENCE_LEN, PADDING_CHAR)

    print(f"Data prepared (train={len(positives_train)}/{len(negatives_train)}, eval={len(positives_eval)}/{len(negatives_eval)}, test={len(positives_test)}/{len(negatives_test)}).")
    print("Learning...", flush=True)
    model_cnn = keras.models.Sequential([
        keras.layers.Input(shape=[SEQUENCE_LEN, 6]),
#        keras.layers.Dropout(rate=DROPOUT),
        keras.layers.Masking(mask_value=padding_mask.numpy()),
        keras.layers.Conv1D(FILTERS, KERNEL_SIZE, activation='relu'),
#        keras.layers.SpatialDropout1D(rate=DROPOUT),#
        keras.layers.MaxPool1D(KERNEL_SIZE),
        keras.layers.Conv1D(FILTERS, KERNEL_SIZE, activation='relu'),
#        keras.layers.SpatialDropout1D(rate=DROPOUT),#
        keras.layers.MaxPool1D(KERNEL_SIZE),
        keras.layers.Flatten(),
        keras.layers.Dense(FILTERS, activation='relu'),
#        keras.layers.Dropout(rate=DROPOUT),#
        keras.layers.Dense(1, activation='sigmoid') # Dense(?=len(hpv)+1, 'softmax'
#        keras.layers.Dense(1, activation='softmax')
    ])

    model_cnn.summary(line_length=120, expand_nested=True, show_trainable=True)
    model_cnn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                        loss='binary_crossentropy', # categorical_crossentropy?
                        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    callback_modelcheckpoint = keras.callbacks.ModelCheckpoint(
        filepath=".\\", \
        monitor='accuracy', \
        mode='max', \
        save_best_only=True
    )
    callback_earlystopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', # validation loss
        mode='auto',
        restore_best_weights=True,
        patience=5,
        verbose=1,
        start_from_epoch=10,
    )
    
    #history = model_cnn.fit(train_dataset, validation_data=eval_dataset, epochs=3, use_multiprocessing=True, workers=64, verbose=2)
#    class History: history=[]
#    history = History()
    history = model_cnn.fit(train_dataset, validation_data=eval_dataset, epochs=140, use_multiprocessing=True, workers=64, verbose=2, callbacks=[callback_earlystopping])
#    history = model_cnn.fit(train_dataset, validation_data=eval_dataset, epochs=30, use_multiprocessing=True, workers=64, verbose=2)
    model_output_dir='cnn_model-'+os.getenv('SLURM_JOBID','unknown')
    model_cnn.save(model_output_dir)
    with open(__file__, "r") as fin, open(f"{model_output_dir}/script.py", "w") as fout:
        fout.write(fin.read())
        import sys

        for i in range(len(sys.argv)): fout.write(f"\n# argv: {i}={sys.argv[i]}")
        for k in os.environ: fout.write(f"\n# env: {k}={os.environ[k]}")

    # https://stackoverflow.com/a/61328750/1786868
    import numpy as np
    np.save('history.npy',history.history)

    # ------------------------------------
    print("Predicting...")
    def print_prediction(filename, prediction):
        histogram, bin_edges = np.histogram(prediction, bins=10, range=(0,1))
        bhistogram, bin_edges = np.histogram(prediction, bins=2, range=(0,1))
        try:
            label = metadata[metadata["sample"] == os.path.basename(filename)]["positivity"].iloc[0]
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            label = None

        print(filename, label, histogram.argmax(), histogram, flush=True)
        print(filename, label, "binary", bhistogram.argmax(), bhistogram, bhistogram/sum(bhistogram), flush=True)

    def print_metrics(label, prediction):
        label_array = [label]*len(prediction)
        y_pred = np.where(prediction > 0.5, True, False)

        print(f"Classification report ({label=}):")
        try:
            print(sklearn.metrics.classification_report(label_array, y_pred))
        except Exception as err:
            print(f"Unexpected {err=}: {label=}")
        
        print(f"Confusion matrix ({label=}):")
        try:
            print(sklearn.metrics.confusion_matrix(label_array, y_pred))
        except Exception as err:
            print(f"Unexpected {err=}: {label=}")

        print("---", flush=True)


    prediction = model_cnn.predict(test_dataset, verbose=0)
    print_prediction("<multiple>",prediction)
    print_metrics(False, prediction)
    print_metrics(True, prediction)

    filenames = [ "sample_05483.fq.gz", "sample_00058.fq.gz", "sample_00064.fq.gz", "sample_00005.fq.gz", "hpv_viruses.fq.gz"] # zupełnie różne spodziewane wartości i wyniki
    filenames = [ f"{DATA_DIR}/{name}" for name in filenames]
    filenames = [ name for name in filenames if os.path.isfile(name)]
    for filename in filenames:
        sequences = load_sequences_from_fqgzip(filename, SEQUENCE_LEN)
        def generator():
            for x in sequences:
                padding = [PADDING_CHAR] * (SEQUENCE_LEN - len(x))
                yield ((x+padding), False)
        dataset = tf.data.Dataset.from_generator(generator, output_signature=(tf.TensorSpec(shape=(None), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.int32)))\
                                                .map(lambda subsequence_x, subsequence_y: (one_hot_encoder(subsequence_x), subsequence_y), num_parallel_calls=tf.data.AUTOTUNE)\
                                                .batch(BATCH_SIZE, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        prediction = model_cnn.predict(dataset, verbose=0)
        print_prediction(filename,prediction)
        print_metrics(False, prediction)
        print_metrics(True, prediction)


if __name__ == "__main__":
    code_to_run()

