def code_to_run():
    import os
    os.environ['TF_GPU_THREAD_MODE'] = "gpu_private"
    VOCAB_LEN = 65
    SEQUENCE_LEN = 80

    WORD_FEATURE_DIM = 20
    DOC_FEATURE_DIM = 64

    BATCH_SIZE=32

    import gzip
    import tensorflow as tf
    from tensorflow import keras
    #keras.mixed_precision.set_global_policy("mixed_float16")
    import time
    import sklearn.metrics
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    import pandas as pd

    try:
        import hops
        import hops.hdfs
        import hops.pandas_helper
        HOPS=True
    except:
        HOPS = False
    print(f"HOPS={HOPS}")

    import random
    random.seed(42)
    def prepare_generator_multifile(filenames, metadata_csv, length = None, padding_symbol = None, key_name="diamond_positivity"):
        def generator():
            label_table=pd.read_csv(metadata_csv) if not HOPS else hops.pandas_helper.read_csv(metadata_csv)
            opened_files = [ open(filename, "rb") if not HOPS else hops.hdfs.open_file (filename) for filename in filenames ]
            opened_gzips = [ gzip.open(file, "rt") for file in opened_files]
            labels = [ label_table[label_table["sample"] == os.path.basename(filename)][key_name].iloc[0] for filename in filenames ]
            data = list(zip(opened_files, filenames, opened_gzips, labels))

            while(len(data) != 0):
                index = random.randrange(0, len(data))
                _, filename, gzip_opened, label = data[index]

                gzip_opened.readline()
                code = gzip_opened.readline().rstrip()
                gzip_opened.readline()
                last = gzip_opened.readline()

                if last == '':
                    del data[index]
                    continue

                x = list(code)
                padding = [] if length is None else [padding_symbol] * (length - len(x))
                yield x + padding, label
            for gzip_opened in opened_gzips:
                gzip_opened.close()
            for file in opened_files:
                file.close()
        return generator

    class OneHotEncoder:
        def __init__(self):
            #cf. A. Geron's book, p. 431
            self.vocabulary = [b'T', b'C', b'G', b'A', b'N', b'P'] # 4 letters for TCGA, 1 for no data, 1 for padding
            indices = tf.range(len(self.vocabulary), dtype=tf.int64)
            table_init = tf.lookup.KeyValueTensorInitializer(self.vocabulary, indices)
            self.lookup_table = tf.lookup.StaticHashTable(table_init, -1)

        def __call__(self, data):
            indices=self.lookup_table.lookup(data)
            one_hot=tf.one_hot(indices, depth=len(self.vocabulary))
            return one_hot
    one_hot_encoder = OneHotEncoder()

    padding_mask = one_hot_encoder(tf.constant([b'P']))
    padding_mask

    def prepare_datasets_multifile(filenames, metadata_csv):
        dataset = tf.data.Dataset.from_generator(prepare_generator_multifile(filenames, metadata_csv, SEQUENCE_LEN, b'P', "positivity"), 
                                             output_signature=(tf.TensorSpec(shape=(None), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.int32)))\
                                            .map(lambda subsequence_x, subsequence_y: (one_hot_encoder(subsequence_x), subsequence_y), num_parallel_calls=tf.data.AUTOTUNE)\
                                            .batch(BATCH_SIZE, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    
    def extract_positive_and_negative_samples(data, num_of_positives, num_of_negatives, key_name='diamond_positivity'):
        positives = data[data[key_name] == True][:num_of_positives]["sample"]
        negatives = data[data[key_name] == False][:num_of_negatives]["sample"]

        return positives, negatives


    def identity_filename_demangler(filename):
        return filename
    
    def subdirectory_filename_demangler(filename):
        #sample_01234.fq.gz -> 01234
        extension = filename.split('_')[1].split('.')[0]
        #01234 -> 01
        prefix = extension[:2]
        #01 -> s_01000_01999/sample_01234.fq.gz
        return f"s_{prefix}000_{prefix}999/{filename}"
    demangler=subdirectory_filename_demangler
    
    def path_provider(filename):
        number = int(filename.split('_')[1].split('.')[0])
        if number < 1_000:
            return f"hdfs:///Projects/library/tsga_sample/{filename}"
        else:
            return f"hdfs:///Projects/library/tcga_sample2/{demangler(filename)}"
        

    #metadata filename changed
    metadata = pd.read_csv(f"../data/1000_metadata.csv") if not HOPS else hops.pandas_helper.read_csv(f"{hops.hdfs.project_path()}tcga_Training_Datasets/10904_metadata.csv")
    #removing missing data, with id < 01000; that is with id that begins with "00"
    #metadata=metadata[~metadata['sample'].str.contains('_00')]
    import os
    cervix_endometrium=pd.DataFrame(metadata[(metadata["site_of_resection_or_biopsy"] == "Endometrium") | (metadata["site_of_resection_or_biopsy"] == "Cervix uteri")])
    if not HOPS:
        #correct when full dataset is on rysy
        sizes = [ os.stat(f"../data/{name}").st_size / (1024 * 1024) if os.path.exists(f"../data/{name}") else 100 for name in cervix_endometrium["sample"] ] 
    else:
        sizes = [ hops.hdfs.stat(path_provider(name)).st_size / (1024 * 1024) for name in cervix_endometrium["sample"] ] 
    cervix_endometrium["sizes"] = sizes
    cervix_endometrium = cervix_endometrium[cervix_endometrium["sizes"] < 35]

    positives, negatives = extract_positive_and_negative_samples(cervix_endometrium, 159, 159, 'positivity')
    positives = positives.to_list()
    negatives = negatives.to_list()

    print (f"Size of positives: {len(positives)}, size of negatives: {len(negatives)}")
    positives_train, positives_eval_test = train_test_split(positives, test_size=58, random_state=42)
    positives_eval, positives_test = train_test_split(positives_eval_test, test_size=0.5, random_state=42)

    negatives_train, negatives_eval_test = train_test_split(negatives, test_size=58, random_state=42)
    negatives_eval, negatives_test = train_test_split(negatives_eval_test, test_size=0.5, random_state=42)

    train = positives_train + negatives_train
    eval_ = positives_eval + negatives_eval
    test = positives_test + negatives_test

    train = shuffle(train, random_state=42)
    eval_ = shuffle(eval_, random_state=42)
    test = shuffle(test, random_state=42)

    train = [ (f"../data/" if not HOPS else path_provider(name)) for name in train]
    eval_ = [ (f"../data/" if not HOPS else path_provider(name)) for name in eval_]
    test = [ (f"../data/" if not HOPS else path_provider(name)) for name in test]

    import sklearn.metrics

    metadata = f"../data/1000_metadata.csv" if not HOPS else f"{hops.hdfs.project_path()}tcga_Training_Datasets/10904_metadata.csv"
    train_dataset = prepare_datasets_multifile(train, metadata)
    eval_dataset = prepare_datasets_multifile(eval_, metadata)
    test_dataset = prepare_datasets_multifile(test, metadata)

    DROPOUT=0
    model_cnn = keras.models.Sequential([
        keras.layers.Input(shape=[SEQUENCE_LEN, 6]),
        keras.layers.Dropout(rate=DROPOUT),
        keras.layers.Masking(mask_value=padding_mask.numpy()),
        keras.layers.Conv1D(128, 5, activation='relu'),
        keras.layers.SpatialDropout1D(rate=DROPOUT),
        keras.layers.MaxPool1D(5),
        keras.layers.Conv1D(128, 5, activation='relu'),
        keras.layers.SpatialDropout1D(rate=DROPOUT),
        keras.layers.MaxPool1D(5),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(rate=DROPOUT),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model_cnn.summary()
    opt = keras.optimizers.Adam(learning_rate=0.002)
    model_cnn.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', keras.metrics.Precision(), \
                                                                          keras.metrics.Recall()])

    from hops import tensorboard as hops_tensorboard
    
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=".\\", \
        monitor='accuracy', \
        mode='max', \
        save_best_only=True)
    
    history = model_cnn.fit(train_dataset, validation_data=eval_dataset, epochs=3, use_multiprocessing=True, workers=32,\
                 callbacks=[keras.callbacks.TensorBoard(log_dir=hops_tensorboard.logdir(), profile_batch=2), model_checkpoint_callback])
    model_cnn.save('cnn_model_for_sequence_classification')
    
    from hops import model as hops_model
    hops_model.export('cnn_model_for_sequence_classification', 'cnn_model_for_sequence_classification', metrics={'accuracy':history.history['accuracy'][-1]})

from hops import experiment
experiment.launch(code_to_run)