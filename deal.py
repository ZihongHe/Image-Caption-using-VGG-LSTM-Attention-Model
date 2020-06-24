from os.path import isfile
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, LSTM, Attention, Dropout
from tensorflow.keras import Model, optimizers, preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

def image_readin(width, height, captions_info_file, categories_info_file, image_load):

    from pycocotools.coco import COCO

    images = []
    categories = []
    captions = []

    coco_captions = COCO(captions_info_file)
    coco_cats = COCO(categories_info_file)

    # id of all images
    ids = list(coco_captions.imgs)

    for i in range(len(coco_captions.imgs.keys())):

        image = Image.open(image_load + coco_captions.imgs[ids[i]]['file_name'])
        image = image.resize((width, height), Image.ANTIALIAS)
        image = np.array(image)

        images.append(image)
        cap_load = coco_captions.loadAnns(coco_captions.getAnnIds(ids[i]))
        cap = []
        for j in range(len(cap_load)):
            cap.append(cap_load[j]['caption'])
        captions.append(cap)
        cat_load = coco_cats.loadAnns(coco_cats.getAnnIds(ids[i]))
        cat = []
        for j in range(len(cat_load)):
            cat.append(coco_cats.cats[cat_load[j]['category_id']]['supercategory'])
        categories.append(cat)
        print(str(i), "of", len(coco_captions.imgs.keys()), "images have been processed")

    return images, categories, captions, ids

def caption_processing(captions):

    from nltk.tokenize import word_tokenize
    import gensim.downloader as api

    captions_new = []
    captions_word2vec = []
    word2vec = api.load('word2vec-google-news-300')

    word_not_in_dict = 0

    for i in range(len(captions)):

        captions_word2vec_sequence = []
        captions_sequence = []

        captions[i][0] = word_tokenize(captions[i][0])

        for j in range(len(captions[i][0])):
            try:
                captions_word2vec_sequence.append(word2vec[captions[i][0][j].lower()])
                captions_sequence.append(captions[i][0][j].lower())
            except:
                word_not_in_dict += 1

        captions_new.append(captions_sequence)
        captions_word2vec.append(captions_word2vec_sequence)

    print("there is", word_not_in_dict, "words not in word2vec dictionary")

    captions = None

    captions_concate = []

    for seqs in captions_new:
        if(seqs == 'a'):
            print('a')
        captions_concate+=seqs

    captions_concate = sorted(list(set((captions_concate))))

    wordvector_dictionary = [word2vec[element] for element in captions_concate]

    wordforid = dict((i, t) for i, t in enumerate(captions_concate))
    idofword = dict((t, i) for i, t in enumerate(captions_concate))

    word2vec = None

    for i in range(len(captions_new)):
        captions_new[i] = [idofword[token] for token in captions_new[i]]
        captions_new[i] = to_categorical(captions_new[i], num_classes=len(wordforid))

    return captions_word2vec, captions_new, idofword, wordforid, wordvector_dictionary

def cat_processing(categories):

    cat_concate = []
    for cat in categories:
        cat_concate+=cat

    cat_concate = sorted(list(set((cat_concate))))

    catforid = dict((i, t) for i, t in enumerate(cat_concate))
    idofcat = dict((t, i) for i, t in enumerate(cat_concate))

    for i in range(len(categories)):
        categories[i] = [idofcat[token] for token in categories[i]]
        categories[i] = to_categorical(categories[i], num_classes=len(catforid))

    return categories, idofcat, catforid

def model_cat(images, categories):

    imgs = []
    labels = []


    for i in range(int(len(categories))):
        if(images[i].shape == (128,128,3)):
            try:
                labels.append((categories[i][0]))
                imgs.append(images[i])
            except:
                print(categories[i])

    imgs_train = np.asarray(imgs[:int(len(imgs) * 0.98)])
    imgs_validate = np.asarray(imgs[int(len(imgs) * 0.98):])

    labels_train = np.asarray(labels[:int(len(labels) * 0.98)])
    labels_validate = np.asarray(labels[int(len(labels) * 0.98):])


    input = Input(shape=(imgs_train.shape[1], imgs_train.shape[2], 3))

    # convolution layers （VGG structure)
    conv_layer1 = Conv2D(filters=32, kernel_size=3, activation='relu')
    conv_output1 = conv_layer1(input)
    conv_layer2 = Conv2D(filters=32, kernel_size=3, activation='relu')
    conv_output2 = conv_layer2(conv_output1)
    pool_layer1 = MaxPool2D(pool_size=(2, 2))
    pool_output1 = pool_layer1(conv_output2)
    conv_layer3 = Conv2D(filters=64, kernel_size=3, activation='relu')
    conv_output3 = conv_layer3(pool_output1)
    conv_layer4 = Conv2D(filters=64, kernel_size=3, activation='relu')
    conv_output4 = conv_layer4(conv_output3)
    pool_layer2 = MaxPool2D(pool_size=(2, 2))
    pool_output2 = pool_layer2(conv_output4)
    drop_out2 = Dropout(0.2)(pool_output2)

    Flatten_output = Flatten()(drop_out2)

    dense_sigmoid = Dense(16, activation='sigmoid')(Flatten_output)

    output = Dense(labels_train.shape[1], activation='softmax')(dense_sigmoid)

    Model_cat = Model(input, output)

    # optimizer
    optimizer = optimizers.Adam(lr=0.00002)

    Model_cat.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['mse', 'accuracy'])

    history = Model_cat.fit(imgs_train, labels_train, epochs=50, batch_size=256, validation_data=[imgs_validate, labels_validate])

    Model_cat.save("Model_cat.hdf5")

    cat_precict = Model_cat.predict(imgs_validate)

    num_true_predict = 0
    for i in range(labels_validate.shape[0]):
        for j in range(labels_validate.shape[1]):
            labels_validate[i][j] = np.argmax(labels_validate[i][j])
        if np.argmax(cat_precict[i]) in labels_validate[i]:
            num_true_predict += 1
        print(np.argmax(cat_precict[i]))
    print("test accuracy for categories is:", num_true_predict / labels_validate.shape[0])

    return Model_cat, history.history


def model_caption(images, captions_word2vec, captions_onehot, word2vec):
    decoder_input_raw = []
    decoder_output_raw = []
    encoder_raw = []

    encoder_test = np.asarray(images[:int(0.1 * len(images))])
    decoder_input_test = np.asarray(captions_word2vec[:int(0.1 * len(captions_word2vec))])
    decoder_output_test = np.asarray(captions_onehot[:int(0.1 * len(captions_onehot))])

    encoder_test = encoder_test.reshape(encoder_test.shape[0], 1, encoder_test.shape[1], encoder_test.shape[2], 3)

    for i in range(int(len(captions_word2vec))):
        if(images[i].shape==(128, 128, 3)):
            decoder_input_raw += [captions_word2vec[i]]
            decoder_output_raw += [captions_onehot[i]]
            encoder_raw +=[images[i]]
    max = 0
    for i in range(len(captions_onehot)):
        if(max < len(captions_onehot[i])):
            max = len(captions_word2vec[i])

    encoder_train_input = np.asarray(encoder_raw[:int(0.98 * len(encoder_raw))])
    encoder_validate_input = np.asarray(encoder_raw[int(0.98 * len(encoder_raw)):])

    decoder_input_raw = np.asarray(decoder_input_raw)
    decoder_output_raw = np.asarray(decoder_output_raw)

    decoder_input_raw = preprocessing.sequence.pad_sequences(decoder_input_raw, maxlen=None, dtype='float',
                                                             padding='post', truncating='post', value=0)

    decoder_output_raw = preprocessing.sequence.pad_sequences(decoder_output_raw, maxlen=None, dtype='float',
                                                              padding='post', truncating='post', value=0)

    decoder_input_raw = preprocessing.sequence.pad_sequences(decoder_input_raw, maxlen=decoder_input_raw.shape[1] + 1,
                                                             dtype='float',
                                                             padding='pre', truncating='pre', value=0)

    decoder_output_raw = preprocessing.sequence.pad_sequences(decoder_output_raw,
                                                              maxlen=decoder_output_raw.shape[1] + 1, dtype='float',
                                                              padding='post', truncating='post', value=0)

    decoder_train_input = decoder_input_raw[:int(len(decoder_input_raw) * 0.98)]

    decoder_train_output = decoder_output_raw[:int(len(decoder_output_raw) * 0.98)]

    decoder_validate_input = decoder_input_raw[int(len(decoder_input_raw) * 0.98):]

    decoder_validate_output = decoder_output_raw[int(len(decoder_output_raw) * 0.98):]

    encoder_input = Input(shape=(encoder_train_input.shape[1], encoder_train_input.shape[2], 3))

    # convolution layers （VGG structure)
    conv_layer1 = Conv2D(filters=32, kernel_size=3, activation='relu')
    conv_output1 = conv_layer1(encoder_input)
    conv_layer2 = Conv2D(filters=32, kernel_size=3, activation='relu')
    conv_output2 = conv_layer2(conv_output1)
    pool_layer1 = MaxPool2D(pool_size=(2, 2))
    pool_output1 = pool_layer1(conv_output2)
    conv_layer3 = Conv2D(filters=64, kernel_size=3, activation='relu')
    conv_output3 = conv_layer3(pool_output1)
    conv_layer4 = Conv2D(filters=64, kernel_size=3, activation='relu')
    conv_output4 = conv_layer4(conv_output3)
    pool_layer2 = MaxPool2D(pool_size=(2, 2))
    pool_output2 = pool_layer2(conv_output4)
    drop_out2 = Dropout(0.2)(pool_output2)

    # flatten layer
    Flatten_layer = Flatten()
    Flatten_output = Flatten_layer(drop_out2)

    Full_connect_h = Dense(128, activation='relu')
    Full_connect_h_output = Full_connect_h(Flatten_output)
    Full_connect_c = Dense(128, activation='relu')
    Full_connect_c_output = Full_connect_c(Flatten_output)
    Full_connect_attention = Dense(512, activation="relu")
    Full_connect_attention_output = Full_connect_attention(Flatten_output)

    # decoder
    decoder_input = Input(shape=(None, decoder_train_input.shape[2]))

    num_hidden_lstm = int(Full_connect_h_output.shape[1])

    decoder_lstm = LSTM(num_hidden_lstm, return_state=True, return_sequences=True)
    decoder_out, decoder_h, decoder_c = decoder_lstm(decoder_input,
                                                     initial_state=[Full_connect_h_output, Full_connect_c_output])

    # ensure a same dimmension for dot product in attention layer
    decoder_dense1 = Dense(Full_connect_attention_output.shape[1], activation='relu')
    decoder_dense1_output = decoder_dense1(decoder_out)


    # Attention layer
    attention = Attention()
    attention_distribute = attention([decoder_dense1_output, Full_connect_attention_output])

    # attention output into softmax for output
    decoder_dense2 = Dense(decoder_train_output.shape[2], activation='softmax')
    decoder_output = decoder_dense2(attention_distribute)


    Model_train = Model([encoder_input, decoder_input], decoder_output)

    # optimizer
    optimizer = optimizers.SGD(lr=0.000002, momentum=0.9)  # , nesterov=True)# decay=1e-6)

    Model_train.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['mse', 'accuracy'])

    Model_train.load_weights("Model_train.h5")


    history = Model_train.fit([encoder_train_input, decoder_train_input], decoder_train_output, epochs=5000,
                              batch_size=256,
                              validation_data=[[encoder_validate_input, decoder_validate_input],
                                               decoder_validate_output])
    Model_train.save("Model_train.hdf5")

    # encoder model

    Model_encoder = Model(encoder_input,
                          [Full_connect_h_output, Full_connect_c_output, Full_connect_attention_output])

    Model_encoder.save("Model_encoder.hdf5")

    # decoder model
    h_decoder_input = Input(shape=(num_hidden_lstm,))
    c_decoder_input = Input(shape=(num_hidden_lstm,))
    attention_decoder_input = Input(shape=(Full_connect_attention_output.shape[1],))

    decoder_out, decoder_h_output, decoder_c_output = decoder_lstm(decoder_input,
                                                                   initial_state=[h_decoder_input, c_decoder_input])

    decoder_dense_out1 = decoder_dense1(decoder_out)
    attention_distribute = attention([attention_decoder_input, decoder_dense_out1])

    decoder_output = decoder_dense2(attention_distribute)

    Model_decoder = Model([decoder_input, h_decoder_input, c_decoder_input, attention_decoder_input],
                          [decoder_output, decoder_h_output, decoder_c_output])

    Model_decoder.save("Model_decoder.hdf5")

    # prediction

    # BLUE score list
    BLUE_score_list = []

    # maximum words in a sentence
    max_word = 20

    for i in range(len(encoder_test)):
        outputs_list = []
        outputs = ""
        for k in range(len(decoder_input_test[i])):
            if (sum(decoder_output_test[i][k]) != 0):
                output = np.argmax(decoder_output_test[i][k])
                output = wordforid[output]
                outputs = outputs + output + ' '
                outputs_list.append(output)
        print("actually reference:", outputs)

        [h, c, attention_] = Model_encoder.predict(encoder_test[i])

        # a blank image for normalisation
        blank_reference = np.ones((1, width, height, 3)) * 128
        [h_reference, c_reference, attention_reference] = Model_encoder.predict(blank_reference)

        current_decoder_input = np.zeros([1, 1, len(decoder_input_test[0][0])])
        current_decoder_input_reference = np.zeros([1, 1, len(decoder_input_test[0][0])])
        outputs_hat = ""
        outputs_hat_list = []

        outputs_hatt = ""
        outputs_hatt_list = []

        # for BLUE score counting
        BLUE = []
        BLUE_element = {}

        for j in range(max_word):
            [current_decoder_output, h, c] = Model_decoder.predict([current_decoder_input, h, c, attention_])
            [current_decoder_output_reference, h_reference, c_reference] = Model_decoder.predict(
                [current_decoder_input_reference, h_reference, c_reference, attention_reference])

            # normalisation of decoder output
            current_decoder_out = (current_decoder_output - current_decoder_output_reference) / (
                    current_decoder_output ** 0.75)

            output_hat = np.argmax(current_decoder_out)
            output_hat = wordforid[output_hat]

            output_hatt = np.argmax(current_decoder_output)
            current_decoder_input = word2vec[output_hatt]
            current_decoder_input = current_decoder_input.reshape(1,1,300)
            output_hatt = wordforid[output_hatt]

            output_hattt = np.argmax(current_decoder_output_reference)
            current_decoder_input_reference = word2vec[output_hattt]
            current_decoder_input_reference = current_decoder_input_reference.reshape(1, 1, 300)

            count_current_word = outputs_list.count(output_hat)

            # We use the advanced BLUE score, if number of the word in output is bigger than reference, use the
            # maximum word number in reference
            if (count_current_word > 0):
                try:
                    if (BLUE_element[output_hat] < count_current_word):
                        BLUE_element[output_hat] += 1
                        BLUE.append(1)
                except:
                    BLUE_element[output_hat] = 1
                    BLUE.append(1)
            else:
                BLUE.append(0)
            outputs_hat = outputs_hat + output_hat + ' '
            outputs_hat_list.append(output_hat)

            outputs_hatt = outputs_hatt + output_hatt + ' '
            outputs_hatt_list.append(output_hatt)
        print("normal prediction:", outputs_hat)
        print("prediction:", outputs_hatt)
        if (len(BLUE) > 0):
            BLUE_score = sum(BLUE) / len(BLUE)
            BLUE_score_list.append(BLUE_score)
            print("BLUE score is", BLUE_score)

        print('\n')

    overall_BLUE_Score = sum(BLUE_score_list) / len(BLUE_score_list)
    print("overall BLUE score is:", overall_BLUE_Score)

    return Model_encoder, Model_decoder, history.history, overall_BLUE_Score


def cat_cap_of_img(given_img_path, width, height, catforid, wordforid, Model_cat, Model_encoder, Model_decoder, wordvector_dictionary):

    image = Image.open(given_img_path)
    image = image.resize((width, height), Image.ANTIALIAS)
    image = np.array(image)
    image = image.reshape([1, width, height, 3])

    # blank captions reference
    blank_reference = np.ones((1,width,height,3))*128

    label_predict = catforid[np.argmax(Model_cat.predict(image))]

    [h, c, Flatten_as_attention] = Model_encoder.predict(image)
    [h_reference, c_reference, Flatten_as_attention_reference] = Model_encoder.predict(blank_reference)

    max_attempt = 200
    word_count = 0
    outputs_hat_list = []
    outputs_hat = ''
    last_word = None

    current_decoder_input = np.zeros([1, 1, 300])
    current_decoder_input_reference = np.zeros([1, 1, 300])
    for j in range(max_attempt):
        [current_decoder_output, h, c] = Model_decoder.predict([current_decoder_input, h, c, Flatten_as_attention])
        [current_decoder_output_reference, h_reference, c_reference] = Model_decoder.predict([current_decoder_input_reference, h_reference, c_reference, Flatten_as_attention_reference])
        # normalisation output
        current_decoder_out = (current_decoder_output - (current_decoder_output_reference)**0.97)/(current_decoder_output**0.6)
        #current_decoder_out = current_decoder_output
        output_hat = np.argmax(current_decoder_out)
        output_hat = wordforid[output_hat]

        output_hatt = np.argmax(current_decoder_output)
        current_decoder_input = wordvector_dictionary[output_hatt]
        current_decoder_input = current_decoder_input.reshape(1, 1, 300)
        output_hatt = wordforid[output_hatt]

        output_hattt = np.argmax(current_decoder_output_reference)
        current_decoder_input_reference = wordvector_dictionary[output_hattt]
        current_decoder_input_reference = current_decoder_input_reference.reshape(1, 1, 300)


        if(output_hat != last_word):
            outputs_hat_list.append(output_hat)
            last_word = output_hat
            outputs_hat = outputs_hat + output_hat + ' '
            word_count+=1
            if(word_count > 10):
                break

    return label_predict, outputs_hat


def start(pic_path):

    width = 128
    height = 128

    if(isfile("Model_cat.hdf5") and isfile("Model_encoder.hdf5") and isfile("Model_decoder.hdf5") and isfile("dictionary") and isfile("wordvector_dictionary")):

        Model_cat = load_model("Model_cat.hdf5")
        Model_encoder = load_model("Model_encoder.hdf5")
        Model_decoder = load_model("Model_decoder.hdf5")

        dict = open("wordvector_dictionary", "rb")
        (wordvector_dictionary) = pickle.load(dict)
        dict.close()

        dict = open("dictionary", "rb")
        (idofword, wordforid, idofcat, catforid) = pickle.load(dict)
        dict.close()

    else:
        if (isfile("preprocess_images")):
            preprocess = open("preprocess_images", "rb")
            (images) = pickle.load(preprocess)
            preprocess.close()
        if (isfile("preprocess_captions_word2vec")):
            preprocess = open("preprocess_captions_word2vec", "rb")
            (captions_word2vec) = pickle.load(preprocess)
            preprocess.close()
        if (isfile("preprocess_captions_onehot")):
            preprocess = open("preprocess_captions_onehot", "rb")
            (captions_onehot) = pickle.load(preprocess)
            preprocess.close()
        if (isfile("preprocess_categories")):
            preprocess = open("preprocess_categories", "rb")
            (categories) = pickle.load(preprocess)
            preprocess.close()

            dict = open("dictionary", "rb")
            (idofword, wordforid, idofcat, catforid) = pickle.load(dict)
            dict.close()

            dict = open("wordvector_dictionary", "rb")
            (wordvector_dictionary) = pickle.load(dict)
            dict.close()
        else:
            captions_info_file = "annotations-2/captions_val2017.json"
            categories_info_file = "annotations-2/instances_val2017.json"

            image_load = "val2017/"

            images, categories, captions, ids = image_readin(width, height, captions_info_file, categories_info_file,
                                                             image_load)

            captions_word2vec, captions_onehot, idofword, wordforid, wordvector_dictionary = caption_processing(captions)
            categories, idofcat, catforid = cat_processing(categories)

            preprocess_save = open("preprocess_images", "wb")
            pickle.dump((images), preprocess_save)
            preprocess_save.close()

            preprocess_save = open("preprocess_captions_word2vec", "wb")
            pickle.dump((captions_word2vec), preprocess_save)
            preprocess_save.close()

            preprocess_save = open("preprocess_captions_onehot", "wb")
            pickle.dump((captions_onehot), preprocess_save)
            preprocess_save.close()

            preprocess_save = open("preprocess_categories", "wb")
            pickle.dump((categories), preprocess_save)
            preprocess_save.close()

            preprocess_save = open("wordvector_dictionary", "wb")
            pickle.dump((wordvector_dictionary), preprocess_save)
            preprocess_save.close()

            preprocess_save = open("dictionary", "wb")
            pickle.dump((idofword, wordforid, idofcat, catforid), preprocess_save)
            preprocess_save.close()


        if (isfile("Model_cat.hdf5")):
            Model_cat = load_model("Model_cat.hdf5")
        else:
            Model_cat, cat_history = model_cat(images, categories)

            save = open("category_fit", "wb")
            pickle.dump((cat_history), save)
            save.close()
            
        
        if (isfile("Model_encoder.hdf5") and isfile("Model_decoder.hdf5")):
            Model_encoder = load_model("Model_encoder.hdf5")
            Model_decoder = load_model("Model_decoder.hdf5")
        else:
            Model_encoder, Model_decoder, cap_history, overall_BLUE_Score = model_caption(images, captions_word2vec, captions_onehot, wordvector_dictionary)

            save = open("caption_fit", "wb")
            pickle.dump((cap_history, overall_BLUE_Score), save)
            save.close()


    given_img_path = pic_path

    labels, caption = cat_cap_of_img(given_img_path, width, height, catforid, wordforid, Model_cat, Model_encoder, Model_decoder, wordvector_dictionary)
    
    return labels, caption