#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('wget http://nlp.stanford.edu/data/glove.42B.300d.zip')
get_ipython().system('unzip glove.42B.300d.zip')
get_ipython().system('rm glove.42B.300d.zip')


# In[ ]:


from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3 as inception
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import tensorflow.keras.preprocessing.image as tf_image
import pandas as pd
import numpy as np


# In[ ]:


encode_model = InceptionV3(weights='imagenet')
encode_model = Model(encode_model.input, encode_model.layers[-2].output)
WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048
START = "startseq"
STOP = "endseq"
EPOCHS = 10
preprocess_input = inception.preprocess_input


# In[ ]:


def encodeImage(img):
    img = img.resize((WIDTH, HEIGHT))
    x = tf_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = encode_model.predict(x) # Get the encoding vector for the image
    x = np.reshape(x, OUTPUT_DIM )

    return x


# In[ ]:


img=tf_image.load_img('/home/db21160/Desktop/images/1.jpg', target_size=(299,299))
encodeImage(img)


# In[ ]:



data=pd.read_csv('/home/db21160/Desktop/image_caption_map.csv')


# In[ ]:


data.head(2)


# In[ ]:


data['caption']=data['caption'].apply(lambda x:START+' '+x+' '+STOP)


# In[ ]:


data.shape[0]


# In[ ]:


remove_these=[]
encoded_images={}
for i in range(data.shape[0]):
    image_path='/home/db21160/Desktop/images/'+data['file_name'][i]
    print(image_path)
    try:
        img = tf_image.load_img(image_path, target_size=(HEIGHT, WIDTH))
        encoded_images[int(data['file_name'][i].split('.')[0])] = encodeImage(img)
    except:
        print('remove: ',i)
        remove_these.append(data['file_name'][i])
# some of the images are not in supported format
# we'll exclude thos


# In[ ]:


len(remove_these)


# In[ ]:


data=data[-data['file_name'].isin(remove_these)]


# In[ ]:


data.shape


# In[ ]:


data


# In[ ]:


data.reset_index(drop=True,inplace=True)


# In[ ]:


data['id']=[int(data['file_name'][i].split('.')[0]) for i in range(data.shape[0])]


# In[ ]:


len(encoded_images)


# In[ ]:


# clean up captions


# In[ ]:


from string import punctuation
import re
from nltk import word_tokenize
import nltk
nltk.download('punkt')


# In[ ]:


punctuation


# In[ ]:


data['caption']=data['caption'].apply(lambda x:re.sub("["+punctuation+"]",' ',x))


# In[ ]:


data['caption']=data['caption'].apply(lambda x:re.sub("\d",' ',x))


# In[ ]:


data['caption']=data['caption'].apply(lambda x:re.sub("\s+",' ',x))


# In[ ]:


data['caption']=data['caption'].str.lower()


# In[ ]:


data['caption']


# In[ ]:


word_count_threshold = 5
word_counts = {}
for caption in data['caption']:
    for w in word_tokenize(caption):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d ==> %d' % (len(word_counts), len(vocab)))


# In[ ]:


caption_lens=[]
for caption in data['caption']:
    words=word_tokenize(caption)
    words=[w for w in words if w in vocab]
    caption_lens.append(len(words))
max_length=max(caption_lens)


# In[ ]:


idxtoword = {}
wordtoidx = {}

ix = 1
for w in vocab:
    wordtoidx[w] = ix
    idxtoword[ix] = w
    ix += 1

vocab_size = len(idxtoword) + 1
vocab_size


# In[ ]:


max_length=max(caption_lens)


# In[ ]:


max_length


# In[ ]:



embeddings_index = {}
f = open( 'glove.42B.300d.txt', encoding="utf-8")

for line in f:
    line=line.strip()
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

f.close()
print(f'Found {len(embeddings_index)} word vectors.')


# In[ ]:


embedding_dim = 300

embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoidx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[ ]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization,add
from tqdm import tqdm


# In[ ]:


data.shape


# In[ ]:


def data_generator(data, encoded_images, wordtoidx, max_length, num_photos_per_batch):
  # x1 - Training data for photos
  # x2 - The caption that goes with each photo
  # y - The predicted rest of the caption
  x1, x2, y = [], [], []
  n=0
  while True:
    for k,caption in enumerate(data['caption']):
      n+=1
      photo = encoded_images[data['id'][k]]
      # Each photo has 5 descriptions
      seq = [wordtoidx[word] for word in word_tokenize(caption) if word in wordtoidx]
        # Generate a training case for every possible sequence and outcome
      for i in range(1, len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        x1.append(photo)
        x2.append(in_seq)
        y.append(out_seq)
      if n==num_photos_per_batch:
        yield ([np.array(x1), np.array(x2)], np.array(y))
        # next when generator gets called iteration will start from where we left off
        # this makes it make a pass through the complete data in an epoch
        x1, x2, y = [], [], []
        n=0


# In[ ]:


len(encoded_images.keys())


# In[ ]:


data.shape


# In[ ]:


# data_generator(data, encoded_images, wordtoidx, max_length, 1)


# In[ ]:


inputs1 = Input(shape=(OUTPUT_DIM,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)


# In[ ]:


caption_model.layers[2].set_weights([embedding_matrix])
caption_model.layers[2].trainable = False
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[ ]:


number_pics_per_batch = 3
steps = len(data['caption'])//number_pics_per_batch


# In[ ]:


encoded_images[99].shape


# In[ ]:


for i in range(EPOCHS*2):
  generator = data_generator(data, encoded_images, wordtoidx, max_length, number_pics_per_batch)
  caption_model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)





# In[ ]:


caption_model.optimizer.lr = 1e-4
number_pics_per_batch = 6
steps = len(data['caption'])//number_pics_per_batch

for i in tqdm(range(EPOCHS)):
    generator = data_generator(data, encoded_images, wordtoidx, max_length, number_pics_per_batch)
    caption_model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)


# In[ ]:


caption_model.save_weights('caption_model.hdf5')


# In[ ]:


def generateCaption(photo):
    in_text = START
    for i in range(max_length):
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idxtoword[yhat]
        in_text += ' ' + word
        if word == STOP:
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


# In[ ]:


index=5  #79
image_file=data.iloc[index,2]

image=encoded_images[int(image_file.split('.')[0])]

image = image.reshape((1,OUTPUT_DIM))
x=plt.imread('/content/drive/MyDrive/collab_mount/image captioning/images/'+image_file)
plt.imshow(x)
plt.show()
print("Caption:",generateCaption(image))


# In[ ]:


index=287 #79
image_file=data.iloc[index,2]

image=encoded_images[int(image_file.split('.')[0])]

image = image.reshape((1,OUTPUT_DIM))
x=plt.imread('/content/drive/MyDrive/collab_mount/image captioning/images/'+image_file)
plt.imshow(x)
plt.show()
print("Caption:",generateCaption(image))


# In[ ]:


index=185  #79
image_file=data.iloc[index,2]

image=encoded_images[int(image_file.split('.')[0])]

image = image.reshape((1,OUTPUT_DIM))
x=plt.imread('/content/drive/MyDrive/collab_mount/image captioning/images/'+image_file)
plt.imshow(x)
plt.show()
print("Caption:",generateCaption(image))


# In[ ]:


index=549  #79
image_file=data.iloc[index,2]

image=encoded_images[int(image_file.split('.')[0])]

image = image.reshape((1,OUTPUT_DIM))
x=plt.imread('/content/drive/MyDrive/collab_mount/image captioning/images/'+image_file)
plt.imshow(x)
plt.show()
print("Caption:",generateCaption(image))


# In[ ]:


caption_model.optimizer.lr = 1e-4
number_pics_per_batch = 10
steps = len(data['caption'])//number_pics_per_batch

for i in tqdm(range(EPOCHS)):
    generator = data_generator(data, encoded_images, wordtoidx, max_length, number_pics_per_batch)
    caption_model.fit_generator(generator, epochs=30, steps_per_epoch=steps, verbose=1)


# In[ ]:


index=549  #79
image_file=data.iloc[index,2]

image=encoded_images[int(image_file.split('.')[0])]

image = image.reshape((1,OUTPUT_DIM))
x=plt.imread('/content/drive/MyDrive/collab_mount/image captioning/images/'+image_file)
plt.imshow(x)
plt.show()
print("Caption:",generateCaption(image))


# In[ ]:


index=185  #79
image_file=data.iloc[index,2]

image=encoded_images[int(image_file.split('.')[0])]

image = image.reshape((1,OUTPUT_DIM))
x=plt.imread('/content/drive/MyDrive/collab_mount/image captioning/images/'+image_file)
plt.imshow(x)
plt.show()
print("Caption:",generateCaption(image))


# In[ ]:


index=287 #79
image_file=data.iloc[index,2]

image=encoded_images[int(image_file.split('.')[0])]

image = image.reshape((1,OUTPUT_DIM))
x=plt.imread('/content/drive/MyDrive/collab_mount/image captioning/images/'+image_file)
plt.imshow(x)
plt.show()
print("Caption:",generateCaption(image))


# In[ ]:


index=79
image_file=data.iloc[index,2]

image=encoded_images[int(image_file.split('.')[0])]

image = image.reshape((1,OUTPUT_DIM))
x=plt.imread('/content/drive/MyDrive/collab_mount/image captioning/images/'+image_file)
plt.imshow(x)
plt.show()
print("Caption:",generateCaption(image))


# In[ ]:


index=1500
image_file=data.iloc[index,2]

image=encoded_images[int(image_file.split('.')[0])]

image = image.reshape((1,OUTPUT_DIM))
x=plt.imread('/content/drive/MyDrive/collab_mount/image captioning/images/'+image_file)
plt.imshow(x)
plt.show()
print("Caption:",generateCaption(image))


# In[ ]:


from tensorflow.keras.models import model_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
json_file.write(model_json)
model.save_weights("model.h5")
# loading the model
from tensorflow.keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.summary()

