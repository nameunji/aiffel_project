"""
자료 다운
$ wget https://aiffelstaticprd.blob.core.windows.net/media/documents/song_lyrics.zip
$ unzip song_lyrics.zip -d ~/aiffel/data/e6_lyricist/lyrics  #lyrics 폴더에 압축풀기
"""
import os
import re
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# 1. 데이터 읽어오기
txt_file_path = os.getenv('HOME') + '/aiffel/data/e6_lyricist/lyrics/*'

txt_list = glob.glob(txt_file_path)  # glob.glob() : 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환
raw_corpus = []

# 여러개의 txt 파일을 모두 읽어서 raw_corpus에 담는다
for txt_file in txt_list:
    with open(txt_file, "r") as f:
        raw = f.read().splitlines()
        raw_corpus.extend(raw)

print("데이터 크기:", len(raw_corpus))  # 187088


# 2. 데이터 정제
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()  # 소문자, 양쪽공백 제거
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)  # 특수문자 양쪽에 공백을 추가
    sentence = re.sub(r'[" "]+', " ", sentence)  # 공백 패턴을 만나면 스페이스 1개로 치환
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)  # 패턴을 제외한 모든 문자(공백문자까지도)를 스페이스 1개로 치환
    sentence = sentence.strip()  # 양쪽 공백 제거
    sentence = '<start> ' + sentence + ' <end>'
    return sentence


corpus = []  # 형태 : ['<start> i m begging of you please don t take my man <end>', ...] length - 175986
for sentence in raw_corpus:
    if len(sentence) == 0: continue
    tmp = preprocess_sentence(sentence)
    if len(tmp.split()) > 15: continue
    corpus.append(tmp)


def tokenize(corpus):
    # num_words:전체 단어의 개수, filters:별도로 전처리 로직을 추가, oov_token: out-of-vocabulary 사전에 없었던 단어는 어떤 토큰으로 대체할지
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=12000, filters=' ', oov_token="<unk>")
    tokenizer.fit_on_texts(corpus)  # corpus로부터 Tokenizer가 사전을 자동구축

    # tokenizer를 활용하여 모델에 입력할 데이터셋 구축(Tensor로 변환)
    tensor = tokenizer.texts_to_sequences(corpus)

    # 입력 데이터 시퀀스 길이 맞춰주기 - padding
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=15)

    return tensor, tokenizer


tensor, tokenizer = tokenize(corpus)

# 단어 사전이 어떻게 구축되었는지 확인 방법
for idx in tokenizer.index_word:
    print(idx, ":", tokenizer.index_word[idx])
    if idx >= 10: break


# 3. 평가데이터셋 분리
"""
x_train : 소스 문장, 형식(<start> 문장), 즉 <end>를 삭제
y_train : 타겟 문장, 형식(문장 <end>), 즉 <start>를 삭제

단어장의 크기는 12,000 이상으로 설정하세요! 총 데이터의 20%를 평가 데이터셋으로 사용해 주세요!
"""
src_input = tensor[:, :-1]  # tensor에서 마지막 토큰을 잘라내서 소스 문장을 생성. 마지막 토큰은 <end>가 아니라 <pad>일 가능성이 높다.
tgt_input = tensor[:, 1:]  # tensor에서 <start>를 잘라내서 타겟 문장을 생성 -> 문장 길이는 14가 됨

# train data를 train, valid로 나눈다.(비율 80:20) 만약 학습데이터 개수가 124960보다 크다면 위 Step 3.의 데이터 정제 과정을 다시 검토
enc_train, enc_val, dec_train, dec_val = train_test_split(src_input, tgt_input, test_size=0.2, random_state=20)
print("Source Train:", enc_train.shape)  # (124960, 14)  # 현재 (124981, 14)
print("Target Train:", dec_train.shape)  # (124960, 14)


# 4. 모델 생성
class TextGenerator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(TextGenerator, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)  # 입력된 텐서에는 단어사전의 인덱스가 들어있는데, 이 인덱스 값을 해당 인덱스 번째의 워드 벡터로 바꿔준다.
        self.rnn_1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.rnn_2 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.linear = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        out = self.embedding(x)
        out = self.rnn_1(out)
        out = self.rnn_2(out)
        out = self.linear(out)

        return out


embedding_size = 1024
hidden_size = 3000
model = TextGenerator(tokenizer.num_words + 1, embedding_size, hidden_size)

# 5. 모델 학습
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
model.compile(loss=loss, optimizer=optimizer)
model.fit(enc_train, dec_train, epochs=5, validation_data=(enc_val, dec_val))


def generate_text(model, tokenizer, init_sentence="<start>", max_len=20):
    # 테스트를 위해서 입력받은 init_sentence도 일단 텐서로 변환합니다.
    test_input = tokenizer.texts_to_sequences([init_sentence])
    test_tensor = tf.convert_to_tensor(test_input, dtype=tf.int64)
    end_token = tokenizer.word_index["<end>"]

    while True:
        predict = model(test_tensor)  # 입력받은 문장의 텐서를 입력합니다.
        predict_word = tf.argmax(tf.nn.softmax(predict, axis=-1), axis=-1)[:, -1]  # 모델이 예측한 마지막 단어가 바로 새롭게 생성한 단어가 됨

        # 모델이 새롭게 예측한 단어를 입력 문장의 뒤에 붙여줌
        test_tensor = tf.concat([test_tensor, tf.expand_dims(predict_word, axis=0)], axis=-1)

        # 모델이 <end>를 예측했거나, max_len에 도달하지 않았다면  while 루프를 또 돌면서 다음 단어를 예측
        if predict_word.numpy()[0] == end_token: break
        if test_tensor.shape[1] >= max_len: break

    generated = ""
    # 생성된 tensor 안에 있는 word index를 tokenizer.index_word 사전을 통해 실제 단어로 하나씩 변환
    for word_index in test_tensor[0].numpy():
        generated += tokenizer.index_word[word_index] + " "

    return generated


# generate_text(model, tokenizer, init_sentence="<start> he")
test_sen = generate_text(model, tokenizer, init_sentence="<start> i love", max_len=20)
print(test_sen)

"""
(256,1024, 10) loss: 1.2140 - val_loss: 2.4363 i love you too i want you
(512,1024, 10) loss: 1.1716 - val_loss: 2.4399 i love you so much
(256,512, 5) loss: 2.3629 - val_loss: 2.6401   i love you , i m a go
(1024, 1024, 5) loss: 1.7103 - val_loss: 2.3196 i love you , i love you
(256, 1024, 5) loss: 1.9214 - val_loss: 2.4038 i love you , liberian girl
(256, 1024, 8) loss: 1.3383 - val_loss: 2.3543 i love you , i love you
(128,1024,, 8) loss: 1.4533 - val_loss: 2.3919 i love you , liberian girl
(384, 1024, 8) loss: 1.3875 - val_loss: 2.3808 i love rock and roll rock and roll
(1024,1024, 10) loss: 1.1743 - val_loss: 2.4587 i love you
(1500,1024, 10) batchsize=512  loss: 2.1339 - val_loss: 2.4590
(2048, 1024, 10) loss: 1.1609 - val_loss: 2.4595
(1024,2048, 5) loss: 1.0359 - val_loss: 2.4251
(1024,2500, 5) loss: 1.3056 - val_loss: 2.2238
(1024,3000, 5) loss: 1.2697 - val_loss: 2.2280
"""