import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf

from tensorflow.keras.layers import Input, LSTM, RepeatVector, concatenate, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model


class Seq2Seq:

    def __init__(self, window, input_dim, latent_dim, epochs, batch_size):

        self.window = window
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size


    # 学習用のLSTM-AEを定義
    def create_learning_model(self):

        # encoder
        ## 入力層
        self.encoder_input = Input(shape=(self.window, self.input_dim), name='input_x')
        ## LSTM, デコーダに状態を引き継ぐために "return_state=True" にしておく
        self.encoder = LSTM(self.latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=False, return_state=True, name='encoder')
        self.encoder_output, self.encoder_state_h, self.encoder_state_c = self.encoder(self.encoder_input) 
        ## デコーダに引き継ぐパラメータ
        self.encoder_state = [self.encoder_state_h, self.encoder_state_c]


        # decoder
        ## デコーダの入力層, 教師データを逆順で入力
        self.decoder_input = Input(shape=(self.window, self.input_dim), name='input_rev_x')
        ## LSTM, "return_sequence=True" で毎時刻の出力を取得, "initial_state=encoder_state" でエンコーダの状態を引き継ぐ
        self.decoder = LSTM(self.latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=True, return_state=True,  name='decoder')
        self.decoder_output, _, _ = self.decoder(self.decoder_input, initial_state=self.encoder_state)
        ## 出力層
        self.decoder_dense = Dense(self.input_dim, activation='linear', name='output')
        self.decoder_output = self.decoder_dense(self.decoder_output)

        # encoder と decoderを結合
        self.seq2seq = Model([self.encoder_input, self.decoder_input], self.decoder_output, name='LSTM_AE')
        self.seq2seq.compile(optimizer='adam', loss='mean_squared_error')


    # 予測用のLSTM-AEを定義
    def create_pred_model(self):

        # encoder
        ## 学習済みエンコーダを使用
        self.pred_encoder = Model(self.encoder_input, self.encoder_state, name='pred_encoder')

        # decoder
        ## 入力層と内部状態の入力を定義
        self.pred_decoder_input = Input(shape=(1, self.input_dim))
        self.pred_decoder_state_in = [Input(shape=(self.latent_dim,)), Input(shape=(self.latent_dim,))]
        ## 学習済みデコーダを使用
        self.pred_decoder_output, self.pred_decoder_state_h, self.pred_decoder_state_c = self.decoder(self.pred_decoder_input, initial_state=self.pred_decoder_state_in)
        self.pred_decoder_state = [self.pred_decoder_state_h, self.pred_decoder_state_c]
        ## 学習済みの出力層
        self.pred_decoder_output = self.decoder_dense(self.pred_decoder_output)
        # エンコーダとデコーダを結合
        self.pred_decoder = Model([self.pred_decoder_input]+self.pred_decoder_state_in, [self.pred_decoder_output]+self.pred_decoder_state, name='pred_decoder') # リストを+で結合


    # LSTMに入力するデータを作成する関数
    def create_subseq(self, data, stride, window):
        sub_seq = []
        for i in range(0, len(data)-window, stride):
            sub_seq.append(data[i:i+window])
        return sub_seq

    # 学習
    def learn(self, input, stride):

        self.create_learning_model()

        sub_seq = self.create_subseq(input, stride, self.window)

        # 入力のために小分けにしたデータをさらに整形
        sub_seq = np.array(sub_seq)
        sub_seq = sub_seq.reshape(sub_seq.shape[0], sub_seq.shape[1], 1)

        # デコーダの入力用に各sub sequenceを逆順にしたものを作成
        rev_sub_seq = sub_seq[::, ::-1, ::]

        self.history = self.seq2seq.fit([sub_seq, rev_sub_seq],sub_seq, epochs=self.epochs, batch_size=self.batch_size, verbose=True)
    
        self.create_pred_model()

    def pred(self, data):

        # 正常データを推論
        sub_seq = self.create_subseq(data, self.window, self.window)
        sub_seq = np.array(sub_seq)
        sub_seq = sub_seq.reshape(sub_seq.shape[0], sub_seq.shape[1], 1)
        
        predicted = []

        for i in tqdm(range(len(sub_seq))):

            state_value = self.pred_encoder.predict(sub_seq[i:i+1], verbose=0)
            decoder_output_value = np.zeros((1,1,1))

            for i in range(self.window):
                input = [decoder_output_value] + state_value
                output, h, c = self.pred_decoder.predict(input, verbose=0)
                extracted_output = output[0][0][0]
                predicted.append(extracted_output)
                state_value = [h, c] # 次に渡す内部状態

        return predicted
    