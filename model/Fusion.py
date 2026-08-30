
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model as Model
from tensorflow.keras import regularizers as regularizers
import utils
import os
import h5py

K.set_floatx('float32')

class Fusion():
    def __init__(self, time_step, size_rnn, size_nn, size_layer_chfn, size_layer_cmfn, M0pred, seed):
        self.time_step = time_step
        self.size_rnn = size_rnn
        self.size_nn = size_nn
        self.size_layer_chfn = size_layer_chfn
        self.size_layer_cmfn = size_layer_cmfn
        self.M0pred = M0pred

        self.seed = seed

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def collate_times(self, dT_dat, dM_dat, Mtarget):
        k = 0
        dT_dat_targets = np.ones_like(dT_dat)
        mask = np.ones_like(dT_dat, dtype=bool)

        for i in range(len(dT_dat)):
            if(dM_dat[i] >= Mtarget):
                dT_dat_targets[i] = sum(dT_dat[i - k:i + 1])
                k = 0
            else:
                dT_dat_targets[i] = sum(dT_dat[i - k:i + 1])
                k += 1
                mask[i] = False
                    
        return dT_dat_targets, mask
    
    def set_train_data(self, times, mags, etas_time_intg_full, etas_mag_intg_full, etas_time_rate_full, etas_mag_rate_full):
        self.T_train = times
        self.M_train = mags
        self.etas_time_train = etas_time_intg_full
        self.etas_mag_train = etas_mag_intg_full
        
        dM_train = np.delete(mags, 0)
        dT_train = np.ediff1d(times)

        etas_time_intg = etas_time_intg_full[1:]
        etas_mag_intg = etas_mag_intg_full[1:]

        etas_time_rate = etas_time_rate_full[1:]
        etas_mag_rate = etas_mag_rate_full[1:]

        dT_targets, mask = self.collate_times(dT_train, dM_train, self.M0pred)
        
        n_windows = len(dT_train) - self.time_step
        input_RNN_times = []
        input_RNN_mags = []
        input_CHFN = []
        input_CMFN = []
        train_mask = []

        etas_time_intg_inputs = []
        etas_mag_intg_inputs = []

        etas_time_rate_inputs = []
        etas_mag_rate_inputs = []

        for i in range(n_windows):
            input_RNN_times.append(dT_train[i:i + self.time_step])
            input_RNN_mags.append(dM_train[i:i + self.time_step])

            input_CHFN.append(dT_train[i + self.time_step])
            input_CMFN.append(dM_train[i + self.time_step])
            train_mask.append(mask[i + self.time_step])

            etas_time_intg_inputs.append(etas_time_intg[i + self.time_step])
            etas_mag_intg_inputs.append(etas_mag_intg[i + self.time_step])

            etas_time_rate_inputs.append(etas_time_rate[i + self.time_step])
            etas_mag_rate_inputs.append(etas_mag_rate[i + self.time_step])

        self.input_RNN_times = np.expand_dims(np.array(input_RNN_times).astype(np.float32), axis=-1) 
        self.input_RNN_mags = np.expand_dims(np.array(input_RNN_mags).astype(np.float32), axis=-1) 

        self.input_CHFN = np.array(input_CHFN).reshape(-1, 1).astype(np.float32)
        self.input_CMFN = np.array(input_CMFN).reshape(-1, 1).astype(np.float32)
        self.train_mask = np.array(train_mask, dtype=np.float32).reshape(-1, 1)

        self.etas_time_intg_inputs = np.array(etas_time_intg_inputs).reshape(-1, 1).astype(np.float32)
        self.etas_mag_intg_inputs = np.array(etas_mag_intg_inputs).reshape(-1, 1).astype(np.float32)
        
        self.etas_time_rate_inputs = np.array(etas_time_rate_inputs).reshape(-1, 1).astype(np.float32)
        self.etas_mag_rate_inputs = np.array(etas_mag_rate_inputs).reshape(-1, 1).astype(np.float32)

        return self
    
    def set_model(self, lam, batch_size=None, stateful=False):
        # =========================
        #   Statistics for normalization
        # =========================
        mu = np.log(np.ediff1d(self.T_train)).mean()
        sigma = np.log(np.ediff1d(self.T_train)).std()
        mu1 = np.log(self.M_train).mean()
        sigma1 = np.log(self.M_train).std()

        mu_et = self.etas_time_train[1:].mean()
        sigma_et = self.etas_time_train[1:].std() + 1e-10
        mu_em = self.etas_mag_train[1:].mean()
        sigma_em = self.etas_mag_train[1:].std() + 1e-10

        def abs_glorot_uniform(shape, dtype=None, partition_info=None):
            return K.abs(keras.initializers.glorot_uniform(seed=None)(shape, dtype=dtype))
        
        # =========================
        #   Inputs
        # =========================
        time_history = layers.Input(shape=(self.time_step, 1), dtype=tf.float32, batch_size=batch_size)
        mag_history = layers.Input(shape=(self.time_step, 1), dtype=tf.float32,batch_size=batch_size)
        elapsed_time = layers.Input(shape=(1, ), dtype=tf.float32, batch_size=batch_size) 
        current_mag = layers.Input(shape=(1, ), dtype=tf.float32, batch_size=batch_size) 
        mask = layers.Input(shape=(1, ), dtype=tf.float32, batch_size=batch_size)

        etas_time_intg_input = layers.Input(shape=(1, ), dtype=tf.float32, batch_size=batch_size) 
        etas_mag_intg_input = layers.Input(shape=(1, ), dtype=tf.float32, batch_size=batch_size)

        etas_time_rate_input = layers.Input(shape=(1,), dtype=tf.float32, batch_size=batch_size) 
        etas_mag_rate_input  = layers.Input(shape=(1,), dtype=tf.float32, batch_size=batch_size) 

        # =========================
        #   Normalization
        # =========================
        elapsed_time_nmlz = layers.Lambda(lambda x: (K.log(x + 1e-10) - mu) / sigma)(elapsed_time)
        time_history_nmlz = layers.Lambda(lambda x: (K.log(x + 1e-10) - mu) / sigma)(time_history)
        mag_history_nmlz = layers.Lambda(lambda x: (K.log(x + 1e-10) - mu1) / sigma1)(mag_history)
        event_history_nmlz = layers.Concatenate(axis=2)([time_history_nmlz, mag_history_nmlz])

        current_mag_nmlz = layers.Lambda(lambda x: (x - self.M0pred))(current_mag)

        etas_time_intg_nmlz = layers.Lambda(lambda x: (x - mu_et) / sigma_et)(etas_time_intg_input)
        etas_mag_intg_nmlz = layers.Lambda(lambda x: (x - mu_em) / sigma_em)(etas_mag_intg_input)

        # =========================
        #   RNN encoder (history → hidden state)
        # =========================
        output_rnn = layers.LSTM(self.size_rnn, 
                                 input_shape=(self.time_step, 2), 
                                 activation='tanh', 
                                 stateful=stateful
                                 )(event_history_nmlz)

        # =========================
        #   CHFN time NN cumulative Λ_NN^(t)(τ)
        # =========================
        hidden_tau = layers.Dense(self.size_nn, 
                                  kernel_initializer=abs_glorot_uniform, 
                                  kernel_constraint=keras.constraints.NonNeg(), 
                                  use_bias=False, 
                                  kernel_regularizer=regularizers.l2(lam)
                                  )(elapsed_time_nmlz) 
        hidden_rnn = layers.Dense(self.size_nn, 
                                  kernel_regularizer=regularizers.l2(lam)
                                  )(output_rnn) 
        hidden_etas_time = layers.Dense(self.size_nn, 
                                       kernel_initializer=abs_glorot_uniform,
                                       kernel_constraint=keras.constraints.NonNeg(),
                                        )(etas_time_intg_nmlz)
        
        hidden = layers.Lambda(lambda inputs: K.tanh(inputs[0] + inputs[1] + inputs[2])
                               )([hidden_tau, hidden_rnn, hidden_etas_time])

        for _ in range(self.size_layer_chfn - 1):
            hidden = layers.Dense(self.size_nn, 
                                  activation='tanh', 
                                  kernel_initializer=abs_glorot_uniform, 
                                  kernel_constraint=keras.constraints.NonNeg(),kernel_regularizer=regularizers.l2(lam)
                                  )(hidden) 
            
        # =========================
        #   CMFN magnitude NN cumulative Λ_NN^(m)(m)
        # =========================
        hidden_mu = layers.Dense(self.size_nn,
                                 kernel_initializer=abs_glorot_uniform,
                                 kernel_constraint=keras.constraints.NonNeg(),
                                 use_bias=False,
                                 activation='relu'
                                 )(current_mag_nmlz) 
        
        hidden_rnn_mag = layers.Dense(self.size_nn)(output_rnn) 

        hidden_etas_mag = layers.Dense(self.size_nn,
                                      kernel_initializer=abs_glorot_uniform,
                                      kernel_constraint=keras.constraints.NonNeg(),
                                        )(etas_mag_intg_nmlz)

        hidden_mag = layers.Lambda(lambda inputs: K.tanh(inputs[0] + inputs[1] + inputs[2] + inputs[3])
                                   )([hidden_mu, hidden_rnn_mag, hidden_tau, hidden_etas_mag])

        for _ in range(self.size_layer_cmfn - 1):
            hidden_mag = layers.Dense(self.size_nn,
                                      activation='tanh',
                                      kernel_initializer=abs_glorot_uniform,
                                      kernel_constraint=keras.constraints.NonNeg()
                                      )(hidden_mag) 
        

        # === NN intg ===
        Int_l_time = layers.Dense(1,
                           activation='softplus',
                           kernel_initializer=abs_glorot_uniform,
                           kernel_constraint=keras.constraints.NonNeg(),
                           name='Int_l_time')(hidden)   # Λ_NN^(t)(τ)

        Int_l_mag = layers.Dense(1,
                          activation='sigmoid',
                          kernel_initializer=abs_glorot_uniform,
                          kernel_constraint=keras.constraints.NonNeg(),
                          name='Int_l_mag')(hidden_mag) # Λ_NN^(m)(m)
        
        # ============================================================
        #   自动微分：把 NN 部分和 ETAS 部分拆开
        # ============================================================

        # ---------- 时间方向 ----------
        l_time_nn = layers.Lambda(
            lambda inputs: K.gradients(inputs[0], inputs[1])[0],
            name='l_time_nn'
        )([Int_l_time, elapsed_time])

        d_Int_time_d_etas = layers.Lambda(
            lambda inputs: K.gradients(inputs[0], inputs[1])[0],
            name='d_Int_time_d_etas'
        )([Int_l_time, etas_time_intg_nmlz])

        l_time_etas = layers.Lambda(
            lambda inputs: (1.0 / sigma_et) * inputs[0] * inputs[1],
            name='l_time_etas'
        )([d_Int_time_d_etas, etas_time_rate_input])

        l_time_rate = layers.Add(name='l_time_total')([l_time_nn, l_time_etas])

        # ---------- 震级方向 ----------
        l_mag_nn = layers.Lambda(
            lambda inputs: K.gradients(inputs[0], inputs[1])[0],
            name='l_mag_nn'
        )([Int_l_mag, current_mag])

        d_Int_mag_d_etas = layers.Lambda(
            lambda inputs: K.gradients(inputs[0], inputs[1])[0],
            name='d_Int_mag_d_etas'
        )([Int_l_mag, etas_mag_intg_nmlz])

        l_mag_etas = layers.Lambda(
            lambda inputs: (1.0 / sigma_em) * inputs[0] * inputs[1],
            name='l_mag_etas'
        )([d_Int_mag_d_etas, etas_mag_rate_input])

        l_mag_rate = layers.Add(name='l_mag_total')([l_mag_nn, l_mag_etas])

        # =========================
        #   Build model
        # =========================

        self.model = Model(
            inputs = [time_history, mag_history, 
                      elapsed_time, current_mag, 
                      mask, 
                      etas_time_intg_input, etas_mag_intg_input, 
                      etas_time_rate_input, etas_mag_rate_input],   
            outputs = [l_time_rate, Int_l_time, 
                       l_mag_rate, Int_l_mag]
        )

        # =========================
        #   Log-likelihood loss
        # =========================
        def log_likelihood_loss(Int_l_time, l_time_rate, Int_l_mag, l_mag_rate, mask):
            eps = 1e-10

            ll_time = tf.math.log(l_time_rate + eps) * mask - Int_l_time
            ll_mag = tf.math.log(l_mag_rate + eps) * mask
            ll_total = ll_time + ll_mag
            return -tf.reduce_sum(ll_total)

        loss = log_likelihood_loss(Int_l_time, l_time_rate, Int_l_mag, l_mag_rate, mask)
        self.model.add_loss(loss)

        return self
    
    def compile(self, lr=1e-3):
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(optimizer=self.optimizer)
        return self
    
    def fit_eval(self, epochs=100, batch_size=256, plot_training=True, validation_split=0.2, callbacks=None):
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

        history = self.model.fit(
            x=[
                self.input_RNN_times, 
                self.input_RNN_mags,
                self.input_CHFN, 
                self.input_CMFN, 
                self.train_mask, 
                self.etas_time_intg_inputs, 
                self.etas_mag_intg_inputs,
                self.etas_time_rate_inputs,
                self.etas_mag_rate_inputs
            ],
            y=None, 
            batch_size=batch_size,
            epochs = epochs,
            validation_split = validation_split,
            callbacks=[es],
            verbose=1,
            shuffle=False
        )

        self.best_val_loss = np.min(history.history['val_loss'])

        if plot_training:
            plt.plot(history.history['loss'], label="Training Loss")
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label="Validation Loss")
            plt.title("Loss Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.show()

        return self
    
    def save_weights(self, ckp_path, file_path):
        path = os.path.join(ckp_path, file_path)
        self.model.save_weights(path)
        return self 
    
    def load_weights(self, ckp_path, file_path):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        checkpoint_path = os.path.join(ckp_path, file_path)

        with h5py.File(checkpoint_path, 'r') as f:
            keras_version = f.attrs['keras_version']
            if isinstance(keras_version, bytes):
                keras_version = keras_version.decode('utf-8')                
            print(f"Loaded keras_version from weights: {keras_version}")

        self.model.load_weights(checkpoint_path)
        return self
    
    def set_test_data(self, test_times, test_mags, test_etas_time_intg_full, test_etas_mag_intg_full, test_etas_time_rate_full, test_etas_mag_rate_full):
        self.T_test = test_times
        self.M_test = test_mags
        self.etas_time_test = test_etas_time_intg_full
        self.etas_mag_test = test_etas_mag_intg_full
        
        dM_test = np.delete(test_mags, 0)
        dT_test = np.ediff1d(test_times)

        test_etas_time_intg = test_etas_time_intg_full[1:]
        test_etas_mag_intg = test_etas_mag_intg_full[1:]

        test_etas_time_rate = test_etas_time_rate_full[1:]
        test_etas_mag_rate = test_etas_mag_rate_full[1:]

        dT_test_targets, mask = self.collate_times(dT_test, dM_test, self.M0pred)
        
        n_windows_test = len(dT_test) - self.time_step
        input_RNN_times_test = []
        input_RNN_mags_test = []
        input_CHFN_test = []
        input_CMFN_test = []
        test_mask = []

        test_etas_time_intg_inputs = []
        test_etas_mag_intg_inputs = []

        test_etas_time_rate_inputs = []
        test_etas_mag_rate_inputs = []

        for i_test in range(n_windows_test):
            input_RNN_times_test.append(dT_test[i_test:i_test + self.time_step])
            input_RNN_mags_test.append(dM_test[i_test:i_test + self.time_step])

            input_CHFN_test.append(dT_test[i_test + self.time_step])
            input_CMFN_test.append(dM_test[i_test + self.time_step])
            test_mask.append(mask[i_test + self.time_step])

            test_etas_time_intg_inputs.append(test_etas_time_intg[i_test + self.time_step])
            test_etas_mag_intg_inputs.append(test_etas_mag_intg[i_test + self.time_step])

            test_etas_time_rate_inputs.append(test_etas_time_rate[i_test + self.time_step])
            test_etas_mag_rate_inputs.append(test_etas_mag_rate[i_test + self.time_step])

        self.input_RNN_times_test = np.expand_dims(np.array(input_RNN_times_test).astype(np.float32), axis=-1)
        self.input_RNN_mags_test = np.expand_dims(np.array(input_RNN_mags_test).astype(np.float32), axis=-1) 

        self.input_CHFN_test = np.array(input_CHFN_test).reshape(-1, 1).astype(np.float32)
        self.input_CMFN_test = np.array(input_CMFN_test).reshape(-1, 1).astype(np.float32)
        self.test_mask = np.array(test_mask, dtype=np.float32).reshape(-1, 1)

        self.test_etas_time_intg_inputs = np.array(test_etas_time_intg_inputs).reshape(-1, 1).astype(np.float32)
        self.test_etas_mag_intg_inputs = np.array(test_etas_mag_intg_inputs).reshape(-1, 1).astype(np.float32)

        self.test_etas_time_rate_inputs = np.array(test_etas_time_rate_inputs).reshape(-1, 1).astype(np.float32)
        self.test_etas_mag_rate_inputs = np.array(test_etas_mag_rate_inputs).reshape(-1, 1).astype(np.float32)

        return self
    
    def collate_likelihoods(self, Lvec, Boolvec):
        collated_vec = np.zeros(np.count_nonzero(Boolvec))
        count = 0
        k = 0

        for i in range(len(Lvec)):
            if Boolvec[i]:
                collated_vec[count] = sum(Lvec[i - k:i + 1])
                k = 0
                count += 1
            else:
                k += 1
        return collated_vec
    
    def predict_eval(self, batch_size=512):
        self.fusion_rate_time, self.fusion_intg_time, self.fusion_rate_mag, self.fusion_intg_mag = self.model.predict(
            [
                self.input_RNN_times_test,
                self.input_RNN_mags_test,
                self.input_CHFN_test,
                self.input_CMFN_test,
                self.test_mask,
                self.test_etas_time_intg_inputs,
                self.test_etas_mag_intg_inputs,
                self.test_etas_time_rate_inputs,
                self.test_etas_mag_rate_inputs
            ],
            batch_size=batch_size
        )

        print("fusion_rate_time: any NaN?", np.isnan(self.fusion_rate_time).any(),
            "min:", np.nanmin(self.fusion_rate_time),
            "max:", np.nanmax(self.fusion_rate_time))

        print("fusion_intg_time: any NaN?", np.isnan(self.fusion_intg_time).any(),
            "min:", np.nanmin(self.fusion_intg_time),
            "max:", np.nanmax(self.fusion_intg_time))


        eps = 1e-10
        self.LLtime = (tf.math.log(self.fusion_rate_time + eps) * self.test_mask - self.fusion_intg_time)

        self.LLmag = tf.math.log(self.fusion_rate_mag + eps) * self.test_mask

        LLmag_full = tf.math.log(self.fusion_rate_mag + eps)
        self.collated_LLmag = tf.boolean_mask(LLmag_full, self.test_mask)

        self.collated_LLtime = self.collate_likelihoods(self.LLtime, self.test_mask)
        
        return self
    
    def summary(self):
        """ summary of the model """
        return self.model.summary()
    
    def magreshape(self, T, M, new_time, exchange_mag, new_etas_time_intg_full, new_etas_mag_intg_full, new_etas_time_rate_full, new_etas_mag_rate_full):
        T_test = np.copy(T)
        T_test = np.append(T_test, new_time)

        M_test = np.copy(M)
        M_test = np.append(M_test, exchange_mag)
        assert T_test.shape[0] == M_test.shape[0], 'add new events error!'

        dT = np.ediff1d(T_test)
        dM = np.delete(M_test, 0)
        
        new_etas_time_intg = new_etas_time_intg_full[1:]
        new_etas_mag_intg = new_etas_mag_intg_full[1:]

        new_etas_time_rate = new_etas_time_rate_full[1:]
        new_etas_mag_rate = new_etas_mag_rate_full[1:]

        n = dT.shape[0]
        n2 = dM.shape[0]
        assert n == n2, "time test data != mag test data"

        _, mask_full = self.collate_times(dT, dM, self.M0pred)

        n_windows = len(dM) - self.time_step
        input_RNN_times = []
        input_RNN_mags = []
        input_CHFN = []
        input_CMFN = []
        # mask = []

        new_etas_time_intg_inputs = []
        new_etas_mag_intg_inputs = []
        
        new_etas_time_rate_inputs = []
        new_etas_mag_rate_inputs = []

        for i in range(n_windows):
            input_RNN_times.append(dT[i:i + self.time_step])
            input_RNN_mags.append(dM[i:i + self.time_step])
            input_CHFN.append(dT[i + self.time_step])
            input_CMFN.append(dM[i + self.time_step])
            # mask.append(mask_full[i + self.time_step])

            new_etas_time_intg_inputs.append(new_etas_time_intg[i + self.time_step])
            new_etas_mag_intg_inputs.append(new_etas_mag_intg[i + self.time_step])

            new_etas_time_rate_inputs.append(new_etas_time_rate[i + self.time_step])
            new_etas_mag_rate_inputs.append(new_etas_mag_rate[i + self.time_step])

        reshape_RNN_times = np.expand_dims(np.array(input_RNN_times), axis=-1)
        reshape_RNN_mags = np.expand_dims(np.array(input_RNN_mags), axis=-1)
        reshape_CHFN = np.array(input_CHFN).reshape(-1, 1)
        reshape_CMFN = np.array(input_CMFN).reshape(-1, 1)
        # reshape_mask = np.array(mask).reshape(-1, 1)
        reshape_mask = np.ones_like(input_CMFN)

        reshape_etas_time_intg_inputs = np.array(new_etas_time_intg_inputs).reshape(-1, 1)
        reshape_etas_mag_intg_inputs = np.array(new_etas_mag_intg_inputs).reshape(-1, 1)
        
        reshape_etas_time_rate_inputs = np.array(new_etas_time_rate_inputs).reshape(-1, 1)
        reshape_etas_mag_rate_inputs = np.array(new_etas_mag_rate_inputs).reshape(-1, 1)

        return reshape_RNN_times, reshape_RNN_mags, reshape_CHFN, reshape_CMFN, reshape_mask, reshape_etas_time_intg_inputs, reshape_etas_mag_intg_inputs, reshape_etas_time_rate_inputs, reshape_etas_mag_rate_inputs
    
    def magdensfunc(self, reshape_RNN_times, reshape_RNN_mags, reshape_CHFN, reshape_CMFN, reshape_mask, reshape_etas_time_intg, reshape_etas_mag_intg, reshape_etas_time_rate, reshape_etas_mag_rate):
        fusion_l_time_pred, fusion_time_intg_pred, fusion_l_mag_pred, fusion_mag_intg_pred = self.model.predict(
            [
                reshape_RNN_times, 
                reshape_RNN_mags, 
                reshape_CHFN, 
                reshape_CMFN,
                reshape_mask,
                reshape_etas_time_intg, 
                reshape_etas_mag_intg,
                reshape_etas_time_rate,
                reshape_etas_mag_rate
            ],
            batch_size = reshape_RNN_times.shape[0]
        )
        return fusion_l_mag_pred[-1], fusion_mag_intg_pred[-1]
    #     # return fusion_l_mag_pred[-1]
