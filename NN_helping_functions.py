import numpy as np 
import tensorflow as tf
import tensorflow_probability as tfp 

import tensorflow.keras as keras 
from tensorflow.keras.layers import Dense, Input, Dropout, Lambda

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Embedding, Flatten, Concatenate
from tensorflow.keras.metrics import (RootMeanSquaredError, MeanAbsoluteError)
import tensorflow_addons as tfa 

# Define model parameters 
hidden_units = [10, 10]
negloglik = lambda y, rv_y: -rv_y.log_prob(y)

# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def create_model(X_train, learning_rate,DROPOUT_RATIO):
    #scale = tf.constant(y_train.std())
    cont_layer = Input(shape=1)
    cont_layer = Dense(1, activation="relu")(cont_layer)

    cat_layer = Input(shape=1)
    cat_layer = Dense(1, activation="relu")(cat_layer)
    embedded = Embedding(54, 10)(cat_layer)
    emb_flat = Flatten()(embedded)

    weekday_input = Input(shape = 1)
    embedded_weekday = Embedding(7, 10)(weekday_input)
    emb_flat_weekday = Flatten()(embedded_weekday)

    year_input = Input(shape=1)
    year_layer = Dense(1, activation="relu")(year_input)
    
    features = Concatenate(-1)([cont_layer, emb_flat,emb_flat_weekday, year_layer])
    for units in hidden_units:
       features = tfp.layers.DenseVariational(
        units=units,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight= 1 / len(X_train),
        activation= "sigmoid",
        )(features)

    dropout_layer = Dropout(DROPOUT_RATIO)(features)
    #output_layer = keras.layers.Dense(1)(features)

    #output_layer = Lambda (lambda x: x*scale) (features) 
    #output_layer = keras.layers.Add()([output_layer, cont_layer])
    distribution_params = keras.layers.Dense(units=2, activation="relu")(dropout_layer)
    output_layer = tfp.layers.IndependentNormal(1)(distribution_params)

    model = keras.Model(inputs=[cont_layer, cat_layer,weekday_input, year_input], outputs=output_layer)
    model.compile(loss= 'binary_crossentropy', 
                optimizer=keras.optimizers.Adam(),
    metrics=[RootMeanSquaredError(), MeanAbsoluteError()])
    return model

def get_prediction_uncertainty(model, X_test,sample):
    predictions = []
    for ix in range(sample):
        predict_ix = model.predict(X_test)
        predictions.append(predict_ix[:,0])
    predications_arr = np.array(predictions)
    prediction_mean = np.mean(predications_arr, axis=0)
    prediction_stdv = np.std(predications_arr, axis=0)
    # The 95% CI is computed as mean ± (1.96 * stdv)
    upper = (prediction_mean + (1.96 * prediction_stdv))
    lower = (prediction_mean - (1.96 * prediction_stdv))
    return prediction_mean, upper, lower 