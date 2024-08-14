
# pylint: disable=import-error
import pandas as pd
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def load_data():
    """Load dataset from CSV file and split into training and testing sets."""

    dataf = pd.read_csv(
        './aigeneateddataset/labeled_data.csv'
    )
    trains_text, tests_text, trains_labels, tests_labels = train_test_split(
        dataf['message'],
        dataf['class'],
        test_size=0.2
    )

    return trains_text, tests_text, trains_labels, tests_labels


def tokenize_data(trained_text, tested_text):
    """Tokenize text data using Tokenizer from Keras and pad sequences to a fixed length."""

    tokenizer_data = Tokenizer(
        num_words=5000
    )
    tokenizer_data.fit_on_texts(
        trained_text
    )
    trained_sequences = pad_sequences(
        tokenizer_data.texts_to_sequences(trained_text),
        maxlen=100
    )
    tested_sequences = pad_sequences(
        tokenizer_data.texts_to_sequences(tested_text),
        maxlen=100
    )

    return tokenizer_data, trained_sequences, tested_sequences


def define_model():
    """Define a convolutional neural network model using Keras."""

    input_layer = Input(
        shape=(100,)
    )
    embedding_layer = Embedding(
        input_dim=5000,
        output_dim=50
    )(input_layer)
    conv_layer = Conv1D(
        filters=128,
        kernel_size=5,
        activation='relu'
    )(embedding_layer)
    pooling_layer = MaxPooling1D(pool_size=5)(conv_layer)
    flatten_layer = Flatten()(pooling_layer)
    output_layer = Dense(
        units=1,
        activation='sigmoid'
    )(flatten_layer)
    model_defined = Model(
        inputs=input_layer,
        outputs=output_layer
    )
    model_defined.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model_defined


def train_model(model_training,sequences_training,labels_training,sequences_testing,labels_testing):
    """Train the convolutional neural network model."""

    model_training.fit(
        sequences_training,
        labels_training,
        epochs=1,
        batch_size=1, # Works better, dont be a puss
        validation_data=(sequences_testing, labels_testing)
    )


def save_model(saved_model):
    """Save the trained convolutional neural network model to a file."""

    saved_model.save(
        'Models/ai_detection_model.h5'
    )


def classify_input(model_classify, tokenizer_classify,user_input):


        sequence = pad_sequences(
            tokenizer_classify.texts_to_sequences([user_input]), 
            maxlen=100
        )
        prediction = model_classify.predict(sequence)[0][0]

        if prediction > 0.5:
            return("Yes")
        else:
            return("No")

def aitext(intxt):
    train_text, test_text, train_labels, test_labels = load_data()
    tokenizer, train_sequences, test_sequences = tokenize_data(
        train_text,
        test_text
    )
    model = define_model()
    train_model(model,
                train_sequences,
                train_labels,
                test_sequences,
                test_labels
    )
    save_model(
        model
    )
    #inputtext="Chips Ahoy is a beloved cookie brand that has been a favorite among cookie lovers for decades. The cookies are known for their crispy texture and generous amount of chocolate chips which are baked into each cookie to provide a delicious and satisfying flavor. Chips Ahoy has become a household name and is enjoyed by people of all ages"
    result=classify_input(
        model,
        tokenizer,
        intxt
    )
    return(result)

#print(aitext("you can't tell me what CSRF is"))
