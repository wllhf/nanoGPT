import tensorflow as tf
import tensorflow.keras as keras


class BigramLanguageModel(keras.Model):

    def __init__(self, n_vocab):
        super().__init__()
        self.embedding = keras.layers.Embedding(n_vocab, n_vocab)

    def call(self, indices):
        """
        Args:
            indices: Tensor [batch_size, time]
        Returns:
            logits: Tensor [batch_size, time, n_vocab]
        """
        return self.embedding(indices)

    def generate(self, indices, n_new_tokens):
        """
        Args:
            indices: Tensor (batch_size, time)
        Returns:
            indices: Tensor (batch_size, time+n_new_tokens)
        """
        for _ in range(n_new_tokens):
            logits = self(indices)
            logits = logits[:, -1, :]
            idx_new = tf.random.categorical(logits, num_samples=1)
            indices = tf.concat((indices, idx_new), axis=1)

        return indices


if __name__ == '__main__':
    from tiny_shakespeare import TinyShakespeare

    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    # params
    buffer_size = 10000
    batch_size = 32
    block_size = 8
    epochs = 10
    learning_rate = 1e-2

    # data
    dataset = TinyShakespeare(decoder='simple')

    data_trn = tf.data.Dataset.from_tensor_slices(dataset.data_trn)
    data_trn = data_trn.batch(block_size+1, drop_remainder=True)
    data_trn = data_trn.map(split_input_target)
    data_trn = data_trn.shuffle(buffer_size).batch(batch_size)

    data_val = tf.data.Dataset.from_tensor_slices(dataset.data_val)
    data_val = data_val.batch(block_size+1, drop_remainder=True)
    data_val = data_val.map(split_input_target)
    data_val = data_val.shuffle(buffer_size).batch(batch_size)

    # model
    model = BigramLanguageModel(dataset.n_vocab)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.AdamW(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)
    history = model.fit(data_trn, validation_data=data_val, epochs=epochs)

    # eval
    context = tf.zeros((1, 1), dtype=tf.int64)
    print(dataset.decode(model.generate(context, 500)[0].numpy()))
