"""
This file is based on:
    https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
"""
import tensorflow as tf
import tensorflow.keras as keras


class MaskedAttention(keras.layers.Layer):

    def __init__(self, size, dropout_rate=0.0):
        super().__init__()
        self.key = keras.layers.Dense(size, use_bias=False)
        self.query = keras.layers.Dense(size, use_bias=False)
        self.value = keras.layers.Dense(size, use_bias=False)
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: Tensor [batch_dim, context_dim, emb_dim]
        Result:
            out: Tensor [batch_dim, context_dim, head_size]
        """
        batch_dim, context_dim, emb_dim = inputs.shape
        tril = tf.cast(tf.linalg.band_part(tf.ones((context_dim, context_dim), dtype=tf.uint8), -1, 0), dtype=tf.bool)

        k = self.key(inputs)
        q = self.query(inputs)
        v = self.value(inputs)

        w = q @ tf.transpose(k, perm=[0, 2, 1]) * emb_dim**-0.5
        w = tf.where(tril, w, float('-inf'))
        w = tf.nn.softmax(w, axis=1)
        w = self.dropout(w, training=training)
        out = w @ v
        return out


class MultiHeadMaskedAttention(keras.layers.Layer):

    def __init__(self, n_heads, head_size, dropout_rate=0.0):
        super().__init__()
        self.heads = [MaskedAttention(head_size, dropout_rate=dropout_rate) for _ in range(n_heads)]
        self.proj = keras.layers.Dense(n_heads*head_size, activation='linear')
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: Tensor [batch_dim, context_dim, d]
        Result:
            out: Tensor [batch_dim, context_dim, n_heads*head_size]
        """
        x = tf.concat([h(inputs) for h in self.heads], axis=-1)
        x = self.proj(x)
        x = self.dropout(x, training=training)
        return x


class FeedForward(keras.layers.Layer):

    def __init__(self, output_dim, activation='relu', dropout_rate=0.0):
        super().__init__()
        self.inner = keras.layers.Dense(4*output_dim, activation=activation)
        self.proj = keras.layers.Dense(output_dim, activation='linear')
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: Tensor [batch_dim, ..., dN]
        Result:
            out: Tensor [batch_dim, ..., output_dim]
        """
        x = self.inner(inputs)
        x = self.proj(x)
        x = self.dropout(x, training=training)
        return x


class Block(keras.layers.Layer):

    def __init__(self, n_heads, emb_dim, dropout_rate=0.0):
        super().__init__()
        self.mhsa = MultiHeadMaskedAttention(n_heads, emb_dim//n_heads, dropout_rate=dropout_rate)
        self.ff = FeedForward(emb_dim, dropout_rate=dropout_rate)
        self.layer_norm_1 = keras.layers.LayerNormalization()
        self.layer_norm_2 = keras.layers.LayerNormalization()

    def call(self, inputs, training=None):
        """
        Args:
            inputs: Tensor [batch_dim, context_dim, emb_dim]
        Result:
            out: Tensor [batch_dim, context_dim, emb_dim]
        """
        x = self.layer_norm_1(inputs)
        x = inputs + self.mhsa(x, training=training)
        x = self.layer_norm_2(x)
        out = x + self.ff(x, training=training)
        return out


class Transformer(keras.Model):

    def __init__(self, n_vocab, emb_dim, context_dim, n_blocks, n_heads, dropout_rate=0.0):
        super().__init__()
        self.context_dim = context_dim

        self.tok_embedding = keras.layers.Embedding(n_vocab, emb_dim)
        self.pos_embedding = keras.layers.Embedding(context_dim, emb_dim)

        self.sa_head = keras.Sequential(
            [Block(n_heads, emb_dim, dropout_rate=dropout_rate) for _ in range(n_blocks)]
        )

        self.layer_norm = keras.layers.LayerNormalization()
        self.head = keras.layers.Dense(n_vocab)


    def call(self, inputs, training=None):
        """
        Args:
            inputs: Tensor [batch_size, context_dim]
        Returns:
            logits: Tensor [batch_size, context_dim, n_vocab]
        """
        batch_dim, context_dim = inputs.shape

        pos = tf.range(0, context_dim, delta=1, dtype=tf.int64)
        emb_tok = self.tok_embedding(inputs)
        emb_pos = self.pos_embedding(pos)
        emb = emb_tok + emb_pos

        x = self.sa_head(emb, training=training)
        x = self.layer_norm(x)
        x = self.head(x)

        return x

    def generate(self, indices, n_new_tokens):
        """
        Args:
            indices: Tensor (batch_size, time)
        Returns:
            indices: Tensor (batch_size, time+n_new_tokens)
        """
        for _ in range(n_new_tokens):
            logits = self(indices[:, -self.context_dim:], training=False)
            logits = logits[:, -1, :]
            idx_new = tf.random.categorical(logits, num_samples=1)
            indices = tf.concat((indices, idx_new), axis=1)

        return indices


if __name__ == '__main__':
    import os
    from tiny_shakespeare import TinyShakespeare

    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # params
    emb_dim = 384
    context_size = 256
    n_blocks = 6
    n_heads = 6
    dropout_rate = 0.2

    buffer_size = 10000
    batch_size = 64

    epochs = 10
    steps_per_epoch = 500
    validation_steps = 2
    learning_rate = 3e-4

    # data
    dataset = TinyShakespeare(decoder='simple')

    data_trn = tf.data.Dataset.from_tensor_slices(dataset.data_trn)
    data_trn = data_trn.batch(context_size+1, drop_remainder=True)
    data_trn = data_trn.map(split_input_target)
    data_trn = data_trn.shuffle(buffer_size).batch(batch_size)
    data_trn = data_trn.repeat(None)

    data_val = tf.data.Dataset.from_tensor_slices(dataset.data_val)
    data_val = data_val.batch(context_size+1, drop_remainder=True)
    data_val = data_val.map(split_input_target)
    data_val = data_val.shuffle(buffer_size).batch(batch_size)
    data_val = data_val.repeat(None)

    # model
    model = Transformer(dataset.n_vocab, emb_dim, context_size, n_blocks, n_heads, dropout_rate=dropout_rate)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.AdamW(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)

    # training
    callbacks = [
        tf.keras.callbacks.TensorBoard(),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=False)
    ]

    history = model.fit(
        data_trn,
        validation_data=data_val,
        validation_steps=validation_steps,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks
        )

    model.summary()

    # eval
    context = tf.zeros((1, 1), dtype=tf.int64)
    print(dataset.decode(model.generate(context, 100)[0].numpy().tolist()))
