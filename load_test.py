import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

def load_client_datasets_from_files():
    # def preprocess_train_dataset(dataset):
    #     # Use buffer_size same as the maximum client dataset size,
    #     # 418 for Federated EMNIST
    #     return dataset.map(
    #         lambda element: (tf.expand_dims(element['pixels'], -1), element['label'])).shuffle(
    #         buffer_size=418).repeat(count=5).batch(application_paramaters.BATCH_SIZE, drop_remainder=False)
    #
    # def preprocess_test_dataset(dataset):
    #     return dataset.map(
    #         lambda element: (tf.expand_dims(element['pixels'], -1), element['label'])).batch(
    #         application_paramaters.TEST_BATCH_SIZE, drop_remainder=False)

    path = os.path.join(".", "cifar10_datasets_1.0/train")
    sampled_clients = np.random.choice(
        os.listdir(path),  # 3400
        size=1,
        replace=False)

    loaded_ds_train = tf.data.experimental.load(
        path=os.path.join(path, sampled_clients[0]), element_spec=None, compression=None, reader_func=None
    )

    # loaded_ds_test = tf.data.experimental.load(
    #     path="emnist_datasets/test/" + sampled_clients[0], element_spec=None, compression=None, reader_func=None
    # )

    return loaded_ds_train


plt.figure(figsize=(40, 40))
ds_train = load_client_datasets_from_files()
iterator = iter(ds_train)
i = 0
for example, label in iterator:
    # print(example)
    # print(tf.squeeze(label))
    ax = plt.subplot(25, 20, i+1)
    plt.imshow(example.numpy().astype("uint8"))
    # plt.title(class_names[int(result[i])])
    plt.title(label.numpy().astype("uint8"))
    plt.axis("off")
    i = i+1
plt.show()

path = os.path.join(".", "cifar10_datasets_1.0/distribution.npy")
smpls_loaded = np.load(path)
print(smpls_loaded)