import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os


def generate_dataset_by_dirichlet_distrib(num_classes, concentration, num_of_clients,
                                          num_of_examples_per_client):
    tfd = tfp.distributions
    np.set_printoptions(suppress=True)
    # num_classes = 10
    prior_class_distrib = tf.fill(num_classes, 1.0 / num_classes)
    # prior_class_distrib = tf.constant([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # concentration = 0.1
    # alpha = tf.constant([1.0, 2.0, 3.0])
    alpha = prior_class_distrib * concentration
    dist = tfd.Dirichlet(alpha)

    # num_of_clients = 10
    # 100 clients 500 images each
    samples = dist.sample(num_of_clients)  # Shape [5, 3]
    # print(samples)
    smpls_integer = tf.cast(tf.round(samples * num_of_examples_per_client), tf.int32)
    return smpls_integer


concentration = 1000.0
smpls = generate_dataset_by_dirichlet_distrib(num_classes=10, concentration=concentration, num_of_clients=20,
                                              num_of_examples_per_client=2500)
print(smpls)
# print(tf.reduce_sum(smpls, axis = 1))

# load cifar 10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# x_train.shape (50000, 32, 32, 3)
# x_test.shape (10000, 32, 32, 3)
# y_train.shape (50000, 1)
# y_test.shape (10000, 1)

num_of_classes = 10
indexes_of_labels = list([list([]) for _ in range(0, num_of_classes)])
print(indexes_of_labels)
# index = label
# element = list of index in x_train
j = 0
for label in y_train:
    # print(label)
    # print(type(label))
    indexes_of_labels[label.item()].append(j)
    j = j + 1

for l in range(0, num_of_classes):
    print(len(indexes_of_labels[l]))

num_of_examples_per_label = 10
print("----------------------")
c = 0

for per_client_sample in smpls:
    client_folder = "c_" + str(c)
    label = 0
    print("\nclient", client_folder)
    # here creating the folder to save

    list_extracted_all_labels = []
    for num_of_examples_per_label in per_client_sample:
        print(num_of_examples_per_label)
        extracted = np.random.choice(indexes_of_labels[label], num_of_examples_per_label, replace=False)
        print("label", label, "extracted", extracted)
        for ee in extracted.tolist():
            list_extracted_all_labels.append(ee)
            # print(ee)
        # here saving the dataset
        #
        label = label + 1

    # x_train.shape (50000, 32, 32, 3)
    # print(list_extracted_all_labels)
    numpy_dataset_y = y_train[list_extracted_all_labels]
    # print(numpy_dataset_y)
    numpy_dataset_x = x_train[list_extracted_all_labels]

    ds = tf.data.Dataset.from_tensor_slices((numpy_dataset_x, numpy_dataset_y))
    ds = ds.shuffle(buffer_size=4096)
    # print(ds)
    tf.data.experimental.save(ds, path=os.path.join(os.path.join("cifar10_datasets_sattler_" + str(concentration), "train"),
                                                    str(c)))
    c = c + 1

path = os.path.join("cifar10_datasets_sattler_" + str(concentration), "distribution.npy")
np.save(path, smpls.numpy())
smpls_loaded = np.load(path)
print(smpls_loaded)
# print(tf.round(samples*500))
# print(tf.reduce_sum(tf.round(samples*500), axis=1))
# print(tf.reduce_sum(tf.round(samples*500), axis = 0))
