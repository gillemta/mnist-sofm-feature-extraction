import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class SOFM:
    def __init__(self, input_size, map_size, learning_rate=.05):
        print("Initializing SOFM with a map size of", map_size, "and input size of", input_size)
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.weights = np.random.random((map_size[0], map_size[1], input_size))

    @staticmethod
    def decay_learning_rate(initial_rate, epoch, time_constant):
        return initial_rate * np.exp(-epoch / time_constant)

    @staticmethod
    def decay_radius(initial_radius, epoch, time_constant):
        return initial_radius * np.exp(-epoch / time_constant)

    def find_winning_neuron(self, input_vec):
        winning_neuron = None
        min_distance = np.inf

        # Iterate over all neurons
        for x in range (self.map_size[0]):
            for y in range(self.map_size[1]):
                neuron_weight = self.weights[x, y, :]
                distance = np.linalg.norm(input_vec - neuron_weight)
                if distance < min_distance:
                    min_distance = distance
                    winning_neuron = (x, y)

        return winning_neuron

    def update_weights(self, input_vec, winning_neuron, radius, learning_rate):
        for x in range(self.map_size[0]):
            for y in range(self.map_size[1]):
                neuron_pos = np.array([x, y])
                winning_neuron_pos = np.array(winning_neuron)
                distance_to_winning_neuron = np.linalg.norm(neuron_pos - winning_neuron_pos)

                if distance_to_winning_neuron <= radius:
                    # Calculate the degree of influence
                    influence = np.exp(-distance_to_winning_neuron / (2 * (radius**2)))

                    self.weights[x, y, :] += influence * learning_rate * (input_vec - self.weights[x, y, :])

    def train(self, train_data, epochs=50):
        print(f"Starting training for {epochs} epochs.")
        initial_radius = self.map_size[0] / 2
        radius_decay_constant = epochs / np.log(initial_radius)
        learning_rate_decay_constant = epochs / np.log(self.learning_rate)

        for epoch in range(epochs):
            current_radius = self.decay_radius(initial_radius, epoch, radius_decay_constant)
            current_learning_rate = self.decay_learning_rate(self.learning_rate, epoch, learning_rate_decay_constant)
            print(f"Epoch {epoch + 1}/{epochs} - Learning rate: {current_learning_rate:.4f}, Radius: {current_radius:.4f}")

            # Iterate through each sample
            for input_vec in train_data:
                # Find winning neuron
                winning_neuron = self.find_winning_neuron(input_vec)

                # Update weights of neighbors
                self.update_weights(input_vec, winning_neuron, current_radius, current_learning_rate)

            print("Training complete.")

    def test(self, test_data, test_labels, num_points_per_class=100):
        activity_matrices = {i: np.zeros(self.map_size) for i in range(10)}

        for i, input_vec in enumerate(test_data):
            winning_neuron = self.find_winning_neuron(input_vec)
            actual_class = test_labels[i]
            activity_matrices[actual_class][winning_neuron] += 1

        for digit in activity_matrices:
            activity_matrices[digit] /= num_points_per_class

        return activity_matrices

    def visualize_weights(self):
        neuron_img_size = 28
        grid_size = self.map_size
        composite_image = np.zeros((grid_size[0] * neuron_img_size, grid_size[1] * neuron_img_size))

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                neuron_weights = self.weights[i, j].reshape((neuron_img_size, neuron_img_size))
                composite_image[i * neuron_img_size:(i + 1) * neuron_img_size,
                                j * neuron_img_size:(j + 1) * neuron_img_size] = neuron_weights

        # Plot the composite image
        plt.figure(figsize=(12, 12))
        plt.imshow(composite_image, cmap='gray')
        plt.title('SOFM Neuron Weights')
        plt.axis('off')
        plt.show()


# Load Data
image_data = pd.read_csv("./data-files/MNISTnumImages5000_balanced.txt", delimiter="\t", header=None)
label_data = pd.read_csv("./data-files/MNISTnumLabels5000_balanced.txt", header=None)

# Normalize and transpose the image data
normalized_data = image_data / 255.0
reshaped_and_transposed_data = normalized_data.values.reshape(-1, 28, 28).transpose(0, 2, 1)
flattened_data_for_sofm = reshaped_and_transposed_data.reshape(-1, 784)

zeros_train_data = image_data.iloc[0:400]
zeros_test_data = image_data.iloc[400:500]

ones_train_data = image_data.iloc[500:900]
ones_test_data = image_data.iloc[900:1000]

twos_train_data = image_data.iloc[1000:1400]
twos_test_data = image_data.iloc[1400:1500]

threes_train_data = image_data.iloc[1500:1900]
threes_test_data = image_data.iloc[1900:2000]

fours_train_data = image_data.iloc[2000:2400]
fours_test_data = image_data.iloc[2400:2500]

fives_train_data = image_data.iloc[2500:2900]
fives_test_data = image_data.iloc[2900:3000]

sixes_train_data = image_data.iloc[3000:3400]
sixes_test_data = image_data.iloc[3400:3500]

sevens_train_data = image_data.iloc[3500:3900]
sevens_test_data = image_data.iloc[3900:4000]

eights_train_data = image_data.iloc[4000:4400]
eights_test_data = image_data.iloc[4400:4500]

nines_train_data = image_data.iloc[4500:4900]
nines_test_data = image_data.iloc[4900:5000]

# Combine and shuffle the training data
train_data = pd.concat([zeros_train_data, ones_train_data, twos_train_data, threes_train_data, fours_train_data, fives_train_data, sixes_train_data, sevens_train_data, eights_train_data, nines_train_data]).sample(frac=1)

test_data = pd.concat([zeros_test_data, ones_test_data, twos_test_data, threes_test_data, fours_test_data, fives_test_data, sixes_test_data, sevens_test_data, eights_test_data, nines_test_data])
test_labels = label_data.iloc[test_data.index].values.flatten()

# Flatten and normalize the test data similar to the train_data
test_data = test_data.values / 255.0
test_data = test_data.reshape(-1, 784)


# Create SOFM
sofm = SOFM(input_size=784, map_size=(12, 12))

# Train SOFM
sofm.train(flattened_data_for_sofm)

# Test SOFM
activity_matrices = sofm.test(test_data, test_labels)

# Plotting each class's activity matrices
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
for digit, ax in enumerate(axes.flatten()):
    sns.heatmap(activity_matrices[digit], cmap='hot', annot=True, fmt='.2f', ax=ax, cbar=False)
    ax.set_title(f"Digit {digit}")
    ax.axis("off")
plt.tight_layout()
plt.show()

sofm.visualize_weights()





