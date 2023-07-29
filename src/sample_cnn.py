import pandas as pd
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming you have your charm jet data and labels stored in 'features' and 'labels' variables

data = pd.read_csv(
    "/Users/sonia/PycharmProjects/jet-physics-and-machine-learning/data/tagged_D0meson_jets.csv"
)

num_samples = len(data)
# num_samples = 28868
px, py, pz = 28868, 28868, 28868
num_channels = 4
num_classes = 2

features = data.iloc[:, 1:-5].values
labels = data.iloc[:, -1].values


# features = data.iloc[:10, 1:-5].values.reshape(len(data), 28, 28, 4)
# labels = data.iloc[:10, -1].values

scaler = StandardScaler()

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define the CNN architecture
model = Sequential()

model.add(
    Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(px, py, pz, num_channels),
    )
)
input_shape_of_conv2d = model.layers[0].input_shape
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(32, activation="relu"))
model.add(Dense(num_classes, activation="sigmoid"))
model.summary()

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
# print('Test loss:', loss)
# print('Test accuracy:', accuracy)

# Make predictions on new charm jet data
# new_data = ...  # Your new charm jet data
# predictions = model.predict(new_data)
