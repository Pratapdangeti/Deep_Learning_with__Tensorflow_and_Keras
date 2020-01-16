



# Deep Auto Encoders
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

from keras.layers import Input,Dense
from keras.models import Model


digits = load_digits()
X = digits.data
y = digits.target

print(X.shape)
print(y.shape)


x_vars_stdscle = StandardScaler().fit_transform(X)
print (x_vars_stdscle.shape)



# 3-Dimensional architecture
input_layer = Input(shape=(64,), name="input")

encoded = Dense(32, activation='relu', name="h1encode")(input_layer)
encoded = Dense(16, activation='relu', name="h2encode")(encoded)
encoded = Dense(3, activation='relu', name="h3latent_layer")(encoded)

decoded = Dense(16, activation='relu', name="h4decode")(encoded)
decoded = Dense(32, activation='relu', name="h5decode")(decoded)
decoded = Dense(64, activation='sigmoid', name="h6decode")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# Fitting Encoder-Decoder model
autoencoder.fit(x_vars_stdscle, x_vars_stdscle, epochs=100, batch_size=256, shuffle=True, validation_split=0.2)



# Output from whole encoder-decoder architecture
output_encdec = autoencoder.predict(x_vars_stdscle)

img_smple = 0

# Plotting input and output of whole model
# for displaying reconstructed image
plt.figure(figsize=(4,2))
plt.subplot(1, 2, 1)
plt.imshow(X[img_smple].reshape(8,8))
plt.subplot(1, 2, 2)
plt.imshow(output_encdec[img_smple].reshape(8,8))
plt.show()



# Extracting Encoder section of the Model for prediction of latent variables
encoder = Model(autoencoder.input, autoencoder.get_layer("h3latent_layer").output)

# Predicting latent variables with extracted Encoder model
reduced_X3D = encoder.predict(x_vars_stdscle)



zero_x, zero_y, zero_z = [], [], []
one_x, one_y, one_z = [], [], []
two_x, two_y, two_z = [], [], []
three_x, three_y, three_z = [], [], []
four_x, four_y, four_z = [], [], []
five_x, five_y, five_z = [], [], []
six_x, six_y, six_z = [], [], []
seven_x, seven_y, seven_z = [], [], []
eight_x, eight_y, eight_z = [], [], []
nine_x, nine_y, nine_z = [], [], []

for i in range(len(reduced_X3D)):

    if y[i] == 10:
        continue

    elif y[i] == 0:
        zero_x.append(reduced_X3D[i][0])
        zero_y.append(reduced_X3D[i][1])
        zero_z.append(reduced_X3D[i][2])

    elif y[i] == 1:
        one_x.append(reduced_X3D[i][0])
        one_y.append(reduced_X3D[i][1])
        one_z.append(reduced_X3D[i][2])

    elif y[i] == 2:
        two_x.append(reduced_X3D[i][0])
        two_y.append(reduced_X3D[i][1])
        two_z.append(reduced_X3D[i][2])

    elif y[i] == 3:
        three_x.append(reduced_X3D[i][0])
        three_y.append(reduced_X3D[i][1])
        three_z.append(reduced_X3D[i][2])

    elif y[i] == 4:
        four_x.append(reduced_X3D[i][0])
        four_y.append(reduced_X3D[i][1])
        four_z.append(reduced_X3D[i][2])

    elif y[i] == 5:
        five_x.append(reduced_X3D[i][0])
        five_y.append(reduced_X3D[i][1])
        five_z.append(reduced_X3D[i][2])

    elif y[i] == 6:
        six_x.append(reduced_X3D[i][0])
        six_y.append(reduced_X3D[i][1])
        six_z.append(reduced_X3D[i][2])

    elif y[i] == 7:
        seven_x.append(reduced_X3D[i][0])
        seven_y.append(reduced_X3D[i][1])
        seven_z.append(reduced_X3D[i][2])

    elif y[i] == 8:
        eight_x.append(reduced_X3D[i][0])
        eight_y.append(reduced_X3D[i][1])
        eight_z.append(reduced_X3D[i][2])

    elif y[i] == 9:
        nine_x.append(reduced_X3D[i][0])
        nine_y.append(reduced_X3D[i][1])
        nine_z.append(reduced_X3D[i][2])

# 3- Dimensional plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(zero_x, zero_y, zero_z, c='r', marker='x', label='zero')
ax.scatter(one_x, one_y, one_z, c='g', marker='+', label='one')
ax.scatter(two_x, two_y, two_z, c='b', marker='s', label='two')

ax.scatter(three_x, three_y, three_z, c='m', marker='*', label='three')
ax.scatter(four_x, four_y, four_z, c='c', marker='h', label='four')
ax.scatter(five_x, five_y, five_z, c='r', marker='D', label='five')

ax.scatter(six_x, six_y, six_z, c='y', marker='8', label='six')
ax.scatter(seven_x, seven_y, seven_z, c='k', marker='*', label='seven')
ax.scatter(eight_x, eight_y, eight_z, c='r', marker='x', label='eight')

ax.scatter(nine_x, nine_y, nine_z, c='b', marker='D', label='nine')

ax.set_xlabel('Latent Feature 1', fontsize=13)
ax.set_ylabel('Latent Feature 2', fontsize=13)
ax.set_zlabel('Latent Feature 3', fontsize=13)

ax.set_xlim3d(0, 60)
ax.set_ylim3d(0, 60)
ax.set_zlim3d(0, 60)

plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=10, bbox_to_anchor=(0, 0))

plt.show()




print("Completed")
