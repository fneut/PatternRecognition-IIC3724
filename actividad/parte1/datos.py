import numpy as np
from utils import imread
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

expressions = np.load('expressions.npy')
gender_age = np.load('gender_age.npy')
landmarks = np.load('landmarks.npy')

imagen1 = imread('./../faces/mb_01.jpg')
x = np.zeros([68, 1])
y = np.zeros([68, 1])
for i in range(68):
    x[i] = landmarks[0][i]
    y[i] = landmarks[0][i + 68]
plt.imshow(imagen1)
plt.hold(True)
plt.plot(x, y, 'o')
print('Imagen 1:')
print(
    'Genero: {:4.2f} - Edad: {:4.2f} - Angry: {:4.2f} - Disgust: {:4.2f} - Scared: {:4.2f} - Happy: {:4.2f} - Sad: {:4.2f} - Surprised: {:4.2f} - Neutral: {:4.2f}'.format(
        *gender_age[0], *expressions[0]))
plt.hold(False)
plt.show()
plt.savefig('imagen1.png')


imagen2 = imread('./../faces/mb_02.jpg')
x = np.zeros([68, 1])
y = np.zeros([68, 1])
for i in range(68):
    x[i] = landmarks[1][i]
    y[i] = landmarks[1][i + 68]
plt.imshow(imagen2)
plt.hold(True)
plt.plot(x, y, 'o')
print('Imagen 2:')
print(
    'Genero: {:4.2f} - Edad: {:4.2f} - Angry: {:4.2f} - Disgust: {:4.2f} - Scared: {:4.2f} - Happy: {:4.2f} - Sad: {:4.2f} - Surprised: {:4.2f} - Neutral: {:4.2f}'.format(
        *gender_age[1], *expressions[1]))
plt.hold(False)
plt.show()
plt.savefig('imagen2.png')


imagen3 = imread('./../faces/sp_01.jpg')
x = np.zeros([68, 1])
y = np.zeros([68, 1])
for i in range(68):
    x[i] = landmarks[2][i]
    y[i] = landmarks[2][i + 68]
plt.imshow(imagen3)
plt.hold(True)
plt.plot(x, y, 'o')
print('Imagen 3:')
print(
    'Genero: {:4.2f} - Edad: {:4.2f} - Angry: {:4.2f} - Disgust: {:4.2f} - Scared: {:4.2f} - Happy: {:4.2f} - Sad: {:4.2f} - Surprised: {:4.2f} - Neutral: {:4.2f}'.format(
        *gender_age[2], *expressions[2]))
plt.hold(False)
plt.show()
plt.savefig('imagen3.png')


imagen4 = imread('./../faces/sp_02.jpg')
x = np.zeros([68, 1])
y = np.zeros([68, 1])
for i in range(68):
    x[i] = landmarks[3][i]
    y[i] = landmarks[3][i + 68]
plt.imshow(imagen4)
plt.hold(True)
plt.plot(x, y, 'o')
print('Imagen 4:')
print(
    'Genero: {:4.2f} - Edad: {:4.2f} - Angry: {:4.2f} - Disgust: {:4.2f} - Scared: {:4.2f} - Happy: {:4.2f} - Sad: {:4.2f} - Surprised: {:4.2f} - Neutral: {:4.2f}'.format(
        *gender_age[3], *expressions[3]))
plt.hold(False)
plt.show()
plt.savefig('imagen4.png')


imagen5 = imread('./../faces/xx_01.jpg')
x = np.zeros([68, 1])
y = np.zeros([68, 1])
for i in range(68):
    x[i] = landmarks[4][i]
    y[i] = landmarks[4][i + 68]
plt.imshow(imagen5)
plt.hold(True)
plt.plot(x, y, 'o')
print('Imagen 5:')
print(
    'Genero: {:4.2f} - Edad: {:4.2f} - Angry: {:4.2f} - Disgust: {:4.2f} - Scared: {:4.2f} - Happy: {:4.2f} - Sad: {:4.2f} - Surprised: {:4.2f} - Neutral: {:4.2f}'.format(
        *gender_age[4], *expressions[4]))
plt.hold(False)
plt.show()
plt.savefig('imagen5.png')


imagen6 = imread('./../faces/xx_02.jpg')
x = np.zeros([68, 1])
y = np.zeros([68, 1])
for i in range(68):
    x[i] = landmarks[5][i]
    y[i] = landmarks[5][i + 68]
plt.imshow(imagen6)
plt.hold(True)
plt.plot(x, y, 'o')
print('Imagen 6:')
print(
    'Genero: {:4.2f} - Edad: {:4.2f} - Angry: {:4.2f} - Disgust: {:4.2f} - Scared: {:4.2f} - Happy: {:4.2f} - Sad: {:4.2f} - Surprised: {:4.2f} - Neutral: {:4.2f}'.format(
        *gender_age[5], *expressions[5]))
plt.hold(False)
plt.show()
plt.savefig('imagen6.png')


imagen7 = imread('./../faces/xx_03.jpg')
x = np.zeros([68, 1])
y = np.zeros([68, 1])
for i in range(68):
    x[i] = landmarks[6][i]
    y[i] = landmarks[6][i + 68]
plt.imshow(imagen7)
plt.hold(True)
plt.plot(x, y, 'o')
print('Imagen 7:')
print(
    'Genero: {:4.2f} - Edad: {:4.2f} - Angry: {:4.2f} - Disgust: {:4.2f} - Scared: {:4.2f} - Happy: {:4.2f} - Sad: {:4.2f} - Surprised: {:4.2f} - Neutral: {:4.2f}'.format(
        *gender_age[6], *expressions[6]))
plt.hold(False)
plt.show()
plt.savefig('imagen7.png')


imagen8 = imread('./../faces/xx_04.jpg')
x = np.zeros([68, 1])
y = np.zeros([68, 1])
for i in range(68):
    x[i] = landmarks[7][i]
    y[i] = landmarks[7][i + 68]
plt.imshow(imagen8)
plt.hold(True)
plt.plot(x, y, 'o')
print('Imagen 8:')
print(
    'Genero: {:4.2f} - Edad: {:4.2f} - Angry: {:4.2f} - Disgust: {:4.2f} - Scared: {:4.2f} - Happy: {:4.2f} - Sad: {:4.2f} - Surprised: {:4.2f} - Neutral: {:4.2f}'.format(
        *gender_age[7], *expressions[7]))
plt.hold(False)
plt.show()
plt.savefig('imagen8.png')
