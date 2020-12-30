import numpy as np
import cv2
import os, os.path
import matplotlib.pyplot as plt

pixel = 64
ACCR=0
bestClassA = 0
bestClassB = 0
bestClassC = 0
bestClassD = 0
bestClassE = 0
epochs=200


def trainLoad():
    paths = ["../flower_photos/daisy", "../flower_photos/dandelion", "../flower_photos/roses",
             "../flower_photos/tulips", "../flower_photos/sunflowers"]
    images = []
    imagesLabel = []

    imagesTrain = []
    imagesTrainLabel = []

    imagesTest = []

    imagesValLabel = []
    count = 0;

    for i in paths:
        print(i, "path")
        sortedFiles = sorted(os.listdir(i))
        for f in sortedFiles:
            img = cv2.imread(i + '/' + f)
            img = cv2.resize(img, (pixel, pixel), interpolation=cv2.INTER_AREA)
            images.append(img)
            imagesLabel.append(count)

        imagesTrain = imagesTrain + images[:len(images) - 100]
        imagesTest = imagesTest + images[len(images) - 100:]

        imagesTrainLabel = imagesTrainLabel + imagesLabel[:len(imagesLabel) - 100]
        imagesValLabel = imagesValLabel + imagesLabel[len(imagesLabel) - 100:]

        count = count + 1
        images = []
        imagesLabel = []

    imagesTrain = np.array(imagesTrain)
    imagesTest = np.array(imagesTest)

    all = np.concatenate([imagesTrain, imagesTest])
    all = all.astype(np.float32)

    imagesTrain = np.reshape(imagesTrain, (imagesTrain.shape[0], -1)).astype(np.float32)
    imagesTest = np.reshape(imagesTest, (imagesTest.shape[0], -1)).astype(np.float32)

    imagesTrain -= np.mean(all)
    imagesTest -= np.mean(all)

    activation(80, 2, 0.00002, imagesTrain, imagesTrainLabel, imagesTest, imagesValLabel)
    return


def running(imagesTest, imagesValLabel,hiddenLayers,W,b):
   global ACCR
   accuracy = 0
   totalLoss=0
   classAccuracy = [0]*5
   hL = [[]] * hiddenLayers

   global bestClassA
   global bestClassB
   global bestClassC
   global bestClassD
   global bestClassE

   for i in range(5):
       totalLoss = 0
       for j in range(100):
       #######activation function RELU

           if (hiddenLayers <= 0):
               scores = (np.dot(W[0].T,  imagesTest[j+(i*100)]) + b[0])[0]
           else:
               #######activation function RELU
               hL[0] = np.maximum(0, np.dot(W[0].T,  imagesTest[j+(i*100)]) + b[0])[0]
               for h in range(1, hiddenLayers):
                   hL[h] = np.maximum(0, np.dot(W[h].T, hL[h - 1]) + b[0])[0]
               scores = (np.dot(W[hiddenLayers].T, hL[-1]) + b[hiddenLayers])[0]

           loss = np.exp(scores)  # 2nd
           probability = loss / np.sum(loss)  # normalized
           totalLoss += -np.log(probability[imagesValLabel[j+(i*100)]])
           maxIndex = np.argmax(probability)

           if (maxIndex == [imagesValLabel[j+(i*100)]]):
               accuracy += 1
               classAccuracy[i]+=1

   if accuracy/5>=ACCR:
       ACCR=accuracy/5
       bestClassA =classAccuracy[0]
       bestClassB = classAccuracy[1]
       bestClassC = classAccuracy[2]
       bestClassD = classAccuracy[3]
       bestClassE = classAccuracy[4]
   print("Val Accuracy ", accuracy / 5, "     ", classAccuracy[0], " ", classAccuracy[1], " ", classAccuracy[2], " ", classAccuracy[3], " ", classAccuracy[4]," Loss: ", totalLoss/(500), "ACCR: " , ACCR)
   return totalLoss/(500)

def activation(nodes, hiddenLayers, learningRate, imagesTrain, imagesTrainLabel, imagesTest, imagesValLabel):

    valLoss=[100000.0]*epochs
    trainLoss = [100000.0] * epochs

    hL = [[]] * hiddenLayers
    W = [[]] * (hiddenLayers + 1)
    b = [[]] * (hiddenLayers + 1)

    dW = [[]] * (hiddenLayers + 1)
    dB = [[]] * (hiddenLayers + 1)

    dH = [[]] * (hiddenLayers)


    numOfImages = imagesTrain.shape[0]
    classes = 5
    pixels = pixel * pixel * 3  # 3072,50

    if(hiddenLayers>=1):
        W[0] = np.random.randn(pixels, nodes) / np.sqrt(pixels)
        b[0] = np.zeros((1, nodes))
        for l in range(1,hiddenLayers):
            W[l] = np.random.randn(nodes, nodes) / np.sqrt(nodes)
            b[l] = np.zeros((1, nodes))
    W[hiddenLayers] = np.random.randn(nodes, classes) / np.sqrt(nodes)
    b[hiddenLayers] = np.zeros((1, classes))

    for j in range(epochs):
        gradientWs = [0]* (hiddenLayers + 1)

        dBs = [0]* (hiddenLayers + 1)
        batchTrainingLoss = 0
        totalLoss = 0
        accuracy = 0
        for i in range(numOfImages):

            if(hiddenLayers<=0):
                scores = (np.dot(W[0].T, imagesTrain[i]) + b[0])[0]
            else:
            #######activation function RELU
                hL[0] = np.maximum(0, np.dot(imagesTrain[i],W[0]) + b[0])[0]
                for h in range(1,hiddenLayers):
                    hL[h] = np.maximum(0, np.dot(hL[h-1],W[h]) + b[0])[0]
                scores = (np.dot(W[hiddenLayers].T, hL[-1]) + b[hiddenLayers])[0]

            ################loss function
            loss = np.exp(scores)  # 2nd
            probability = loss / np.sum(loss)  # normalized
            maxIndex = np.argmax(probability)
            if (maxIndex == [imagesTrainLabel[i]]):
                accuracy += 1
            totalLoss += -np.log(probability[imagesTrainLabel[i]])
            batchTrainingLoss += -np.log(probability[imagesTrainLabel[i]])

            correctClass = imagesTrainLabel[i]
            ################backpropegation
            ####dervitive of loss
            dScores = probability
            dScores[correctClass] -= 1

            if (hiddenLayers >= 1):
                dW[hiddenLayers] = np.dot(hL[hiddenLayers-1].reshape(hL[hiddenLayers-1].shape[0], 1), dScores.reshape(dScores.shape[0], 1).T)
                dB[hiddenLayers] = np.sum(dScores, axis=0, keepdims=True)

                dH[hiddenLayers-1] = np.dot(dScores, W[hiddenLayers].T)
                dH[hiddenLayers-1] = dH[hiddenLayers-1] * (hL[hiddenLayers-1] > 0)


            for l in range(hiddenLayers-1,0,-1):
                    dW[l] = np.dot(hL[l-1].reshape(hL[l-1].shape[0], 1), dH[l].reshape(dH[l].shape[0], 1).T)
                    dB[l] = np.sum(dH[l], axis=0, keepdims=True)
                    dH[l-1] = np.dot(dH[l], W[l].T)
                    dH[l-1] = dH[l-1]  * (hL[l-1] > 0)

            dW[0] = np.dot(imagesTrain[i].reshape(imagesTrain[i].shape[0], 1), dH[-1].reshape(dH[-1].shape[0], 1).T)
            dB[0] = np.sum(dH[-1], axis=0, keepdims=True)

            gradientWs[0] += dW[0]
            dBs[0] += dB[0]
            for k in range(1,hiddenLayers+1):
                gradientWs[k] += dW[k]
                dBs[k] += dB[k]

            miniBatch = 100
            if ((i % miniBatch) == 0 and i!=0 ):
                for f in range(0, hiddenLayers+1):
                    W[f] += -learningRate * (gradientWs[f] / miniBatch)
                    b[f] += -learningRate * (dBs[f] / miniBatch)

                gradientWs = [0]* (hiddenLayers + 1)

                dBs =  [0]* (hiddenLayers + 1)
                batchValLoss= running(imagesTest, imagesValLabel,hiddenLayers,W,b)
                print("T Loss ", batchTrainingLoss/10, " V Loss: ", batchValLoss)
                if(batchValLoss<valLoss[j]):
                    valLoss[j]=batchValLoss
                if(batchTrainingLoss/10<trainLoss[j]):
                    trainLoss[j]=batchTrainingLoss/10

                batchTrainingLoss=0
                print(j, " Accuracy: ", (accuracy / i) * 100, " Loss: ", totalLoss / i)
                print("BEST ACCR: ", ACCR)
                print("CCR: ", bestClassA, " ", bestClassB, " ", bestClassC, " ", bestClassD, " ", bestClassE)
        print(j, " Accuracy: ", (accuracy / numOfImages)*100, " Loss: ", totalLoss / numOfImages)
        print("BEST ACCR: ", ACCR)
        print("CCR: ", bestClassA, " ", bestClassB, " ", bestClassC, " ", bestClassD, " ", bestClassE)
        # running(imagesTest, imagesValLabel, W[0], W[hiddenLayers], b[0], b[hiddenLayers],j)
    # print("BEST ACCR: ", ACCR)
    # print("CCR: ", bestClassA," ", bestClassB, " ", bestClassC, " ", bestClassD, " ", bestClassE)
    print(valLoss)
    print(trainLoss)
    plt.plot( range(epochs),trainLoss, 'ro-',  range(epochs),valLoss, 'bo-')
    plt.grid()
    plt.xlabel('Red: TrainLoss, Blue: ValLoss')
    plt.ylabel('Epochs')
    plt.show()

    return


trainLoad()
