import SqueezeNet as sq
import chainer
from chainer.backends import cuda
from chainer import gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer.dataset import concat_examples
import chainer.links as L
import numpy as np
from chainer.training import extensions
from chainer.backends.cuda import to_cpu
from chainer import Function as F
import time
import tqdm



results = open("results.txt", "a")
root = "/scratch/users/chrlam/Deception_project/Frames"
dataSet = chainer.datasets.LabeledImageDataset('/scratch/users/chrlam/Deception_project/data.txt',root)
gpu_id = 0
smallData = dataSet[1:300]

#Make training, validation and test set
ratio = 0.7
train_size = int(len(smallData) *ratio)
train, rest = chainer.datasets.split_dataset_random(smallData, train_size)
test_size = int(len(rest) * 0.5)
validation,test = chainer.datasets.split_dataset_random(smallData, test_size)    #split remaining 30% of dataset in validation and test set


#initialize iterators
batchsize = 16
max_epoch = 30
train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize, False, True)
validation_iter = iterators.SerialIterator(validation,batchsize,False, shuffle=False)


#initialize model
model = sq.SqueezeNet()
model = model.to_gpu()
print("Length of training set: {}".format(len(train)))
print("Length of test set: {}".format(len(test)))
print("Length of validation set: {}".format(len(validation)))

#setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

length= (len(train)/batchsize)
epochBar = tqdm.tqdm(total=length)


#actual training
while train_iter.epoch < max_epoch:


    # ------- One iteration of the training loop --------
    train_batch = train_iter.next()
    epochBar.update(1)
    image_train, target_train = concat_examples(train_batch,gpu_id) #check shapes
    image_train = Variable(cuda.to_gpu(image_train))
    target_train = Variable(cuda.to_gpu(target_train))


    # Calculate the prediction
    prediction_train = model(image_train)

    # Calculate the loss with softmax_cross_entropyq
    loss = chainer.functions.softmax_cross_entropy(prediction_train, target_train)

    # Calculate the gradients in the network
    model.cleargrads()
    loss.backward()

    # Update all the trainable parameters
    optimizer.update()
    # --------- until here-------------------------------



    # Check the validation accuracy of prediction after every epoch
    if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch
        epochBar.close()
        valBar = tqdm.tqdm(total=(len(validation)/batchsize))
        # Display the training loss
        results.write('epoch:{:02d} train_loss:{:.04f} '.format(
            train_iter.epoch, float(loss.data)))


        val_losses = []
        val_accuracies = []
        while True:
            valBar.update(1)
            val_batch = validation_iter.next()
            image_val, target_val = concat_examples(val_batch, gpu_id)

            # Forward the test data
            prediction_val = model(image_val)

            # Calculate the loss
            loss_test = chainer.functions.softmax_cross_entropy(prediction_val, target_val)
            val_losses.append(to_cpu(loss_test.data))

            # Calculate the accuracy
            accuracy = chainer.functions.accuracy(prediction_val, target_val)
            accuracy.to_cpu()
            val_accuracies.append(accuracy.data)

            if validation_iter.is_new_epoch:
                validation_iter.epoch = 0
                validation_iter.current_position = 0
                validation_iter.is_new_epoch = False
                validation_iter._pushed_position = None
                valBar.close()
                break

        results.write('val_loss:{:.04f} val_accuracy:{:.04f}\n'.format(
            np.mean(val_losses), np.mean(val_accuracies)))
        epochBar = tqdm.tqdm(total=length)

valBar.close()
epochBar.close()
serializers.save_npz('/scratch/users/chrlam/Deception_project/Results/singleFrame', model)



