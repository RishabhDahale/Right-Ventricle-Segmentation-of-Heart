import argparse
import time
from tqdm import tqdm
import pickle
from models.unet import UNet
import matplotlib.pyplot as plt
import numpy as np
from utility import parameters
import torchvision.transforms as transforms
import logging
import os
import random
import math
from utility.dataset import DataLoader
from torch import optim
import torch
from utility.loss import CombinedLoss, DiceLoss
from utility.regularization import L1_regularization, L1L2_regularization

np.random.seed(12)
torch.manual_seed(12)

def train():
    startTime = time.time()
    args = parameters.parse_arguments()
    logging.basicConfig(filename=args.logfile, level=logging.INFO)
    logging.critical("\n\n" + args.log_header)
    logging.info(args)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"TIME: {time.time() - startTime}s Using device {device}")

    logging.info(f"TIME: {time.time()-startTime}s Loading dataset")
    try:
        with open(os.path.join(args.datadir, "data.pkl"), "rb") as f:
            data = pickle.load(f)
    except:
        data = DataLoader(args.datadir, int(args.batchsize), shuffle=int(args.shuffle))
        with open(os.path.join(args.datadir, "data.pkl"), "wb") as f:
            pickle.dump(data, f)
    data.batchSize = int(args.batchsize)
    logging.info(f"TIME: {time.time()-startTime}s Dataset Loaded")

    random.seed(args.seed)
    indices = list(range(len(data)))
    random.shuffle(indices) # 0:floor((1-validationFrac)*len(data)) will be training data, rest will be validation data
    trainEndIndex = math.floor((1-args.validation_frac)*(len(data)))

    model = UNet(in_channels=1, num_classes=2, start_filts=int(args.conv_filters), up_mode=args.mode,
    			 depth=int(args.depth), batchnorm=args.batchnorm)
    model.reset_params()
    model = model.to(device)
    optimizer = None
    if args.optimizer=='adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lrstart)
        logging.info(f"TIME: {time.time()-startTime}s Optimizer: adam")
    elif args.optimizer=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lrstart, momentum=args.momentum)
        logging.info(f"TIME: {time.time()-startTime}s Optimizer: SGD")
    elif args.optimizer=='rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lrstart)
        logging.info(f"TIME: {time.time()-startTime}s Optimizer: RMSProp")
    else:
        logging.error(f"TIME: {time.time()-startTime}s Incorrect optimizer given")

    scheduler = []
    if args.lrscheduler=="steplr":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay)
        logging.info(f"TIME: {time.time()-startTime}s LRScheduler: StepLR")
    elif args.lrscheduler=="exponentiallr":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay)
        logging.info(f"TIME: {time.time()-startTime}s LRScheduler: exponentialLR")
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs))
        logging.info(f"TIME: {time.time()-startTime}s LRScheduler: lr shouldn't change with epochs")

    criteria = CombinedLoss(args.lambda_loss, args.loss_type)
    diceCoeff = DiceLoss()
    TL = []
    VL = []
    if not os.path.exists(os.path.join(os.getcwd(), "loss_files")):
    	os.makedirs(os.path.join(os.getcwd(), "loss_files"))
    lossFile = open(os.path.join("loss_files", args.log_header+".csv"), "w+")
    lossFile.write("Epoch,TrainLoss,ValidationLoss,Dice Coefficient\n")

    for epoch in tqdm(range(1, int(args.epochs) + 1), desc="Training model"):
        trainLoss = 0
        valLoss = 0
        trainingSample = 0
        testSample = 0
        netCoeff = 0
        for i in range(len(data)):
            images, masks = data[i]
            images = torch.tensor(images.astype(np.float32))
            masks = torch.tensor(masks.astype(np.float32))
            images = images.to(device)
            masks = masks.to(device)
            images = torch.transpose(images, 1, 3)
            masks = torch.transpose(masks, 1, 3)
            if i in indices[:trainEndIndex]:
                trainingSample += images.shape[0]
                networkPred = model(images)
                if args.regularization == 'l1':
                    reg = L1_regularization(model, args.reg_lamda1)
                    loss = criteria(masks, networkPred) + reg
                elif args.regularization =='l1l2':
                    reg = L1L2_regularization(model, args.reg_lamda1, args.reg_lamda2)
                    loss = criteria(masks, networkPred) + reg
                else:
                    loss = criteria(masks, networkPred)
                loss.backward()
                trainLoss += loss.item()
                optimizer.step()
                model.zero_grad()
            else:
                with torch.no_grad():
                    testSample += images.shape[0]
                    prediction = model(images)
                    if (epoch%args.save_epochs==0) or (epoch==1) or (epoch==args.epochs):
                        imgPath = os.path.join("validation_sample", args.log_header, f"epoch {epoch}")
                        if not os.path.exists(imgPath):
                        	os.makedirs(imgPath)
                        hrt = images[0, 0, :, :].to("cpu")
                        plt.imshow(np.array(hrt), cmap='gray')
                        plt.title("Heart Image")
                        plt.savefig(os.path.join(imgPath, "heart.png"))
                        plt.clf()
                        # ax = figure.add_subplot(232, title="Mask 1 Predicted")
                        msk1 = prediction[0, 0, :, :].to("cpu")
                        plt.imshow(np.array(msk1), cmap='gray')
                        plt.title("Predicted Mask 1")
                        plt.savefig(os.path.join(imgPath, "pred-mask1.png"))
                        plt.clf()
                        # ax = figure.add_subplot(231, title="Mask 2 Predicted")
                        msk2 = prediction[0, 1, :, :].to("cpu")
                        plt.imshow(np.array(msk2), cmap='gray')
                        plt.title("Predicted Mask 2")
                        plt.savefig(os.path.join(imgPath, "pred-mask2.png"))
                        plt.clf()

                        msk = np.zeros((192, 192, 3))
                        msk[:, :, 0] = np.array(msk1)
                        msk[:, :, 1] = np.array(msk2)
                        plt.imshow(np.array(hrt), cmap='gray')
                        plt.imshow(msk, cmap='jet', alpha=0.4)
                        plt.title("predicted-RV")
                        plt.savefig(os.path.join(imgPath, "pred-RV.png"))
                        plt.clf()
                        # ax = figure.add_subplot(231, title="Mask 2 Real")
                        msk1 = masks[0, 0, :, :].to("cpu")
                        plt.imshow(np.array(msk1), cmap='gray')
                        plt.title("Actual Mask 1")
                        plt.savefig(os.path.join(imgPath, "actual-mask1.png"))
                        plt.clf()
                        # ax = figure.add_subplot(231, title="Mask 2 Real")
                        msk2 = masks[0, 1, :, :].to("cpu")
                        plt.imshow(np.array(msk2), cmap='gray')
                        plt.title("Actual Mask 2")
                        plt.savefig(os.path.join(imgPath, "actual-mask2.png"))
                        plt.clf()
                        # plt.savefig(os.path.join("validation_sample", f"{args.log_header}-epoch {epoch}.png"))
                        msk = np.zeros((192, 192, 3))
                        msk[:, :, 0] = np.array(msk1)
                        msk[:, :, 1] = np.array(msk2)
                        plt.imshow(np.array(hrt), cmap='gray')
                        plt.imshow(msk, cmap='jet', alpha=0.4)
                        plt.title("actual-RV")
                        plt.savefig(os.path.join(imgPath, "actual-RV.png"))
                        plt.clf()
                        
                    if args.regularization == 'l1':
                        reg = L1_regularization(model, args.reg_lamda1)
                        loss = criteria(masks, prediction) + reg
                    elif args.regularization =='l1l2':
                        reg = L1L2_regularization(model, args.reg_lamda1, args.reg_lamda2)
                        loss = criteria(masks, prediction) + reg
                    else:
                        loss = criteria(masks, prediction)
                    valLoss += loss.item()
                    coeff = diceCoeff(masks, prediction)
                    netCoeff += torch.sum(1-coeff).item()
        if (epoch%int(args.save_epochs)==0) or (epoch==int(args.epochs)):
            if not os.path.exists(args.model_save_dir):
                os.makedirs(args.model_save_dir)
            # save model
            torch.save({
                "epoch" : epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(args.model_save_dir, f"model-epoch({epoch}).hdf5"))
            logging.info(f"TIME: {time.time()-startTime}s Model state saved for epoch: {epoch}")
        logging.info(f"TIME: {time.time()-startTime}s TRAINING: Epoch: {epoch}, lr: {scheduler.get_last_lr()}, loss: {trainLoss/(2*trainingSample)}")
        logging.info(f"TIME: {time.time()-startTime}s VALIDATION: Epoch: {epoch}, lr: {scheduler.get_last_lr()}, loss: {valLoss/(2*testSample)}")
        TL.append(trainLoss/(2*trainingSample))
        VL.append(valLoss/(2*testSample))
        lossFile.write(f"{epoch},{trainLoss/(2*trainingSample)},{valLoss/(2*testSample)},{netCoeff/(2*testSample)}\n")
        scheduler.step()  # https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/
    plt.plot(list(range(1, int(args.epochs) + 1)), TL, label="Training loss")
    plt.plot(list(range(1, int(args.epochs) + 1)), VL, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    if not os.path.exists(os.path.join(os.getcwd(), "plots")):
    	os.makedirs(os.path.join(os.getcwd(), "plots"))
    plt.savefig(os.path.join("plots", args.log_header+".png"))

train()
