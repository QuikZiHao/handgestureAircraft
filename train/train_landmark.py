import torch
import argparse
import random
import numpy as np
from model_landmark import LandMarkModel
from dataset_landmark import LandMarkDataSet
from dataloader_landmark import LandMarkDataLoader
from torch import nn, optim
from tqdm import tqdm
from timeit import default_timer as timer
from model_process import save_model 

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_train_ratio", type=float, default=0.8)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument('--dataset_dir', type=str, default='model\dataset')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--model_path', type=str, default='model/save/model.pth')

    args = parser.parse_args()

    return args

def set_seed(seed,device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def accuracy_fn(y_pred,y_true):
    # Get the predicted class by finding the index with the maximum value
    y_pred_class = (y_pred >= 0.5).float()
    
    # Check how many predictions are correct
    correct_predictions = (y_pred_class == y_true).float()
    
    # Calculate the accuracy
    accuracy = (correct_predictions.sum() / len(y_true)) * 100
    
    return accuracy

def train():
    args = get_args()

    device = args.device
    set_seed(args.random_seed,device)
    data = LandMarkDataSet(args.dataset_dir)
    data_loader = LandMarkDataLoader(dataset=data, test_train_ratio=args.test_train_ratio, batch_size=args.batch_size, shuffle= args.shuffle)
    model = LandMarkModel(input_size= 42,output_size = 1)
    model.to(device)

    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    epochs = args.epochs
    time_start = timer()
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        train_acc = 0
        model.train()
        for batch, (X, y) in enumerate(data_loader.train_loader):
            #frontpropagation
            X = X.type(torch.FloatTensor).to(device)       
            y = y.type(torch.FloatTensor).squeeze().to(device)
            y_pred = model(X).squeeze()

            #calculate loss and costfunction
            acc = accuracy_fn(y_pred=y_pred,y_true=y)
            loss = loss_fn(y_pred,y)
            train_loss += loss.item()
            train_acc += acc

            #grad to zero
            optimizer.zero_grad()

            #backpropagation
            loss.backward()

            #gradient descent
            optimizer.step()

        train_loss = train_loss/len(data_loader.train_loader)
        train_acc = train_acc/len(data_loader.train_loader)

        #val dataset
        test_loss = 0
        test_acc = 0
        for batch, (X_test_batch, y_test) in enumerate(data_loader.test_loader):
            X_test_batch = X_test_batch.type(torch.FloatTensor).to(device)
            y_test = y_test.type(torch.FloatTensor).squeeze().to(device)
            model.eval()
            with torch.inference_mode():
                y_test_pred = model(X_test_batch).squeeze()
                loss  = loss_fn(y_test_pred,y_test)
                acc = accuracy_fn(y_test_pred,y_test)
                test_loss += loss.item()
                test_acc += acc
        test_loss = test_loss/len(data_loader.test_loader)
        test_acc = test_acc/len(data_loader.test_loader)
        if epoch % 20 == 19:
            print()
            print(f"Epoch = {epoch+1}\n-------------")
            print(f"Train loss: {train_loss:.5f} ,Train acc: {train_acc:.2f}\n val loss: {test_loss:.5f}, val acc: {test_acc:.2f}\n")
    time_end = timer()
    duration = time_end-time_start
    print(f"time use:{(duration):2f}seconds")
    save_model(model,args.model_path)
    print(f"Model saved to {args.model_path}")

train()

