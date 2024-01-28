import torch
import tqdm
import numpy as np
from model import Yolov2
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
from yolov2_data import Image_dataloader
from yolov2_loss import custom_loss
import matplotlib.pyplot as plt

# HYPERPARAMS:
LEARNING_RATE = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
WEIGHT_DECAY = 0.0005
EPOCHS = 30

def main():
    # checkpoint to save the model later:
    global checkpoint
    # print device:
    print(f'Device: {DEVICE}')
    # set the model, optimizer and custom loss:
    model = Yolov2().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    loss_fn = custom_loss()

    # Take full Dataset, divide into train and validation:
    full_dataset = Image_dataloader()
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Use dataloader for the datasets separately:
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=test_dataset, batch_size=VAL_BATCH_SIZE, shuffle=True, drop_last=True)

    # initialize mean training and validation loss:
    training_loss = np.zeros((1, EPOCHS))
    valid_loss = np.zeros((1, EPOCHS))

    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch+1} ...")
        loop = tqdm(train_loader)
        val_loop = tqdm(test_loader)
        train_loss = []
        val_loss = []
        torch.autograd.set_detect_anomaly(True)
        # TRAINING LOOP:
        model.train()
        for batch_idx, (x,y,z) in enumerate(loop):
            x, y, z = x.to(DEVICE, dtype=torch.float), y.to(DEVICE), z.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(y,out,z)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # Print mean loss and return:
        print(f"Training loss was {sum(train_loss)/len(train_loss)}")
        training_loss[0,epoch] = sum(train_loss)/len(train_loss)
        model.eval()
        # VALIDATION LOOP:
        with torch.no_grad():
            for batch_idx, (a,b,c) in enumerate(val_loop):
                a, b, c = a.to(DEVICE, dtype=torch.float), b.to(DEVICE), c.to(DEVICE)
                output = model(a)
                loss = loss_fn(b,output,c)
                val_loss.append(loss.item())
        print(f"Validation loss was {sum(val_loss)/len(val_loss)}")
        valid_loss[0,epoch] = sum(val_loss)/len(val_loss)
        print(f"EPOCH {epoch+1} finished...")

    # SAVE training and validation results:
    print('Saving training and validation loss data:')
    np.savetxt("training_loss.csv",training_loss , delimiter=",")
    np.savetxt("valid_loss.csv", valid_loss, delimiter=",")
    # Saving the model
    torch.save(model.state_dict(), "model.pt")
    #Plotting the loss vs epoch graph
    x_axis = np.arange(1,EPOCHS+1)
    plt.figure()
    plt.plot(x_axis, training_loss[0,:], label = 'Train')
    plt.plot(x_axis, valid_loss[0, :], label='Validation')
    plt.title("Mean Loss in each Epoch")
    plt.xlabel('Epochs')
    plt.ylabel('Mean Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
