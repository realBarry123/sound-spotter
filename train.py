import os, random
import torch
from tqdm import tqdm
from model import SoundSpotter
from random_long import create_random_long, categories
from add_noise import add_noise

LR = 0.01
F = 64
B = 1
SAMPLE_RATE = 44100
DEVICE = "cpu"

NUM_EPOCHS = 4

TEST_SIZE = 5

SAVING = False
SAVE_PATH = "models/model.pt"

model = SoundSpotter(F, B).to(DEVICE)

mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

if SAVING:
    try: 
        state_dict, epoch = torch.load(SAVE_PATH)
        model.load_state_dict(state_dict)
        start_epoch = epoch + 1
    except FileNotFoundError:
        pass

for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0
    train_size = len(list(categories)) - TEST_SIZE
    print(f"Epoch {epoch}/{NUM_EPOCHS}")
    for i in tqdm(range(train_size), desc=f"Training epoch {epoch}"):
        short_category = list(categories)[i]
        y = round(random.triangular(0, 5, 2))
        x = create_random_long(int(y), short_category)

        short = torch.load("dataset_pt_files/" + short_category + "/" + random.choice(os.listdir("dataset_pt_files/" + short_category)))

        short = add_noise(short)
        short = short.unsqueeze(0)

        y_pred = model(x, short)
        y = torch.tensor([[y]]).float()

        loss = mse_loss(y, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    print(f"Average train loss: {total_train_loss / train_size}")

    model.eval()
    total_val_loss = 0
    for i in tqdm(range(train_size, len(list(categories))), desc=f"Testing epoch {epoch}"):
        short_category = list(categories)[i]
        y = round(random.triangular(0, 5, 2))
        x = create_random_long(int(y), short_category)
        short = torch.load("dataset_pt_files/" + short_category + "/" + random.choice(os.listdir("dataset_pt_files/" + short_category)))
        short = add_noise(short)
        short = short.unsqueeze(0)
        y_pred = model(x, short)
        y = torch.tensor([[y]]).float()
        total_val_loss = mse_loss(y, y_pred).item()

    print(f"Average validation loss: {total_val_loss / TEST_SIZE}")

    if SAVING:
        torch.save([model.state_dict(), epoch], SAVE_PATH)
