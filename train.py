import os, random, math
import torch
from tqdm import tqdm
from model import SoundSpotter

LR = 0.01
F = 64
B = 1
SAMPLE_RATE = 44100
DEVICE = "cpu"

NUM_EPOCHS = 1

TEST_SIZE = 5

SAVING = False
SAVE_PATH = "models/model.pt"

model = SoundSpotter(F, B).to(DEVICE)

mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model, LR)

if SAVING:
    try: 
        state_dict, epoch = torch.load(SAVE_PATH)
        model.load_state_dict(state_dict)
        start_epoch = epoch + 1
    except FileNotFoundError:
        pass

for epoch in tqdm(range(NUM_EPOCHS)):
    model.train()
    for i in range(len(list(categories))-TEST_SIZE):
        short_category = list(categories)[i]
        y = math.round(random.triangular(0, 10, 3))
        x = create_random_long(random.triangular(0, 10, 3), short_category)
        short = torch.load(random.choice(os.listdir("dataset_pt_files/" + short_category)))
        short = add_noise(short)
        y_pred = model(x, short)
        loss = mse_loss(y, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    total_val_loss = 0
    for i in range(TEST_SIZE, len(list(categories))):
        short_category = list(categories)[i]
        y = math.round(random.triangular(0, 10, 3))
        x = create_random_long(random.triangular(0, 10, 3), short_category)
        short = torch.load(random.choice(os.listdir("dataset_pt_files/" + short_category)))
        short = add_noise(short) # TODO: implement add_noise()
        y_pred = model(x, short)
        total_val_loss = mse_loss(y, y_pred).item()