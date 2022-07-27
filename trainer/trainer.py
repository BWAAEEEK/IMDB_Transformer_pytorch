import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Transformer

class Trainer:
    def __init__(self, model: Transformer, data_loader: DataLoader, args):

        cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.model = model

        # if torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for model" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=args.cuda_devices)

        # Setting the train and test data loader
        self.data_loader = data_loader

        # Setting Adam with hyper parameter
        self.optim = Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        print("Total Model Parameters :", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        data_iter = tqdm(enumerate(self.data_loader),
                         desc="EP_train:%d" % epoch,
                         total=len(self.data_loader),
                         bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        accuracy = 0.0
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            outputs = self.model.forward(data["input"], data["input_position"])


            loss = self.criterion(outputs, data["label"])
            loss.backward()

            # calculate accuracy
            pred = torch.argmax(outputs, 1)

            corrects = (pred == data["label"])

            # print(pred)
            # print(corrects)

            acc = corrects.sum().float() / float(data["label"].size(0))

            avg_loss += loss.item()
            accuracy += acc

            post_fix = {
                "epoch": epoch + 1,
                "iter": i+1,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item(),
                "acc": acc
                }

        print(f"EP{epoch}_train, avg_loss={avg_loss / len(data_iter)}, accuracy={accuracy / len(data_iter)}")

    def save(self, epoch, file_path="transformer_trained.model"):
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path




