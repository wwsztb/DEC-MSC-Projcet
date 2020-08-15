import click
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from tensorboardX import SummaryWriter
import uuid

from sklearn.preprocessing import MinMaxScaler
from ptdec.dec import DEC
from ptdec.model import train, predict
from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
from ptdec.utils import cluster_accuracy

aa = pd.DataFrame()

train1 = pd.read_csv('data/train1.csv') #,header=None)
trainlabel = pd.read_csv('data/trainlabel.csv',header=None)
test1 = pd.read_csv('data/train1.csv') #,header=None)
testlabel = pd.read_csv('data/trainLabel.csv',header=None)

scaler_2 = MinMaxScaler(feature_range=(0, 1))  #自动将dtype转换成float64
tr = scaler_2.fit_transform(train1)

scaler_2 = MinMaxScaler(feature_range=(0, 1))  #自动将dtype转换成float64
te = scaler_2.fit_transform(test1)

#tr = train1.values
tr = torch.from_numpy(tr)
tr = torch.tensor(tr, dtype=torch.float32)
trl = trainlabel.values
trl = torch.from_numpy(trl).int()


#te = test1.values
te = torch.from_numpy(te)
te = torch.tensor(te, dtype=torch.float32)
tel = testlabel.values
tel = torch.from_numpy(tel).int()

print(tr.shape)
print(trl.shape)
print(te.shape)
print(tel.shape)
tr = torch.utils.data.TensorDataset(tr,trl)
te = torch.utils.data.TensorDataset(te,tel)


class CachedKDDTr(Dataset):
    def __init__(self, train, cuda, testing_mode=False):
        img_transform = transforms.Compose([transforms.Lambda(self._transformation)])
        self.ds = tr
        self.cuda = cuda
        self.testing_mode = testing_mode
        self._cache = dict()

    @staticmethod
    def _transformation(img):
        return (
            torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float()
            * 0.02
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            self._cache[index] = list(self.ds[index])
            if self.cuda:
                self._cache[index][0] = self._cache[index][0].cuda(non_blocking=True)
                self._cache[index][1] = self._cache[index][1].cuda(non_blocking=True)
        return self._cache[index]

    def __len__(self) -> int:
        return 128 if self.testing_mode else len(self.ds)

    
class CachedKDDTe(Dataset):
    def __init__(self, train, cuda, testing_mode=False):
        img_transform = transforms.Compose([transforms.Lambda(self._transformation)])
        self.ds = te
        self.cuda = cuda
        self.testing_mode = testing_mode
        self._cache = dict()

    @staticmethod
    def _transformation(img):
        return (
            torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float()
            * 0.02
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            self._cache[index] = list(self.ds[index])
            if self.cuda:
                self._cache[index][0] = self._cache[index][0].cuda(non_blocking=True)
                self._cache[index][1] = self._cache[index][1].cuda(non_blocking=True)
        return self._cache[index]

    def __len__(self) -> int:
        return 128 if self.testing_mode else len(self.ds)

   

@click.command()
@click.option(
    "--cuda", help="whether to use CUDA (default False).", type=bool, default=False
)
@click.option(
    "--batch-size", help="training batch size (default 256).", type=int, default=2048
)
@click.option(
    "--pretrain-epochs",
    help="number of pretraining epochs (default 300).",
    type=int,
    default=50,
)
@click.option(
    "--finetune-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=100,
)
@click.option(
    "--testing-mode",
    help="whether to run in testing mode (default False).",
    type=bool,
    default=False,
)
def main(cuda, batch_size, pretrain_epochs, finetune_epochs, testing_mode):
    writer = SummaryWriter()  # create the TensorBoard object
    # callback function to call during training, uses writer from the scope

    def training_callback(epoch, lr, loss, validation_loss):
        writer.add_scalars(
            "data/autoencoder",
            {"lr": lr, "loss": loss, "validation_loss": validation_loss,},
            epoch,
        )

    ds_train = CachedKDDTr(
        train=True, cuda=cuda, testing_mode=testing_mode
    )  # training dataset
    ds_val = CachedKDDTe(
        train=False, cuda=cuda, testing_mode=testing_mode
    )  # evaluation dataset
    autoencoder = StackedDenoisingAutoEncoder(
        [122, 500, 1000, 64, 5], final_activation=None
    )
    if cuda:
        autoencoder.cuda()
    print("Pretraining stage.")
    ae.pretrain(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
        scheduler=lambda x: StepLR(x, 100, gamma=0.1),
        corruption=0.2,
    )
    print("Training stage.")
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
    ae.train(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=finetune_epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
        corruption=0.2,
        update_callback=training_callback,
    )
    print("DEC stage.")
    model = DEC(cluster_number=2, hidden_dimension=2, encoder=autoencoder.encoder)
    if cuda:
        model.cuda()
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(
        dataset=ds_train,
        model=model,
        epochs=100,
        batch_size=1024,
        optimizer=dec_optimizer,
        stopping_delta=0.00000001,
        cuda=cuda,
    )
    predicted, actual = predict(
        ds_train, model, 1024, silent=True, return_actual=True, cuda=cuda
    )
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    reassignment, accuracy = cluster_accuracy(actual, predicted)
    print("Final DEC accuracy: %s" % accuracy)
    if not testing_mode:
        predicted_reassigned = [
            reassignment[item] for item in predicted
        ]  # TODO numpify
        confusion = confusion_matrix(actual, predicted_reassigned)
        normalised_confusion = (
            confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
        )
        confusion_id = uuid.uuid4().hex
        sns.heatmap(normalised_confusion).get_figure().savefig(
            "confusion_%s.png" % confusion_id
        )
        print("Writing out confusion diagram with UUID: %s" % confusion_id)
        writer.close()
    actual = actual.reshape(-1)
    predicted = predicted.reshape(-1)
    aa['actual'] = actual
    aa['predicted'] = predicted
    aa.to_csv('ret.csv',index=None)

if __name__ == "__main__":
    main()
