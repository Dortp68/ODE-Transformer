from torch.utils.data import DataLoader
from data_prepro import *
from Models import *
import argparse
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', help="dataset", type=str)
parser.add_argument('--train_share', type=int, default=0.8, help='size of train split')
parser.add_argument('--model_name', type=str, default='ODETransformer')
parser.add_argument('--num_epochs', type=int, default=100, help='epochs to train the model')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='the batch size to use')
parser.add_argument('--train_dir', type=str, default='train_dir/')
args = parser.parse_args()

data_path = ''
data_name = args.dataset
if 'sml' in data_name:
    data_path = 'raw_data/sml_rawdata.csv'
elif 'elec' in data_name:
    data_path = 'raw_data/electricity_rawdata.csv'
elif 'etth1' in data_name:
    data_path = 'raw_data/ETTh1.csv'
elif 'etth2' in data_name:
    data_path = 'raw_data/ETTh2.csv'
else:
    print("wrong dataset name")

target_col = 15
indep_col = [0, 14]
win_size = 20
pre_T = 5
method = "rk4"
train_share = args.train_share
batch_size = args.batch_size
lr = args.lr
n_epoch = args.num_epochs

hidden_rnn = 64
hidden_size = 5
att_dim = 50
n_heads = 4
if 'sml' in data_name:
    target_col = 13
    indep_col = [0,12]
    lr = 0.01
    n_epoch = 80
    hidden_rnn = 32
    hidden_size = 5
    att_dim = 30
    n_heads = 4
elif 'elec' in data_name:
    target_col = 15
    indep_col = [0, 14]
    lr = 0.01
    n_epoch = 120
    hidden_rnn = 64
    hidden_size = 5
    att_dim = 30
    n_heads = 4
elif 'etth1' in data_name:
    target_col = 6
    indep_col = [0,5]
    lr = 0.01
    n_epoch = 200
    hidden_rnn = 128
    hidden_size = 7
    att_dim = 45
    n_heads = 6
elif 'etth2' in data_name:
    target_col = 6
    indep_col = [0, 5]
    lr = 0.01
    n_epoch = 150
    hidden_rnn = 128
    hidden_size = 7
    att_dim = 50
    n_heads = 6
train_T = torch.linspace(0., 3., 4).to(device)
test_T = torch.Tensor([0., 1., 1.5, 2., 2.5, 3.]).to(device)
criterion = torch.nn.MSELoss()
criterionL1 = torch.nn.L1Loss()
path = args.train_dir
model_name = args.model_name
log_file = path+data_name + "_" + model_name + "_" + 'trainlogs'
log_metrics = path+data_name + "_" + model_name + "_" + 'metrics'


class RegressionPipeline():
    def __init__(self, model, train_loader, ystd, ymean, seq_len):
        super(RegressionPipeline, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.tar_mean = ymean
        self.tar_std = ystd
        self.seq_len = seq_len
        self.step_losses = []
        self.train_losses = []
        self.valid_mae = []
        self.valid_rmse =[]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, betas=(0.9, 0.999), weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=10)

    def save(self):
        filename = path+model_name+'.pth'
        torch.save(self.model, filename)
        train_data = {"Losses": self.train_losses, "Validation mae": self.valid_mae, "Validation rmse": self.valid_rmse}
        df = pd.DataFrame(train_data)
        df.to_csv(log_file + '.csv', index=False)


    def load(self, filename):
        self.model = torch.load(filename)

    def _train(self, epoch):
        print('At epoch: {}'.format(epoch))
        self.model.train()
        loss_list = []
        for i, (inputs, target) in enumerate(train_loader):
            target = target.to(device)
            train_tar = inputs[:, :, target_col].to(device)
            train_dri = inputs[:, :, indep_col[0]:(indep_col[1] + 1)].to(device)
            ts = torch.linspace(0, self.seq_len - 1, self.seq_len).to(device)
            pred = self.model(train_tar, train_dri, ts, train_T+seq_len)
            pred_y = pred * self.tar_std + self.tar_mean
            # Calculate Loss
            loss = criterion(target, pred_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())
            torch.cuda.empty_cache()
            if i % 20 == 0:
                self.step_losses.append(loss.item())
                print('step_loss:', loss.item())

        total_loss = np.array(loss_list).mean()
        return total_loss

    def test(self, x, y):
        self.model.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = torch.Tensor(x).to(device)
                target = torch.Tensor(y).to(device)
            else:
                inputs = x

            train_tar = inputs[:, :, target_col]
            train_dri = inputs[:, :, indep_col[0]:(indep_col[1] + 1)]
            ts = torch.linspace(0, self.seq_len - 1, self.seq_len).to(device)
            pred = self.model(train_tar, train_dri, ts, test_T + seq_len)
            pred_y = pred * self.tar_std + self.tar_mean

            test_mse = criterion(target, pred_y[:, 1:6])
            test_rmse = torch.sqrt(test_mse)
            test_mae = criterionL1(target, pred_y[:, 1:6])

        return test_rmse, test_mae

    def fit(self, n_epoch):
        best_val_mae = float("inf")
        best_val_rmse = float("inf")

        for epoch in range(n_epoch):
            train_loss = self._train(epoch)
            torch.cuda.empty_cache()
            val_rmse, val_mae = self.test(val_x, val_y)
            self.scheduler.step(val_rmse)
            self.valid_mae.append(val_mae.to('cpu'))
            self.valid_rmse.append(val_rmse.to('cpu'))
            self.train_losses.append(train_loss)
            print('Train_Loss: {}. Validation mae: {}. Validation rmse: {}'.format(train_loss, val_mae, val_rmse))
            torch.cuda.empty_cache()

            if val_rmse < best_val_rmse:
                best_val_mae = val_mae
                best_val_rmse = val_rmse
                test_rmse, test_mae = self.test(test_x, test_y)

                print('best_val_mae: {}. best_val_rmse: {}'.format(best_val_mae, best_val_rmse))
                print('test_mae: {}. test_rmse: {}'.format(test_mae, test_rmse))
        print('best_val_mae: {}. best_val_rmse: {}'.format(best_val_mae, best_val_rmse))

    def evaluate_metrics(self, x, y):
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = torch.Tensor(x).to(device)
                target = torch.Tensor(y).to(device)
            else:
                inputs = x

            train_tar = inputs[:, :, target_col]
            train_dri = inputs[:, :, indep_col[0]:(indep_col[1] + 1)]
            ts = torch.linspace(0, self.seq_len - 1, self.seq_len).to(device)
            pred = self.model(train_tar, train_dri, ts, test_T + seq_len)
            pred_y = pred * self.tar_std + self.tar_mean

            test_mse = criterion(target, pred_y[:, 1:6])
            test_rmse = torch.sqrt(test_mse)
            test_mae = criterionL1(target, pred_y[:, 1:6])
            test_mse1 = criterion(target[:, [0]], pred_y[:, [1]])
            test_rmse1 = torch.sqrt(test_mse1)
            test_mae1 = criterionL1(target[:, [0]], pred_y[:, [1]])
            test_mse2 = criterion(target[:, [1]], pred_y[:, [2]])
            test_rmse2 = torch.sqrt(test_mse2)
            test_mae2 = criterionL1(target[:, [1]], pred_y[:, [2]])
            test_mse3 = criterion(target[:, [2]], pred_y[:, [3]])
            test_rmse3 = torch.sqrt(test_mse3)
            test_mae3 = criterionL1(target[:, [2]], pred_y[:, [3]])
            test_mse4 = criterion(target[:, [3]], pred_y[:, [4]])
            test_rmse4 = torch.sqrt(test_mse4)
            test_mae4 = criterionL1(target[:, [3]], pred_y[:, [4]])
            test_mse5 = criterion(target[:, [4]], pred_y[:, [5]])
            test_rmse5 = torch.sqrt(test_mse5)
            test_mae5 = criterionL1(target[:, [4]], pred_y[:, [5]])

            return test_mae, test_rmse, test_mae1, test_rmse1, test_mae2, test_rmse2, test_mae3, test_rmse3, test_mae4, \
                test_rmse4, test_mae5, test_rmse5, pred_y




if __name__ == '__main__':
    generator = Datagen(data_path, target_col, indep_col, win_size, pre_T, train_share)
    train_x, train_y, val_x, val_y, test_x, test_y, y_mean, y_std = generator.generate_arbitarystep2()
    ystd = torch.Tensor([y_std]).to(device)
    ymean = torch.Tensor([y_mean]).to(device)
    dataset_train = subDataset(train_x, train_y)
    dataset_val = subDataset(val_x, val_y)
    dataset_test = subDataset(test_x, test_y)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)

    input_size = train_x.shape[-1]
    driving_size = input_size - 1
    seq_len = train_x.shape[1]

    model = ODETransformer(n_heads, driving_size, att_dim, hidden_rnn, hidden_size, 1, method, device).to(device)
    pipeline = RegressionPipeline(model, train_loader, ystd, ymean, seq_len)
    pipeline.fit(n_epoch)
    pipeline.save()
    total_mae, total_rmse, mae_1, rmse_1, mae_2, rmse_2, mae_3, rmse_3, mae_4, rmse_4, mae_5, rmse_5, prediction = pipeline.evaluate_metrics(test_x, test_y)
    with open(log_metrics, "a") as text_file:
        text_file.write("\n testing error: %s \n\n" % (
        [total_mae, total_rmse, mae_1, rmse_1, mae_2, rmse_2, mae_3, rmse_3, mae_4, rmse_4, mae_5, rmse_5]))
    prediction = prediction.to("cpu")
    train_data = {"step1": prediction[:,0], "step1.5": prediction[:,1], "step2": prediction[:,2], "step2.5": prediction[:,3], "step3": prediction[:,4]}
    df = pd.DataFrame(train_data)
    df.to_csv(path+data_name + model_name + '_predicions.csv', index=False)

