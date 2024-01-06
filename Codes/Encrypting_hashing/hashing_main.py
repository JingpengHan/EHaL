import argparse
from torch.utils.data import DataLoader
from DPSH import *
from DPH import *
from HashNet import *
from dataset_input import *

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
parser = argparse.ArgumentParser()

# the information of data set
parser.add_argument('--dataset_name', dest='dataset', default='SpaceNet')
parser.add_argument('--data_dir', dest='data_dir', default='./data/')
parser.add_argument('--database_file', dest='database_file', default='database_img.txt')
parser.add_argument('--database_label', dest='database_label', default='database_label.txt')
parser.add_argument('--train_file', dest='train_file', default='train_img.txt')
parser.add_argument('--train_label', dest='train_label', default='train_label.txt')
parser.add_argument('--test_file', dest='test_file', default='test_img.txt')
parser.add_argument('--test_label', dest='test_label', default='test_label.txt')

# train or test the deep hashing network
parser.add_argument('--train', dest='train', type=bool, default=True)
parser.add_argument('--test', dest='test', type=bool, default=True)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
parser.add_argument('--hashing_network', dest='method', default='DPSH')
parser.add_argument('--backbone', dest='backbone', default='ResNet50')
parser.add_argument('--code_length', dest='bit', type=int, default=64)
parser.add_argument('--load_exist_model', dest='load', type=bool, default=False)
parser.add_argument('--epoch', dest='n_epochs', type=int, default=50)
parser.add_argument('--learning_rate', dest='lr', type=float, default=0.05)
parser.add_argument('--weight_decay', dest='wd', type=float, default=1e-5)
parser.add_argument('--save_dir', dest='save', default='network/')
args = parser.parse_args()


dset_database = Datasetinput(args.data_dir + args.dataset, args.database_file, args.database_label)
dset_train = Datasetinput(args.data_dir + args.dataset, args.train_file, args.train_label)
dset_test = Datasetinput(args.data_dir + args.dataset, args.test_file, args.test_label)
num_database, num_test, num_train = len(dset_database), len(dset_test), len(dset_train)


database_loader = DataLoader(dset_database, batch_size=args.batch_size, shuffle=False, num_workers=4)
database_labels = load_label(args.database_label, args.data_dir + args.dataset)
train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
train_labels = load_label(args.train_label, args.data_dir + args.dataset)
test_loader = DataLoader(dset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_labels = load_label(args.test_label, args.data_dir + args.dataset)

model = None
if args.method == 'DPSH':
    model = DPSH(args)
elif args.method == 'DPH':
    model = DPH(args)
else:
    model = HashNet(args)

if args.train:
    model.train(train_loader, train_labels, num_train)

if args.test:
    model.load_model()
    model.test(database_loader, test_loader, database_labels, test_labels, num_database, num_test)
