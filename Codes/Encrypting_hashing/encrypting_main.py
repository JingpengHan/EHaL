import argparse
from dataset_input import *
from patch_encryption_model import *

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
parser = argparse.ArgumentParser()

# the information of data set
parser.add_argument('--dataset_name', dest='dataset', default='SpaceNet')
parser.add_argument('--dataset_classes', dest='classes', type=int, default=20)
parser.add_argument('--data_dir', dest='data_dir', default='./data/')
parser.add_argument('--database_file', dest='database_file', default='database_img.txt')
parser.add_argument('--database_label', dest='database_label', default='database_label.txt')
parser.add_argument('--train_file', dest='train_file', default='train_img.txt')
parser.add_argument('--train_label', dest='train_label', default='train_label.txt')
parser.add_argument('--test_file', dest='test_file', default='test_img.txt')
parser.add_argument('--test_label', dest='test_label', default='test_label.txt')

# training or test the class encryptor, encrypted patch generator and encrypted patch discriminator
parser.add_argument('--train', dest='train', type=bool, default=True)
parser.add_argument('--test', dest='test', type=bool, default=True)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
parser.add_argument('--hashing_network', dest='hash_method', default='DPSH')
parser.add_argument('--backbone', dest='backbone', default='ResNet50')
parser.add_argument('--code_length', dest='bit', type=int, default=32)
parser.add_argument('--load_exist_model_model', dest='load', type=bool, default=False)
parser.add_argument('--beta1', dest='rec_w', type=int, default=200)
parser.add_argument('--beta2', dest='ham_w', type=int, default=10)
parser.add_argument('--beta3', dest='disc_w', type=int, default=1)
parser.add_argument('--patches_dir', dest='patches', default='patches/')
parser.add_argument('--save_dir', dest='save', default='network/')
parser.add_argument('--test_dir', dest='test_dir', default='test/')
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=50)
parser.add_argument('--learning_rate', dest='lr', type=float, default=1e-4)
parser.add_argument('--epoch_count', type=int, default=1)
parser.add_argument('--n_epochs_decay', type=int, default=50)
parser.add_argument('--lr_policy', type=str, default='linear')
parser.add_argument('--lr_decay_iters', type=int, default=50)
parser.add_argument('--train_step', dest='print_freq', type=int, default=50)
parser.add_argument('--patch_step', dest='patches_freq', type=int, default=50)
args = parser.parse_args()

#import the imgae patches and its' labels
dset_database = Datasetinput(args.data_dir + args.dataset, args.database_file, args.database_label)
dset_train = Datasetinput(args.data_dir + args.dataset, args.train_file, args.train_label)
dset_test = Datasetinput(args.data_dir + args.dataset, args.test_file, args.test_label)
num_database, num_test, num_train = len(dset_database), len(dset_test), len(dset_train)
database_loader = DataLoader(dset_database, batch_size=args.batch_size, shuffle=False, num_workers=4)
train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dset_test, batch_size=1, shuffle=False, num_workers=4)

#import the labels and generate the encryption labels
database_labels = load_label(args.database_label, args.data_dir + args.dataset)
train_labels = load_label(args.train_label, args.data_dir + args.dataset)
test_labels = load_label(args.test_label, args.data_dir + args.dataset)
train_encryption_labels = train_labels.unique(dim=0)
test_encryption_labels = test_labels.unique(dim=0)

model = PatchEncryptionModel(args=args)
if args.train:
    if args.load:
        model.load_model()
    model.train_generator(train_loader, train_encryption_labels, train_labels, database_loader, database_labels, num_database, num_train, num_test)

if args.test:
    model.load_model()
    model = model.cuda()
    model.test_generator(test_encryption_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test)
