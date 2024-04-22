from utils import *
from data import *

INV_DATA_MEANS = torch.tensor([-m for m in DATA_MEANS]).view(-1, 1, 1)
INV_DATA_STD = torch.tensor([1.0 / s for s in DATA_STD]).view(-1, 1, 1)

def loading_data(train_loader):
    start_time = time.time()
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    end_time = time.time()
    print(f"Time for loading a batch: {(end_time - start_time):6.5f}s")
    return images, labels

def imshow(img):
        img = img.div_(INV_DATA_STD).sub_(INV_DATA_MEANS)  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis("off")
        plt.show()
        plt.close()

def  download_CIFAR10_data(DATASET_PATH, train_transform,test_transform):
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    return train_dataset, 

def _main_():
    random_seed(seed=seed, deterministic=True)
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset,val_dataset,test_set = download_CIFAR10_data(DATASET_PATH,train_transform,test_transform)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))


    # Create data loaders for later. Adjust batch size if you have a smaller GPU
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=3)
    images,labels =loading_data(train_loader)
    imshow(torchvision.utils.make_grid(images))
    print("GroundTruth (1st row): ", " ".join(f"{classes[labels[j]]:5s}" for j in range(8)))