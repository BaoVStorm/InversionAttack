import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pickle
import math
import PIL.Image as Image

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())

class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs # img paths
        self.labs = labs # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab

def lfw_dataset(lfw_path, shape_img):
    images_all = []
    labels_all = []
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst

def create_Visualize_Loss(type, method, epoch_points, history, image_history, save_path, imidx_value, imidx_list):
    # Create visualization with images and loss values
    fig, ax = plt.subplots(figsize=(25, 12))

    history = history[::10]

    # print("epoch_points: ", epoch_points )
    # print("history: ", history )
    # print("image_history: ", image_history )

    # Plot loss curve 
    ax.plot(epoch_points, history, 'b-', linewidth=2, zorder=1)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel(f'L2 {type}', fontsize=12)
    ax.set_title(f'Epoch - {type} progress', fontsize=14)

    # Set logarithmic scale
    ax.set_yscale('log')
    min_value = 1e-6
    while min_value * 10 <= min(history):
        min_value *= 10

    ax.set_ylim(min_value, max(history))
    ax.grid(True, alpha=0.3)

    # Add images and loss values
    for x, y, img in zip(epoch_points, history, image_history):
        # Add image
        imagebox = OffsetImage(np.array(img),
                            zoom=0.8,
                            resample=True)
        
        ab_img = AnnotationBbox(imagebox, (x, y),
                            xybox=(0, 40),
                            xycoords='data',
                            boxcoords="offset points",
                            frameon=True,
                            bboxprops=dict(facecolor='white',
                                            edgecolor='gray',
                                            alpha=0.9))

        ax.add_artist(ab_img)

        # Add loss value text below image
        ax.annotate(f'{y:.2e}',  # Scientific notation
                    xy=(x, y),
                    xytext=(0, -20),  # Position below point
                    textcoords='offset points',
                    ha='center',
                    va='top',
                    bbox=dict(facecolor='white',
                            edgecolor='none',
                            alpha=0.7))

    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig('%s/%05d_%s_%s_on_%s.png' % (save_path, imidx_value, method, type, imidx_list))
    plt.close()

def main(dataset = 'MNIST', num_exp = 40):
    print("=============== Load in dataset ===============")

    root_path = '.'
    data_path = os.path.join(root_path, '../data').replace('\\', '/')
    save_path = os.path.join(root_path, 'results/attack/Attack_%s'%dataset).replace('\\', '/')
    
    lr = 1.0
    num_dummy = 1
    Iteration = 300

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ''' load data '''
    if dataset == 'MNIST':
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 588
        dst = datasets.MNIST(root="dataset/MNIST/.", download=False)

    elif dataset == 'CIFAR100':
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        hidden = 768
        dst = datasets.CIFAR100(root="dataset/CIFAR100/.", download=False)

    elif dataset == 'LFW':
        shape_img = (32, 32)
        num_classes = 5749
        channel = 3
        hidden = 768
        lfw_path = os.path.join(root_path, 'dataset/LFW/lfw-py/lfw_funneled')
        dst = lfw_dataset(lfw_path, shape_img)

    else:
        exit('unknown dataset')

    # varibale for final draw model
    label_count_DLG = 0
    label_count_iDLG = 0
    
    skip_exp = 0
    size_MSE = (num_exp - skip_exp) * Iteration
    array_MSE = ["0.01", "0.005", "0.001", "0.0005", "0.0001"]
    array_MSE_DLG = [0, 0, 0, 0, 0]
    array_MSE_iDLG = [0, 0, 0, 0, 0]
    
    ''' train DLG and iDLG '''
    for idx_net in range(num_exp):
        cur_savepath = save_path + f"/exp_{idx_net}"
        
        os.makedirs(cur_savepath, exist_ok=True)

        net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
        net.apply(weights_init)

        print('running %d|%d experiment'%(idx_net, num_exp))
        net = net.to(device)
        idx_shuffle = np.random.permutation(len(dst))

        loss_values_DLG = []
        loss_values_iDLG = []
        mse_values_DLG = []
        mse_values_iDLG = []

        step = 30
        check_skip = False

        for method in ['DLG', 'iDLG']:
            print('%s, Try to generate %d images' % (method, num_dummy))

            criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []

            for imidx in range(num_dummy):
                idx = idx_shuffle[imidx]
                imidx_list.append(idx)
                tmp_datum = tt(dst[idx][0]).float().to(device)
                tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
                tmp_label = tmp_label.view(1, )
                if imidx == 0:
                    gt_data = tmp_datum
                    gt_label = tmp_label
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)

            # compute original gradient
            out = net(gt_data)
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, net.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))

            # generate dummy data and label
            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

            if method == 'DLG':
                optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
            elif method == 'iDLG':
                optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
                # predict the ground-truth label
                label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)

            history = []
            history_iters = []
            losses = []
            mses = []
            train_iters = []

            print('lr =', lr)

            for iters in range(Iteration):

                def closure():
                    optimizer.zero_grad()
                    pred = net(dummy_data)
                    if method == 'DLG':
                        dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                        # dummy_loss = criterion(pred, gt_label)
                    elif method == 'iDLG':
                        dummy_loss = criterion(pred, label_pred)

                    dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward()
                    return grad_diff

                optimizer.step(closure)
                current_loss = closure().item()
                train_iters.append(iters)
                losses.append(current_loss)
                mses.append(torch.mean((dummy_data-gt_data)**2).item())

                if iters % int(Iteration / 30) == 0:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    print(current_time, iters, 'loss = %.8f, mse = %.8f' %(current_loss, mses[-1]))
                    history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                    history_iters.append(iters)

                    for imidx in range(num_dummy):
                        plt.figure(figsize=(12, 8))
                        plt.subplot(3, 10, 1)
                        plt.imshow(tp(gt_data[imidx].cpu()))
                        for i in range(min(len(history), 29)):
                            plt.subplot(3, 10, i + 2)
                            plt.imshow(history[i][imidx])                            
                            plt.title('iter=%d' % (history_iters[i]))
                            plt.axis('off')
                        if method == 'DLG':
                            plt.savefig('%s/%05d_DLG_on_%s.png' % (cur_savepath, imidx_list[imidx], imidx_list))
                            plt.close()
                        elif method == 'iDLG':
                            plt.savefig('%s/%05d_iDLG_on_%s.png' % (cur_savepath, imidx_list[imidx], imidx_list))
                            plt.close()

                    # if current_loss < 0.000001: # converge
                    #     break

                # check loss (nan OR high loss)
                if math.isnan(current_loss) or (iters >= 290 and current_loss >= 900):
                    check_skip = True
                    break

                if method == 'DLG':
                    mse_values_DLG.append(mses[-1])
                    loss_values_DLG.append(current_loss)
                    
                    for i, value in enumerate(array_MSE):
                        if mses[-1] <= float(value):
                            array_MSE_DLG[i] += 1
                    
                elif method == 'iDLG':
                    mse_values_iDLG.append(mses[-1])
                    loss_values_iDLG.append(current_loss)
                    
                    for i, value in enumerate(array_MSE):
                        if mses[-1] <= float(value):
                            array_MSE_iDLG[i] += 1

            if check_skip:
                break

            if method == 'DLG':
                loss_DLG = losses
                label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                mse_DLG = mses

                create_Visualize_Loss("MSE", method, history_iters, mses, [img[0] for img in history], cur_savepath, imidx_list[imidx], imidx_list)
                create_Visualize_Loss("Loss", method, history_iters, losses, [img[0] for img in history], cur_savepath, imidx_list[imidx], imidx_list)

            elif method == 'iDLG':
                loss_iDLG = losses
                label_iDLG = label_pred.item()
                mse_iDLG = mses

                create_Visualize_Loss("MSE", method, history_iters, mses, [img[0] for img in history], cur_savepath, imidx_list[imidx], imidx_list)
                create_Visualize_Loss("Loss", method, history_iters, losses, [img[0] for img in history], cur_savepath, imidx_list[imidx], imidx_list)

        if check_skip:
            skip_exp += 1

            # Thông báo exp đã được skip
            plt.figure(figsize=(4, 4))
            plt.text(0.5, 0.5, "Skip", fontsize=20, ha='center', va='center')
            plt.axis('off')
            plt.savefig('%s/skip.png' % (cur_savepath))
            plt.close()
            
            print(f" ============ Skip Exp {idx_net} ============ \n")
            continue

        # Draw MSE
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(5, step + 1), mse_values_DLG[::10][4:], 'go-', label='DLG')
        plt.plot(np.arange(5, step + 1), mse_values_iDLG[::10][4:], 'r*-', label='iDLG')
        # plt.ticklabel_format(style='plain', axis='x')
        plt.xlabel('step of Iteration')
        plt.ylabel('Fidelity Threshold (MSE)')
        plt.title(dataset)
        plt.legend()
        plt.grid(True)
        plt.savefig('%s/%05d_MSE_DLG_and_iDLG.png' % (cur_savepath, imidx_list[imidx]))
        plt.close()

        # Draw Loss
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(5, step + 1), loss_values_DLG[::10][4:], 'go-', label='DLG')
        plt.plot(np.arange(5, step + 1), loss_values_iDLG[::10][4:], 'r*-', label='iDLG')
        # plt.ticklabel_format(style='plain', axis='x')
        plt.xlabel('step of Iteration')
        plt.ylabel('Loss')
        plt.title(dataset)
        plt.legend()
        plt.grid(True)
        plt.savefig('%s/%05d_Loss_DLG_and_iDLG.png' % (cur_savepath, imidx_list[imidx]))
        plt.close()
        
        print("\n== Save Success grid ==\n")
        
        print('-------- INFO --------')
        lab_gt = gt_label.detach().cpu().data.numpy()
        print('imidx_list:', imidx_list)
        print('loss_DLG:', loss_DLG[-1], 'loss_iDLG:', loss_iDLG[-1])
        print('mse_DLG:', mse_DLG[-1], 'mse_iDLG:', mse_iDLG[-1])
        print('gt_label:', lab_gt, 'lab_DLG:', label_DLG, 'lab_iDLG:', label_iDLG)
        print('----------------------\n\n')

        if lab_gt == label_DLG:
            label_count_DLG += 1
        
        if lab_gt == label_iDLG:
            label_count_iDLG += 1

    size_MSE = (num_exp - skip_exp) * Iteration

    # vẽ biểu đồ chính xác (label)
    cities = ['DLG', 'iDLG']
    percentages = [label_count_DLG * 100 / (num_exp - skip_exp), label_count_iDLG * 100 / (num_exp - skip_exp)]  
    plt.bar(cities, percentages, color=['blue', 'red'])
    plt.title(f'Tỉ lệ dự đoán đúng nhãn dataset {dataset}')
    plt.xlabel('Mô hình tấn công')
    plt.ylabel('Tỉ lệ chính xác (%)')
    for i, value in enumerate(percentages):
        plt.text(i, value + 1, f"{value}%", ha='center')
    plt.savefig('%s/Accuracy_Label_on_DLG_iDLG.png' % (save_path))
    plt.close()

    # vẽ biểu đồ tỉ lệ MSE
    plt.figure(figsize=(10, 6))
    plt.plot(array_MSE, [x * 100 / size_MSE for x in array_MSE_DLG], 'go-', label='DLG')
    plt.plot(array_MSE, [x * 100 / size_MSE for x in array_MSE_iDLG], 'r*-', label='iDLG')
    plt.xlabel('Fidelity Threshold (MSE)')
    plt.ylabel('% Good Fidelity')
    plt.xticks(array_MSE)
    plt.yticks([0, 25, 50, 75, 100])
    plt.title(f"dataset {dataset}")
    plt.legend()
    plt.grid(True)
    plt.savefig('%s/Good_Fidelity_DLG_and_iDLG.png' % (save_path))
    plt.close()

    print(f"\n\n ================= skip: {skip_exp} ================= \n\n")

    cities = ["Skip (bad)", "Dont Skip (good)"]
    percentages = [skip_exp * 100 / num_exp, (num_exp - skip_exp) * 100 / num_exp]  
    plt.bar(cities, percentages, color=['blue', 'red'])
    plt.title(f'Tỉ lệ skip trên dataset {dataset}')
    plt.xlabel('Thử nghiệm (Exp)')
    plt.ylabel('Tỉ lệ xuất hiện trên tất cả thử nghiệm (%)')
    for i, value in enumerate(percentages):
        plt.text(i, value + 1, f"{value}%", ha='center')
    plt.savefig('%s/per_skip_on_DLG_iDLG.png' % (save_path))
    plt.close()

if __name__ == '__main__':
    for dataset in ["MNIST", "CIFAR100", "LFW"]:    
        # dataset và số lần thử nghiệm
        main(dataset = dataset, num_exp = 40)