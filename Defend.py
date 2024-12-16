# %matplotlib inline
import random
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
torch.manual_seed(50)

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# print(torch.__version__, torchvision.__version__)

def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
    
class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        feature = out.view(out.size(0), -1)
        #print(feature.size())
        out = self.fc(feature)
        return out, feature

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


def create_Visualize(defend, type, epoch_points, history, image_history, save_path):
    # Create visualization with images and loss values
    fig, ax = plt.subplots(figsize=(25, 12))

    epoch_points = epoch_points[::10]
    history = history[::10]
    image_history = image_history[::10]

    # Plot loss curve 
    ax.plot(epoch_points, history, 'b-', linewidth=2, zorder=1)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel(f'L2 {type}', fontsize=12)
    ax.set_title(f'Epoch - {defend} progress', fontsize=14)

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
    plt.savefig('%s/%s/%s_%s.png' % (save_path, defend, defend, type))
    plt.close()


def process(defend, dataset, save_path, index_img = -1):
    if dataset == 'MNIST':
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 588
        dst = datasets.MNIST(root="MNIST/.", download=False)

    elif dataset == 'CIFAR100':
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        hidden = 768
        dst = datasets.CIFAR100(root="CIFAR100/.", download=False)

    elif dataset == 'LFW':
        shape_img = (32, 32)
        num_classes = 5749
        channel = 3
        hidden = 768
        lfw_path = os.path.join(".", 'LFW/lfw-py/lfw_funneled')
        dst = lfw_dataset(lfw_path, shape_img)

    # dst = datasets.CIFAR10("~/.torch", download=True)

    tp = transforms.Compose([
        transforms.Resize(shape_img[0]),
        transforms.CenterCrop(shape_img[0]),
        transforms.ToTensor()
    ])
    tt = transforms.ToPILImage()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    # print("Running on %s" % device)

    print("=== dataset:", dataset, "- defend:", defend, "===\n")

    net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes).to(device)
    net.apply(weights_init)

    criterion = cross_entropy_for_onehot

    # print("------ image ------")
    if defend == "nan":
        img_index = random.randint(0, len(dst))
    else:
        img_index = index_img 

    print("img_index: ", img_index)
    
    gt_data = tp(dst[img_index][0]).to(device)
    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label, num_classes=num_classes)

    gt_data.requires_grad = True

    label_gt = torch.argmax(gt_onehot_label, dim=-1).item()
    print("Onehot label is %d." % label_gt)

    if defend == "nan":
        plt.imshow(tt(gt_data[0].cpu()))
        plt.title("Ground truth image")
        plt.savefig('%s/img_Dataset.png' % (save_path))
        plt.close()

    # ------------------------------------ PHÒNG THỦ ------------------------------------
    # compute ||dr/dX||/||r|| 
    out, feature_fc1_graph = net(gt_data)
    deviation_f1_target = torch.zeros_like(feature_fc1_graph)
    deviation_f1_x_norm = torch.zeros_like(feature_fc1_graph)
    for f in range(deviation_f1_x_norm.size(1)):
        deviation_f1_target[:,f] = 1
        feature_fc1_graph.backward(deviation_f1_target, retain_graph=True)
        deviation_f1_x = gt_data.grad.data
        deviation_f1_x_norm[:,f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1)/(feature_fc1_graph.data[:,f])
        net.zero_grad()
        gt_data.grad.data.zero_()
        deviation_f1_target[:,f] = 0

    # prune r_i corresponding to smallest ||dr_i/dX||/||r_i||
    deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
    thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), 1)
    mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)
    # print(sum(mask))

    y = criterion(out, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())

    #print(dy_dx[4].shape)

    # share the gradients with other clients
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    print()

    if defend == "MG":
        print("=== Apply Masking Gradients Defend ===")

        # Masking Gradients
        original_dy_dx[8] = original_dy_dx[8] * torch.Tensor(mask).to(device)

        plt.figure()
        plt.plot(deviation_f1_x_norm_sum.cpu().numpy(), label='||dev r||')
        plt.plot(feature_fc1_graph.cpu().detach().numpy().flatten(), label='||r||')
        plt.legend()
        plt.savefig('%s/%s/MG_Visualize.png' % (save_path, defend))
        plt.close()

    elif defend == "DP":
        print("=== Apply Different Privacy Defend ===")

        # differential privacy
        for i in range(len(original_dy_dx)):
            grad_tensor = original_dy_dx[i].cpu().numpy()
            noise = np.random.laplace(0,1e-1, size=grad_tensor.shape)
            grad_tensor = grad_tensor + noise
            original_dy_dx[i] = torch.Tensor(grad_tensor).to(device)

    elif defend == "MC":
        print("=== Apply Model Compression Defend ===")

        # model compression
        for i in range(len(original_dy_dx)):
            grad_tensor = original_dy_dx[i].cpu().numpy()
            flattened_weights = np.abs(grad_tensor.flatten())
            # Generate the pruning threshold according to 'prune by percentage'. (Your code: 1 Line) 
            thresh = np.percentile(flattened_weights, 10)
            grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
            original_dy_dx[i] = torch.Tensor(grad_tensor).to(device)

    else:
        print("=== NOT Apply Defend ===")

    # ------------------------------------ TẤN CÔNG ------------------------------------
    # generate dummy data and label
    dummy_data_init = torch.randn(gt_data.size())
    dummy_label_init = torch.randn(gt_onehot_label.size())

    # generate dummy data and label
    dummy_data = torch.Tensor(dummy_data_init).to(device).requires_grad_(True)
    dummy_label = torch.Tensor(dummy_label_init).to(device).requires_grad_(True)

    # plt.imshow(tt(dummy_data[0].cpu()))
    # plt.title("Dummy data")
    # print("Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item())

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=1.0 )
    #optimizer = torch.optim.SGD([dummy_data, dummy_label], lr=0.1, momentum=0.9 )

    history = []
    loss_arr = []
    mse_arr = []
    history_iters = []

    MSE_min = 1000
    for iters in range(300):
        def closure():
            optimizer.zero_grad()
            #out, [feature_fc1_graph, feature_fc2_graph, feature_fc3_graph] = net(gt_data)
            pred, f1 = net(dummy_data) 
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(pred, dummy_onehot_label) # TODO: fix the gt_label to dummy_label in both code and slides.
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            
            grad_diff = 0
            grad_count = 0
            
            i = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx): # TODO: fix the variablas here
                if i <= 100:
                    grad_diff += ((gx - gy) ** 2).sum()
                    grad_count += gx.nelement()
                i += 1
            #grad_diff = grad_diff / grad_count * 1000
            
            #grad_diff = ((feature_fc1_graph - f1) ** 2).sum()
            grad_diff.backward()
            
            return grad_diff
        
        optimizer.step(closure)

        current_loss = closure()

        history_iters.append(iters)
        loss_arr.append(current_loss.item())
        mse_arr.append((gt_data[0] - dummy_data[0]).pow(2).mean().item())

        if MSE_min > mse_arr[-1]:
            MSE_min = mse_arr[-1]
        if iters % 10 == 0: 
            print("{}, loss: {}, MSE: {}, MSE_min: {}".format(history_iters[-1], loss_arr[-1], mse_arr[-1], MSE_min))
        history.append(tt(dummy_data[0].cpu()))
        
    label_pd = torch.argmax(dummy_label, dim=-1).item()

    # Xuất biểu đồ tấn công
    plt.figure(figsize=(12, 8))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i * 10])
        plt.title("iter=%d" % (i * 10))
        plt.axis('off')
    plt.savefig('%s/%s/%s_GradientInversionAttack.png' % (save_path, defend, defend))
    plt.close()
    print("\nDummy label is %d." %label_pd)

    create_Visualize(defend, "MSE", history_iters, mse_arr, history, save_path)
    create_Visualize(defend, "Loss", history_iters, loss_arr, history, save_path)

    return loss_arr, mse_arr, MSE_min, label_pd, img_index, label_gt

def main(dataset, exp_num = 1):
    step = 30

    for i in range(exp_num):
        print(f"\n============================ Experiment: {i} ============================\n")

        cur_savepath = f'results/defend/dataset_{dataset}/exp_{i}'
        save_path = os.path.join('.', cur_savepath).replace('\\', '/')

        os.makedirs(cur_savepath, exist_ok=True)

        loss_list = {}
        mse_list = {}
        min_list = {}
        label_list = {}
        index_img = -1
        label_gt = -1

        # non def
        os.makedirs(f"{cur_savepath}/nan", exist_ok=True)
        loss_list["nan"], mse_list["nan"], min_list["nan"], label_list["nan"], index_img, label_gt = process(defend = "nan", dataset = dataset, save_path = save_path)

        if loss_list["nan"][-1] >= 10:
            # Thông báo exp đã được skip
            plt.figure(figsize=(4, 4))
            plt.text(0.5, 0.5, "Skip", fontsize=20, ha='center', va='center')
            plt.axis('off')
            plt.savefig('%s/skip.png' % (save_path))
            plt.close()

            print(f" ============ Skip Exp {i} ============ \n")

            continue

        for defend in ["MG", "DP", "MC"]:   
            os.makedirs(f"{cur_savepath}/{defend}", exist_ok=True)
            loss_list[defend], mse_list[defend], min_list[defend], label_list[defend], index_img, label_gt =  process(defend = defend, dataset = dataset, save_path = save_path, index_img = index_img)
        
        # vẽ biểu đồ so sánh tấn công với không tấn công (3 kỹ thuật phòng thủ) - 6 cái
        for defend in ["MG", "DP", "MC"]:   
            # Draw MSE
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(5, step + 1), mse_list[defend][::10][4:], 'go-', label=defend)
            plt.plot(np.arange(5, step + 1), mse_list["nan"][::10][4:], 'r*-', label='non Defend')
            plt.xlabel('step of Iteration')
            plt.ylabel('Fidelity Threshold (MSE)')
            plt.title(dataset)
            plt.legend()
            plt.grid(True)
            plt.savefig('%s/%s/%s_NanDef_MSE.png' % (save_path, defend, defend))
            plt.close()

            # Draw Loss
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(5, step + 1), loss_list[defend][::10][4:], 'go-', label=defend)
            plt.plot(np.arange(5, step + 1), loss_list["nan"][::10][4:], 'r*-', label='non Defend')
            plt.xlabel('step of Iteration')
            plt.ylabel('Loss')
            plt.title(dataset)
            plt.legend()
            plt.grid(True)
            plt.savefig('%s/%s/%s_NanDef_Loss.png' % (save_path, defend, defend))
            plt.close()

        # Draw MSE
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(5, step + 1), mse_list["MG"][::10][4:], 'r^-', label="MG")  # Đỏ, tam giác lên, đường liền
        plt.plot(np.arange(5, step + 1), mse_list["DP"][::10][4:], 'bs--', label="DP")  # Xanh dương, hình vuông, đường đứt đoạn
        plt.plot(np.arange(5, step + 1), mse_list["MC"][::10][4:], 'kd-.', label="MC")  # Đen, hình thoi, đường chấm gạch
        plt.xlabel('step of Iteration')
        plt.ylabel('Fidelity Threshold (MSE)')
        plt.title(dataset)
        plt.legend()
        plt.grid(True)
        plt.savefig('%s/Compare_3_Defend_MSE.png' % (save_path))
        plt.close()

        # Draw Loss
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(5, step + 1), loss_list["MG"][::10][4:], 'r^-', label="MG")  # Đỏ, tam giác lên, đường liền
        plt.plot(np.arange(5, step + 1), loss_list["DP"][::10][4:], 'bs--', label="DP")  # Xanh dương, hình vuông, đường đứt đoạn
        plt.plot(np.arange(5, step + 1), loss_list["MC"][::10][4:], 'kd-.', label="MC")  # Đen, hình thoi, đường chấm gạch
        plt.xlabel('step of Iteration')
        plt.ylabel('Loss')
        plt.title(dataset)
        plt.legend()
        plt.grid(True)
        plt.savefig('%s/Compare_3_Defend_Loss.png' % (save_path))
        plt.close()

        # Vẽ bảng nhãn
        columns = ["Origin Label", "non Def", "MG", "DP", "MC"]
        row_labels = ["Thông tin nhãn"]
        table_data = [[label_gt, label_list["nan"], label_list["MG"], label_list["DP"], label_list["MC"]]]
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis('off')  
        table = ax.table(cellText=table_data, 
                        colLabels=columns, 
                        rowLabels=row_labels, 
                        loc='center', 
                        cellLoc='center',
                        rowLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        plt.savefig('%s/Label_Predict.png' % (save_path))
        plt.close()

        # Vẽ bảng min MSE
        columns = ["non Def", "MG", "DP", "MC"]
        row_labels = ["Min MSE"]
        table_data = [[round(min_list["nan"], 10), round(min_list["MG"], 10), round(min_list["DP"], 10), round(min_list["MC"], 10)]]
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis('off')  
        table = ax.table(cellText=table_data, 
                        colLabels=columns, 
                        rowLabels=row_labels, 
                        loc='center', 
                        cellLoc='center',
                        rowLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        plt.savefig('%s/Min_MSE.png' % (save_path))
        plt.close()

if __name__ == '__main__':
    for dataset in ["MNIST", "CIFAR100", "LFW"]:   
        main(dataset, 5)

    # main("MNIST", 5)

    # for dataset in ["CIFAR100", "LFW"]:   
    #     main(dataset, 3)