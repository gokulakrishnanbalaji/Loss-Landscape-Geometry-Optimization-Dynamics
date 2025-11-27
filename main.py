
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Models

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(x)))


class SmallCNN(nn.Module):
    def __init__(self, use_residual=False):
        super().__init__()
        self.use_residual = use_residual

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)

        if use_residual:
            self.skip = nn.Conv2d(1, 16, 1)  # 1x1 skip
        else:
            self.skip = None

        self.fc = nn.Linear(16*28*28, 10)

    def forward(self, x):
        if self.use_residual:
            out = F.relu(self.conv1(x) + self.skip(x))
        else:
            out = F.relu(self.conv1(x))

        out = F.relu(self.conv2(out))
        out = out.view(out.size(0), -1)
        return self.fc(out)

# Data

def get_mnist_loaders(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(".", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(".", train=False, download=True, transform=transform)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset,  batch_size=batch_size, shuffle=False),
    )

# Training & Evaluation

def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()*x.size(0)
            preds = logits.argmax(1)
            total_correct += (preds==y).sum().item()
            total_samples += x.size(0)

    return total_loss/total_samples, total_correct/total_samples


def train_model(model, train_loader, test_loader, epochs=3, name="Model", device="cpu"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        total_loss=0; total_correct=0; total_samples=0

        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits,y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*x.size(0)
            preds = logits.argmax(1)
            total_correct += (preds==y).sum().item()
            total_samples += x.size(0)

        train_loss = total_loss/total_samples
        train_acc  = total_correct/total_samples

        test_loss, test_acc = evaluate(model, test_loader, device)

        print(f"[{name}] Epoch {ep+1}/{epochs} "
              f"| Train Loss {train_loss:.4f}, Acc {train_acc:.3f} "
              f"| Test Loss {test_loss:.4f}, Acc {test_acc:.3f}")

    return model

# Param utilities

def flatten_params(model):
    return torch.cat([p.detach().view(-1) for p in model.parameters()])

def set_params(model, flat_vec):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_( flat_vec[idx:idx+n].view_as(p) )
        idx += n

# Hessian / Eigenvalue

def hessian_vector_product(model, loss, v):
    grads = torch.autograd.grad(
        loss, model.parameters(),
        create_graph=True,
        retain_graph=True
    )
    grad_flat = torch.cat([g.reshape(-1) for g in grads])

    Hv = torch.autograd.grad(
        grad_flat, model.parameters(),
        grad_outputs=v,
        retain_graph=True
    )
    Hv_flat = torch.cat([h.reshape(-1) for h in Hv])
    return Hv_flat


def approx_top_eigenvalue(model, loader, device="cpu", iters=20):
    model.eval()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    loss = F.cross_entropy(model(x), y)
    n = sum(p.numel() for p in model.parameters())

    v = torch.randn(n, device=device)
    v = v / v.norm()

    for _ in range(iters):
        Hv = hessian_vector_product(model, loss, v)
        v = Hv / (Hv.norm()+1e-8)

    Hv = hessian_vector_product(model, loss, v)
    return torch.dot(v, Hv).item()

# Flatness metric

def flatness_metric(model, loader, device, epsilon=1e-3, num_samples=10):
    x,y = next(iter(loader))
    x,y = x.to(device), y.to(device)
    base_params = flatten_params(model).to(device)

    base_loss = F.cross_entropy(model(x), y).item()

    deltas=[]
    for _ in range(num_samples):
        noise = torch.randn_like(base_params)
        noise = epsilon*noise / noise.norm()

        set_params(model, base_params + noise)
        pert_loss = F.cross_entropy(model(x),y).item()
        deltas.append(pert_loss - base_loss)

    set_params(model, base_params)
    return float(np.mean(deltas))

# 1D Loss Slice

def loss_slice(model, loader, device="cpu", radius=0.05, points=21):
    x,y = next(iter(loader))
    x,y = x.to(device), y.to(device)

    base = flatten_params(model).to(device)
    d = torch.randn_like(base)
    d = d / d.norm()

    alphas = np.linspace(-radius, radius, points)
    losses=[]

    for a in alphas:
        set_params(model, base + a*d)
        with torch.no_grad():
            losses.append(F.cross_entropy(model(x),y).item())

    set_params(model, base)
    return alphas, losses

# Mode Connectivity

def mode_connectivity(modelA, modelB, loader, device="cpu", points=21):
    x,y = next(iter(loader))
    x,y = x.to(device), y.to(device)

    pA = flatten_params(modelA).to(device)
    pB = flatten_params(modelB).to(device)

    base = type(modelA)()  # exact SAME architecture
    base.to(device)

    alphas = np.linspace(0,1,points)
    losses=[]

    for a in alphas:
        params = (1-a)*pA + a*pB
        set_params(base, params)
        with torch.no_grad():
            losses.append(F.cross_entropy(base(x),y).item())

    return alphas, losses

# Main

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_loader, test_loader = get_mnist_loaders()

    # ARCHITECTURES
    archs = {
        "MLP": MLP(),
        "CNN_no_residual": SmallCNN(use_residual=False),
        "CNN_residual": SmallCNN(use_residual=True),
    }

    trained = {}
    trained2 = {}

    print("\n========== TRAINING MODELS ==========")
    for name, model in archs.items():
        print(f"\n--- Training {name} ---")
        trained[name] = train_model(model, train_loader, test_loader, 3, name, device)

    print("\n========== TRAINING SECOND COPIES FOR MODE CONNECTIVITY ==========")
    for name, model in archs.items():
        print(f"\n--- Training {name} (run2) ---")
        if name == "MLP":
            modelB = MLP()
        elif name == "CNN_no_residual":
            modelB = SmallCNN(use_residual=False)
        else:
            modelB = SmallCNN(use_residual=True)
        trained2[name] = train_model(modelB, train_loader, test_loader, 3, name+"_run2", device)

    # 1. Eigenvalue
    print("\n========== COMPUTING MAX EIGENVALUE ==========")
    lambdas = {}
    for name, model in trained.items():
        lam = approx_top_eigenvalue(model, train_loader, device)
        lambdas[name] = lam
        print(f"{name}: lambda_max = {lam:.4f}")

    plt.figure()
    plt.bar(lambdas.keys(), lambdas.values())
    plt.ylabel("lambda_max")
    plt.title("Curvature per Model")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("eigenvalues.png")
    plt.close()

    # 2. Flatness metric
    print("\n========== COMPUTING FLATNESS ==========")
    flats={}
    for name, model in trained.items():
        flats[name] = abs(flatness_metric(model, train_loader, device))
        print(f"{name}: flatness |ΔL| = {flats[name]:.6f}")

    plt.figure()
    plt.bar(flats.keys(), flats.values())
    plt.ylabel("|ΔLoss|")
    plt.title("Flatness Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("flatness.png")
    plt.close()

    # 3. 1D loss slice
    print("\n========== 1D LOSS SLICE ==========")
    for name, model in trained.items():
        a, L = loss_slice(model, train_loader, device)
        plt.figure()
        plt.plot(a, L, marker='o')
        plt.title(f"1D Loss Slice: {name}")
        plt.xlabel("alpha")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig(f"loss_slice_{name}.png")
        plt.close()

    # 4. Mode connectivity
    print("\n========== MODE CONNECTIVITY ==========")
    for name in archs.keys():
        a, L = mode_connectivity(trained[name], trained2[name], train_loader, device)
        plt.figure()
        plt.plot(a, L, marker='o')
        plt.title(f"Mode Connectivity: {name}")
        plt.xlabel("alpha")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig(f"mode_connectivity_{name}.png")
        plt.close()

    print("\n==== DONE! All plots saved ====")
    print("Files generated:")
    print("- eigenvalues.png")
    print("- flatness.png")
    print("- loss_slice_<model>.png")
    print("- mode_connectivity_<model>.png")

if __name__ == "__main__":
    main()
