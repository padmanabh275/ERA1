{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "47931abb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "from torchvision import datasets, transforms"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1b0ab8ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "# CUDA?\n",
    "cuda = torch.cuda.is_available()\n",
    "print(\"CUDA Available?\", cuda)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e5ca605d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train data transformations\n",
    "train_transforms = transforms.Compose([\n",
    "    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),\n",
    "    transforms.Resize((28, 28)),\n",
    "    transforms.RandomRotation((-15., 15.), fill=0),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize((0.1307,), (0.3081,)),\n",
    "    ])\n",
    "\n",
    "# Test data transformations\n",
    "test_transforms = transforms.Compose([\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize((0.1307,), (0.3081,))\n",
    "    ])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f52aebe1",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)\n",
    "test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0c230b16",
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size = 512\n",
    "\n",
    "kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}\n",
    "\n",
    "test_loader = torch.utils.data.DataLoader(test_data, **kwargs)\n",
    "train_loader = torch.utils.data.DataLoader(train_data, **kwargs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "26c1a24c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data to plot accuracy and loss graphs\n",
    "train_losses = []\n",
    "test_losses = []\n",
    "train_acc = []\n",
    "test_acc = []\n",
    "\n",
    "test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6fee3062",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "from tqdm import tqdm\n",
    "\n",
    "def GetCorrectPredCount(pPrediction, pLabels):\n",
    "  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()\n",
    "\n",
    "def train(model, device, train_loader, optimizer, criterion):\n",
    "  model.train()\n",
    "  pbar = tqdm(train_loader)\n",
    "\n",
    "  train_loss = 0\n",
    "  correct = 0\n",
    "  processed = 0\n",
    "\n",
    "  for batch_idx, (data, target) in enumerate(pbar):\n",
    "    data, target = data.to(device), target.to(device)\n",
    "    optimizer.zero_grad()\n",
    "\n",
    "    # Predict\n",
    "    pred = model(data)\n",
    "\n",
    "    # Calculate loss\n",
    "    loss = criterion(pred, target)\n",
    "    train_loss+=loss.item()\n",
    "\n",
    "    # Backpropagation\n",
    "    loss.backward()\n",
    "    optimizer.step()\n",
    "    \n",
    "    correct += GetCorrectPredCount(pred, target)\n",
    "    processed += len(data)\n",
    "\n",
    "    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')\n",
    "\n",
    "  train_acc.append(100*correct/processed)\n",
    "  train_losses.append(train_loss/len(train_loader))\n",
    "\n",
    "def test(model, device, test_loader, criterion):\n",
    "    model.eval()\n",
    "\n",
    "    test_loss = 0\n",
    "    correct = 0\n",
    "\n",
    "    with torch.no_grad():\n",
    "        for batch_idx, (data, target) in enumerate(test_loader):\n",
    "            data, target = data.to(device), target.to(device)\n",
    "\n",
    "            output = model(data)\n",
    "            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss\n",
    "\n",
    "            correct += GetCorrectPredCount(output, target)\n",
    "\n",
    "\n",
    "    test_loss /= len(test_loader.dataset)\n",
    "    test_acc.append(100. * correct / len(test_loader.dataset))\n",
    "    test_losses.append(test_loss)\n",
    "\n",
    "    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\\n'.format(\n",
    "        test_loss, correct, len(test_loader.dataset),\n",
    "        100. * correct / len(test_loader.dataset)))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb62e426",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Net().to(device)\n",
    "optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)\n",
    "scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)\n",
    "# New Line\n",
    "criterion = F.nll_loss\n",
    "num_epochs = 20\n",
    "\n",
    "for epoch in range(1, num_epochs+1):\n",
    "  print(f'Epoch {epoch}')\n",
    "  train(model, device, train_loader, optimizer, criterion)\n",
    "  test(model, device, test_loader, criterion)\n",
    "  scheduler.step()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
