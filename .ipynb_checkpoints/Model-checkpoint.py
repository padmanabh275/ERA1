{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b4a78acc",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Net(nn.Module):\n",
    "    #This defines the structure of the NN.\n",
    "    def __init__(self):\n",
    "        super(Net, self).__init__()\n",
    "        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)\n",
    "        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)\n",
    "        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)\n",
    "        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)\n",
    "        self.fc1 = nn.Linear(4096, 50)\n",
    "        self.fc2 = nn.Linear(50, 10)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = F.relu(self.conv1(x), 2) # 28>26 | 1>3 | 1>1\n",
    "        x = F.relu(F.max_pool2d(self.conv2(x), 2)) #26>24>12 | 3>5>6 | 1>1>2\n",
    "        x = F.relu(self.conv3(x), 2) # 12>10 | 6>10 | 2>2\n",
    "        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # 10>8>4 | 10>14>16 | 2>2>4\n",
    "        x = x.view(-1, 4096) # 4*4*256 = 4096\n",
    "        x = F.relu(self.fc1(x))\n",
    "        x = self.fc2(x)\n",
    "        return F.log_softmax(x, dim=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "50286de8",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Net2(nn.Module):\n",
    "    #This defines the structure of the NN.\n",
    "    def __init__(self):\n",
    "        super(Net2, self).__init__()\n",
    "        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=False)\n",
    "        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=False)\n",
    "        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, bias=False)\n",
    "        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, bias=False)\n",
    "        self.fc1 = nn.Linear(4096, 50, bias=False)\n",
    "        self.fc2 = nn.Linear(50, 10, bias=False)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = F.relu(self.conv1(x), 2) # 28>26 | 1>3 | 1>1\n",
    "        x = F.relu(F.max_pool2d(self.conv2(x), 2)) #26>24>12 | 3>5>6 | 1>1>2\n",
    "        x = F.relu(self.conv3(x), 2) # 12>10 | 6>10 | 2>2\n",
    "        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # 10>8>4 | 10>14>16 | 2>2>4\n",
    "        x = x.view(-1, 4096) # 4*4*256 = 4096\n",
    "        x = F.relu(self.fc1(x))\n",
    "        x = self.fc2(x)\n",
    "        return F.log_softmax(x, dim=1)\n",
    "\n",
    "!pip install torchsummary\n",
    "from torchsummary import summary\n",
    "use_cuda = torch.cuda.is_available()\n",
    "device = torch.device(\"cuda\" if use_cuda else \"cpu\")\n",
    "model = Net2().to(device)\n",
    "summary(model, input_size=(1, 28, 28))"
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
