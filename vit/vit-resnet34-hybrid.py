# Inspired By:
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
# https://github.com/ra1ph2/Vision-Transformer/tree/main

import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torchvision.models import resnet34

class Attention(nn.Module):

    def __init__(self, emb_dim, heads, dropout=0.1):
        super(Attention, self).__init__()
        self.heads = heads
        self.emb_dim = emb_dim

        # define q, k, v
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim) 
        self.softmax = nn.Softmax(dim=-1) # perform softmax on the last dimension
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input shape: (batch_size, seq_len, emb_dim)
        batch_size, seq_len, emb_dim = x.size()

        # obtain q, k, v
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        # attention = softmax(q*k^T)
        k_T = k.transpose(1, 2)
        attn = self.softmax(q @ k_T)
        attn = self.dropout(attn) 
        
        # mat mul with v
        out = attn @ v

        # output of _reshape_heads_back(): (batch_size, seq_len, embed_size)
        # change the dimensions back to original dimensions
        out = self._reshape_heads_back(out)

        return out, attn

    def _reshape_heads(self, x):
        # reshapes input for multi-head attention.
        # emb_dim is divided by the number of heads
        # input shape: (batch_size, seq_len, emb_dim)
        
        # precompute sizes 
        batch_size, seq_len, emb_dim = x.size()
        reduced_dim = self.emb_dim // self.heads

        out = x.reshape(batch_size, seq_len, self.heads, reduced_dim)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(-1, seq_len, reduced_dim)

        # output shape: (batch_size * heads, seq_len, reduced_dim)
        return out

    def _reshape_heads_back(self, x):
        # returns input back to original shape of emb_dim
        # input shape: (batch_size * heads, seq_len, reduced_dim)
        
        # precompute sizes 
        batch_size_mul_heads, seq_len, reduced_dim = x.size()
        batch_size = batch_size_mul_heads // self.heads

        out = x.reshape(batch_size, self.heads, seq_len, reduced_dim)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(batch_size, seq_len, self.emb_dim)

        # output shape: (batch_size, seq_len, emb_dim)
        return out
    
class FeedForward(nn.Module):

    def __init__(self, emb_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.emb_dim = emb_dim
        self.fc1 = nn.Linear(emb_dim, emb_dim) # linear layer 
        self.gelu = nn.GELU() # GELU layer
        self.fc2 = nn.Linear(emb_dim, emb_dim) # linear layer 
        self.dropout = nn.Dropout(dropout) # dropout

    def forward(self, x):
        # input shape: (batch_size, seq_len, emb_dim)
        batch_size, seq_len, emb_dim = x.size()
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        # output shape: (batch_size, seq_len, emb_dim)
        return out
    

class ResNetLayers(nn.Module):
    def __init__(self):
        super(ResNetLayers, self).__init__()
        layers = list(resnet34(weights='DEFAULT').children())[:5]
        self.feature_extractor = nn.Sequential(*layers)
        
    def forward(self, inp):
        out = self.feature_extractor(inp)
        return out
    
class Encoder(nn.Module):

    def __init__(self, emb_dim, heads, dropout=0.1):
        # dropout default set to 0.1 
        super(Encoder, self).__init__()
        self.emb_dim = emb_dim
        self.ln1 = nn.LayerNorm(emb_dim) # default activation: layer norm
        self.attention = Attention(emb_dim, heads, dropout) # MHA 
        self.ln2 = nn.LayerNorm(emb_dim) # default activation: layer norm
        self.feed_forward = FeedForward(emb_dim, dropout)

    def forward(self, x):
        # input size: (batch_size, seq_len, emb_dim)
        # Note: addition operation serves as residual connection to facilitate the flow
        # of info and gradients. Adding original input and the result of inner layers
        # creates a shortcut connection

        batch_size, seq_len, emb_dim = x.size()
        residual = x # residual connection
        
        # pass through layer norm before attention layer 
        out = self.ln1(x) 
        out, _ = self.attention(out) # multiheaded attention
        out = out + residual
        
        # residual connection
        residual = out # residual connection
        out = self.ln2(out) # layer norm

        # call MLP/Feed Forward
        out = self.feed_forward(out)
        out = out + residual

        # output size: (batch_size, seq_len, emb_dim)
        return out
    
class Transformer(nn.Module):

    def __init__(self, emb_dim, num_layers, heads, dropout=0.1):
        super(Transformer, self).__init__()
        # adjust the number of layers accordingly 
        self.trans_blocks = nn.ModuleList(
            [Encoder(emb_dim, heads, dropout) for i in range(num_layers)]
        )

    def forward(self, x):
        # input size: (batch_size, seq_len, emb_dim)
        out = x
        for block in self.trans_blocks:
            out = block(out)
        # output size: (batch_size, seq_len, emb_dim)
        return out
    
class MLPHead(nn.Module):

    def __init__(self, emb_dim, num_classes, dropout=0.1):
        super(MLPHead, self).__init__()
        
        self.emb_dim = emb_dim
        self.num_classes = num_classes # number of classes: 100 
        self.fc1 = nn.Linear(emb_dim, emb_dim // 2) # linear layer 
        self.gelu = nn.GELU() # GELU activation
        self.fc2 = nn.Linear(emb_dim // 2, num_classes) # linear layer 
        self.dropout = nn.Dropout(dropout) # default: dropout = 0.1 

    def forward(self, x):
        # input size: (batch_size, emb_dim)
        batch_size, emb_dim = x.size()
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # output size: (batch_size, num_classes)
        return x
    
# Main ViT class 
class ViT(nn.Module):

    def __init__(self, patch_size, max_len, emb_dim, num_classes, num_layers, num_channels, heads, dropout=0.1):

        super(ViT, self).__init__()

        self.emb_dim = emb_dim
        self.num_channels = num_channels
        self.patch_size = patch_size

        # linear projection
        self.patch_to_embed = nn.Linear(patch_size * patch_size * num_channels, emb_dim)

        # creating a learnable parameter for positional embeddings
        # sampled from a normal distribution with mean 0 and standard deviation 1
        self.pos_emb = nn.Parameter(torch.randn((max_len, emb_dim)))

        # call transformer encoder layer (default 12 layers)
        self.transformer = Transformer(emb_dim, num_layers, heads, dropout)
        # self.MLP_head = MLPHead(emb_dim, num_classes)
        self.class_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.gelu = nn.GELU() # GELU layer
        self.linear = nn.Linear(emb_dim, num_classes)

    def forward(self, img):

        # image size: (256, 3, 32, 32)
        batch_size, num_channels, img_width, img_height = img.size()
        num_patches = ((img_height // self.patch_size) * (img_width // self.patch_size))

        # patch_dim = num_channels * self.patch_size * self.patch_size

        # unfold patches along the width (dim = 2) and height (dim = 3) dimensions
        out = img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).contiguous() # output shape: [256, 3, 8, 8, 4, 4]

        # Reshape the unfolded image to create a sequence of patches
        out = out.view(batch_size, num_channels, -1, self.patch_size, self.patch_size) # output shape: [256, 3, 64, 4, 4]
        out = out.permute(0, 2, 3, 4, 1) # output shape: [256, 64, 4, 4, 3]
        out = out.reshape(batch_size, num_patches, -1) # output shape: [256, 64, 48]

        # output shape: [256, 64, 512]
        # linear transformation: map to larger dimension (512)
        out = self.patch_to_embed(out)

        # output shape: [256, 65, 512]
        # +1 to the num patches
        # concatenate the cls token embedding patch to the first column
        class_token = self.class_token.expand(batch_size, -1, -1)
        out = torch.cat([class_token, out], dim=1)

        # positional embedding, output shape: [65, 512]
        pos_emb = self.pos_emb[:num_patches+1]

        # output shape: [256, 65, 512]
        # expand position size to match batch size
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, num_patches+1, self.emb_dim)

        # output shape: [256, 65, 512]
        out = out + pos_emb # add the learnable positional embedding

        ##### beginning of Transformer #####

        # pass through transformer
        # out: (batch_size, num_patches+1, emb_dim)
        # output shape: [256, 65, 512]
        out = self.transformer(out)

        # class_token: (batch_size, emb_dim)
        # output shape: [256, 512]
        class_token = out[:, 0]

        ########## Classification ##########
        # predictions = self.MLP_head(class_token)
        
        predictions = self.gelu(self.linear(class_token))
        return predictions, out


def CIFARDataLoader(train, batch_size, num_workers, shuffle=True, size='32'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train == True:

        if size == '224':
            train_transform = transforms.Compose([
                transforms.RandomApply([transforms.RandomRotation(30)], p=0.5), 
                transforms.RandomResizedCrop((224,224), scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),   
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        dataloader = DataLoader(cifar100, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    elif train == False:

        if size == '224':
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        cifar100 = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        dataloader = DataLoader(cifar100, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return dataloader

def train(model, dataloader, criterion, optimizer, scheduler, resnet_features):

    train_running_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)        
        with torch.no_grad():
            if resnet_features != None:
                inputs = resnet_features(inputs)           
        # pass through model, compute CE loss 
        predictions, _ = model(inputs)
        
        loss = criterion(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        
        # perform gradient clipping  with max norm of 1 
        clip_grad_norm_(model.parameters(), max_norm = 1.0)
        
        # step after weight update 
        optimizer.step()
        scheduler.step()

        train_correct += (predictions.argmax(dim=1) == labels).float().sum().detach().cpu().item()
        train_running_loss += loss.item()
        train_total += len(labels)
        
    avg_train_acc = train_correct / train_total
    avg_train_loss = train_running_loss/len(dataloader)
    
    return avg_train_loss, avg_train_acc

def test(model, dataloader, criterion, resnet_features):

    # testing 
    model.eval()
    with torch.no_grad():
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # pass through model 
            if resnet_features != None:
                inputs = resnet_features(inputs)

            predictions, _ = model(inputs)
            loss = criterion(predictions, labels)

            # compute accacy and loss 
            test_running_loss += loss.item()
            test_correct += (predictions.argmax(dim=1) == labels).float().sum().detach().cpu().item()
            test_total += len(labels)
            
    avg_test_acc = test_correct / test_total
    avg_test_loss = test_running_loss/len(dataloader)
    
    return avg_test_loss, avg_test_acc

if __name__ == '__main__':
    
    # GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # hyperparameters used during training 
    lr = 0.0001
    batch_size = 256
    num_workers = 2
    patch_size = 7 # default: 16
    image_sz = 32
    max_len = 100 
    emb_dim = 512 # default: 768
    num_classes = 100
    num_layers = 8 # default: 12
    num_channels = 3
    heads = 16 # default: 12
    epochs = 300
    resnet_features_channels = 64
    

    model = ViT(patch_size, max_len, emb_dim, num_classes, num_layers, resnet_features_channels, heads).to(device)
    
    resnet_features = ResNetLayers().to(device).eval()
    
    
    # training and testing dataloader
    train_dataloader = CIFARDataLoader(train= True, batch_size=batch_size, num_workers=num_workers, shuffle=True, size='224')
    test_dataloader = CIFARDataLoader(train= False, batch_size=batch_size, num_workers=num_workers, shuffle=False, size='224')
    
    # define loss function, optimizer, scheduler 
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=epochs)
    
    # lists for plots  
    train_accs_list = []
    test_accs_list = []
    train_loss_list = []
    test_loss_list = []
    
    for epoch in range(epochs):
        print(optimizer.param_groups[0]['lr'])
        # TRAINING
        running_loss, running_accuracy = train(model, train_dataloader, criterion, optimizer, scheduler, resnet_features)
        train_accs_list.append(running_accuracy)
        train_loss_list.append(running_loss)
        
        # TESTING
        test_loss, test_accuracy = test(model, test_dataloader, criterion, resnet_features)
        print(f"[Epoch: {epoch+1}] train acc: {running_accuracy:.4f} || train loss: {running_loss:.4f} || test acc: {test_accuracy:.4f} || test loss: {test_loss:.4f}")
        test_accs_list.append(test_accuracy)
        test_loss_list.append(test_loss)
    
        if (epoch+1)%5 == 0:
            torch.save({'model': model}, './' + "VisionTransformer" + '_CIFAR100_checkpoint.pt')
            
    with open(r'./train_accs_list.txt', 'w') as fp:
        for item in train_accs_list:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')

    with open(r'./test_accs_list.txt', 'w') as fp:
        for item in test_accs_list:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')
    
    with open(r'./train_loss_list.txt', 'w') as fp:
        for item in train_loss_list:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')
    
    with open(r'./test_loss_list.txt', 'w') as fp:
        for item in test_loss_list:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')
