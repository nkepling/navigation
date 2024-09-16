from  torch.utils.data import DataLoader
from nn_training import train_pnet,validate_pnet,AutoEncoderDataset
from dl_models import PNetResNet,CAE_Loss,ContractiveAutoEncoder
from utils import *
import pickle
import os


with open("obstacle.pkl","rb") as f:
    obstacle_map = pickle.load(f)

data_dir= "training_data/auto_encoder_data"

dataset_size =len([os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith(('pt'))])

train_size = int(0.7 * dataset_size)  # 70% for training
val_size = int(0.15 * dataset_size)   # 15% for validation
test_size = dataset_size - train_size - val_size  # 15% for testing

train_ind,test_ind,val_ind = torch.utils.data.random_split(range(dataset_size),[train_size,test_size,val_size])

train_data = AutoEncoderDataset(data_dir,train_ind)
val_data = AutoEncoderDataset(data_dir,val_ind)
test_data = AutoEncoderDataset(data_dir,test_ind)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data,batch_size=1,shuffle=True)


cae_model = ContractiveAutoEncoder()
model_path = "model_weights/CAE_1.pth"
# criterion = CAE_Loss(beta=1e-4)

# Adjust lr based on your needs

pnet_model = PNetResNet(coord_dim=2,latent_dim=128,hidden_dim=128,num_blocks=8)
pnet_model_path = "model_weights/pnet_resnet_2.pth"
optimizer = torch.optim.Adam(pnet_model.parameters(), lr=1e-4,weight_decay=1e-4)

criterion = torch.nn.CrossEntropyLoss()

train_pnet(train_loader,val_loader,pnet_model,cae_model,model_path,pnet_model_path,num_epochs=300,criterion=criterion,optimizer=optimizer)
    
