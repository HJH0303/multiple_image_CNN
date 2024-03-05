import torch.optim as optim
import numpy as np
import sys
sys.path.append(r'/home/aims/2024/image_CNN/data')
from data_load import Data_Pre
from torch import nn
import pickle
import torch
import random
from torch.utils.data import DataLoader, Dataset,random_split
from matplotlib import pyplot as plt

#gpu설정
USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else 'cpu')
device='cuda'

print("다음 기기로 학습합니다:", device)
random.seed(777)
torch.manual_seed(777)
if device=='cuda':
    torch.cuda.manual_seed_all(777)
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()  
        self.Leakyrelu = nn.LeakyReLU()  # 쓰면 안됨
        
      
        self.fc_1 = nn.Linear(hidden_size,30, bias=True)  #
        self.bn1 = torch.nn.BatchNorm1d(30)
        self.fc_2 = nn.Linear(30, 30, bias=True)
        self.bn2 = torch.nn.BatchNorm1d(30)
        self.fc = nn.Linear(30, 1, bias=True)  # You can adjust the output layer as needed
       

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        out, (hn, cn) = self.lstm(x, (h0, c0))  
        
        hn = hn.view(-1, self.hidden_size)
     
        out = self.fc_1(out)  # Apply LeakyReLU activation
        out = self.bn1(out)
        out = self.relu(out)  # Apply LeakyReLU activation
        out = self.fc_2(out)  # Apply LeakyReLU activation
        out = self.bn2(out)
        out = self.relu(out)
        
        # Index hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out
    
    
    def predict(self, X):
        return self.model.predict(X)


class CustomDataset(Dataset): 
  def __init__(self,x_data,y_data):
    self.x = x_data
    self.y= y_data

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self,idx):
      
    # Specify the columns to delete (3rd and 6th columns, 0-based index)
    # columns_to_delete = [2, 5]   
    # arr = np.delete(self.x[idx].reshape(30,6),  columns_to_delete, axis=1)
    arr=self.x[idx].reshape(30,6)
    
    x=torch.from_numpy(arr).to(device).float()
    y=torch.from_numpy(self.y[idx]).to(device).float()
    y=y.squeeze()
    return x, y


def saveModel(net): 
    path = "/home/aims/2024/weights/final.pth" 
    torch.save(net.state_dict(), path)  

# Example usage with progress printout every 10 epochs:
if __name__ == "__main__":
    # 데이터 로드
    input_arr, input_label=Data_Pre.data_load()
    print(input_arr.shape)
    print(input_label.shape)
    dataset =CustomDataset(input_arr,input_label)

    train_size=4379
    valid_size=1094
    print("total size:",train_size+valid_size)
    print("trainsets size:",train_size)
    print("validsets size:",valid_size)

    train_set, valid_set=random_split(dataset,[train_size,valid_size])

    train_loader= DataLoader(train_set, batch_size=4, shuffle=False)
    val_loader= DataLoader(valid_set, batch_size=4, shuffle=False)

    input_size = 6  # Number of features
    hidden_size = 2  # Number of LSTM units
    num_layers = 2 # Number of LSTM layers
    batch_size = 4  # Set the batch size

    # Initialize the CustomLSTM model
    lstm_model = CustomLSTM(input_size, hidden_size, num_layers).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.0005)
    training_epochs =300 # Total number of epochs
    trainingEpoch_loss = []
    check_path1= "/home/aims/2024/weights" 

    # Training Mode
    lstm_model.train()
    for epoch in range(training_epochs):
        step_loss = []
        train_loss = 0.0

        for X, Y in train_loader: 
            X = X.to(device)
            Y = Y.to(device)   
            optimizer.zero_grad()
            output1 = lstm_model(X)
            new_shape = len(X)
            y_tensor_1 = Y[:,:1].view(new_shape)
            loss1 = criterion(output1[:,0], y_tensor_1).to(device)
            loss1.backward()
            optimizer.step()
            
            train_loss+=loss1
            step_loss.append(loss1.item())
        check_path2 = f"/{epoch + 1}.pth"
        torch.save(lstm_model.state_dict(), check_path1 + check_path2)
        print (f'Epoch [{epoch+1}/{training_epochs}], Loss: {train_loss:.4f}')
        print("save : " ,check_path2)

        trainingEpoch_loss.append(np.array(step_loss).mean())
    saveModel(lstm_model)




#inference


# # Load the saved model from a .pt file (replace 'your_model.pt' with the actual file path)
# model = CustomLSTM(input_size=6, hidden_size=30, num_layers = 10 ).to(device)  # Create an instance of the model
# model.load_state_dict(torch.load("/home/aims/obb_contents/weights/only_x/lstm_x_50.pth" ))  # Load the model weights
# inference_data = torch.tensor(np.random.rand(1,30, 6), dtype=torch.float32).to(device)

# output = model(inference_data)
plt.plot(trainingEpoch_loss, color='r', label='train')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss'], loc='upper right')