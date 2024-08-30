from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from RadCloud.datasets.Segmentation_Dataset import SegmentationDataset
from torch.nn import Module
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os

class Trainer:

    def __init__(self,
                 model:Module,
                 dataset_path,
                 input_directory,
                 output_directory,
                 test_split = 0.15,
                 working_dir = "working_dir",
                 save_name = "trained",
                 input_transforms:list = None,
                 output_transforms:list = None,
                 batch_size = 64,
                 epochs = 40,
                 learning_rate = 0.001,
                 loss_fn = BCEWithLogitsLoss(),
                 cuda_device = "cuda:0",
                 multiple_GPUs = True):

        #determine if device is cuda
        # determine the device to be used for training and evaluation
        if torch.cuda.is_available():

            self.device = cuda_device
            torch.cuda.set_device(self.device)
        else:
            self.device = "cpu"
        
        #setting for multiple GPUs
        self.multiple_GPUs = multiple_GPUs
        
        self.dataset_path = dataset_path
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_paths = None
        self.output_paths = None

        #paths to train and test inputs
        self.test_split = test_split
        self.train_inputs = None
        self.train_outputs = None
        self.test_inputs = None
        self.test_outputs = None

        #save the path to the working directory
        self.working_dir = working_dir
        self.save_name = save_name

        #define transforms
        if not input_transforms:
            self.input_transforms = [transforms.Resize((128,128)),transforms.ToTensor()]
        else:
            self.input_transforms = input_transforms
        
        if not output_transforms:
             self.output_transforms = [transforms.Resize((128,128)),transforms.ToTensor()]
        else:
            self.output_transforms = output_transforms

        #initialize datasets
        self.train_dataset:SegmentationDataset = None
        self.test_dataset:SegmentationDataset = None

        #initialize dataloaders
        self.train_data_loader:DataLoader = None
        self.test_data_loader:DataLoader = None
        self.pin_memory = True if self.device == "cuda" else False #determine if pinning memory during training
        
        
        #batch size, train/test steps, train/test loss history
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_steps = 0
        self.test_steps = 0
        self.history = {"train_loss":[],"test_loss":[]} #to store train/test loss history

        #initialize the model
        self.model:Module = None
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.optimizer = None
        
        #run other initialization functions
        self._init_model(model)
        self._init_input_output_paths()
        self._init_test_train_split()
        self._init_test_train_datasets()
        self._init_test_train_data_loaders()

    def _init_model(self,model:Module):

        #configure for multiple GPUs if set
        if self.multiple_GPUs and torch.cuda.is_available() and (torch.cuda.device_count() > 1):

            self.model = nn.DataParallel(model)
            print("Trainer._init_model: using {} GPUs".format(torch.cuda.device_count()))
        else: 
            self.model = model
        
        #send the model to the cuda device
        self.model.to(self.device)

        #set the optimizer
        self.optimizer = Adam(self.model.parameters(),lr=self.learning_rate)
    
    def _init_input_output_paths(self):

        input_files = sorted(os.listdir(os.path.join(self.dataset_path,self.input_directory)))
        output_files = sorted(os.listdir(os.path.join(self.dataset_path,self.output_directory)))

        self.input_paths = [os.path.join(self.dataset_path,self.input_directory,file) for file in input_files]
        self.output_paths = [os.path.join(self.dataset_path,self.output_directory,file) for file in output_files]
    
    def _init_test_train_split(self):

        #for random train and val
        # self.train_inputs,self.test_inputs,self.train_outputs,self.test_outputs = \
        #     train_test_split(self.input_paths,self.output_paths,test_size=self.test_split,random_state=2023)
        
        #for non-random train and val
        self.train_inputs,self.test_inputs,self.train_outputs,self.test_outputs = \
            train_test_split(self.input_paths,self.output_paths,test_size=self.test_split,shuffle=True)
        
        print("[INFO] saving test image paths...")
        f = open(os.path.join(self.working_dir,"test_paths.txt"),"w")
        f.write("\n".join(self.test_inputs))
        f.close()
    
    def _init_test_train_datasets(self):

        self.train_dataset = SegmentationDataset(
            input_paths= self.train_inputs,
            mask_paths= self.train_outputs,
            input_transforms= self.input_transforms,
            output_transforms= self.output_transforms
        )

        self.test_dataset = SegmentationDataset(
            input_paths= self.test_inputs,
            mask_paths= self.test_outputs,
            input_transforms = self.input_transforms,
            output_transforms=self.output_transforms
        )

        print("ModelTrainer._init_test_train_dataset: found {} samples in training dataset".format(len(self.train_dataset)))
        print("ModelTrainer._init_test_train_dataset: found {} samples in test dataset".format(len(self.test_dataset)))
    
    def _init_test_train_data_loaders(self):

        self.train_data_loader = DataLoader(
            dataset= self.train_dataset,
            batch_size= self.batch_size,
            pin_memory= self.pin_memory,
            shuffle= True,
            num_workers= os.cpu_count()
        )

        self.test_data_loader = DataLoader(
            dataset= self.test_dataset,
            batch_size= self.batch_size,
            pin_memory= self.pin_memory,
            shuffle= True,
            num_workers= os.cpu_count()
        )

        #initialize train and test steps
        self.train_steps = len(self.train_dataset) // self.batch_size
        self.test_steps = len(self.test_dataset) // self.batch_size
    
    def train_model(self):

        print("ModelTrainer.train: training the network...")
        start_time = time.time()

        for epoch in (tqdm(range(self.epochs))):
            
            #put model into training mode
            self.model.train()

            #initialize total training and validation loss
            total_train_loss = 0
            total_test_loss = 0
            
            for (x,y) in self.train_data_loader:

                #send the input to the device
                (x,y) = (x.to(self.device),y.to(self.device))

                #perform forward pass and calculate training loss
                pred = self.model(x)
                loss = self.loss_fn(pred,y)
                #zero out any previously accumulated gradients, perform back propagation, update model parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #add the loss to the total loss
                total_train_loss += loss

            #switch off autograd
            with torch.no_grad():

                #set model in evaluation mode
                self.model.eval()

                #loop over validation set
                for (x,y) in self.test_data_loader:

                    (x,y) = (x.to(self.device),(y.to(self.device)))

                    #perform forward pass and calculate training loss
                    pred = self.model(x)
                    loss = self.loss_fn(pred,y)

                    total_test_loss += loss
            
            avg_train_loss = total_train_loss / self.train_steps
            avg_test_loss = total_test_loss / self.test_steps

            #update training history
            self.history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
            self.history["test_loss"].append(avg_test_loss.cpu().detach().numpy())

            print("EPOCH: {}/{}".format(epoch + 1, self.epochs))
            print("\t Train loss: {}, Test loss:{}".format(avg_train_loss,avg_test_loss))

        end_time = time.time()
        print("ModelTrainer.train: total training time {:.2f}".format(end_time - start_time))
        
        #plot the results
        self.plot_results()

        #save the model
        file_name = "{}.pth".format(self.save_name)

        #save the state dict
        if self.multiple_GPUs and torch.cuda.is_available() and (torch.cuda.device_count() > 1):
            torch.save(self.model.module.state_dict(),os.path.join(self.working_dir,file_name))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.working_dir,file_name))


    def plot_results(self):

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(self.history["train_loss"], label="train_loss")
        plt.plot(self.history["test_loss"], label="test_loss")
        plt.title("Training Loss on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        file_name = "{}.png".format(self.save_name)
        plt.savefig(os.path.join(self.working_dir,file_name))
