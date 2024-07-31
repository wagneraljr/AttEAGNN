from src.entities.data_interface import DataInterface
from src.config import Config
from src.utils.data_util import DataUtil
import torch
import numpy as np
from tqdm import tqdm
from src.constants import Constants
class TrainUtil:
    @staticmethod
    def train_step(model,optimizer,loss_fn,packaged_data,target,scheduler=None):
        optimizer.zero_grad()
        predictions = model(*packaged_data)
        loss = loss_fn(predictions, target)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        return loss.item()

    @staticmethod
    def train_epoch(epochs,data:DataInterface,traffic_matrix_files,model,optimizer,loss_fn,scheduler=None):
        losses = []
        for epoch in tqdm(range(epochs)):
            for traffic_matrix_filepath in traffic_matrix_files:
                tm = Constants.path_to_day_tm_files + traffic_matrix_filepath
                node_loads = DataUtil.get_node_loads(tm)
                loss = TrainUtil.train_step(model,optimizer,loss_fn,data.get_model_args(),node_loads,scheduler)
                losses.append(loss)
        return losses
