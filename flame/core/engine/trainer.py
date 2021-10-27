import torch

from ignite import engine as e
from abc import abstractmethod

from ...module import Module


class Engine(Module):
    '''
        Base class for all engines. Your engine should subclass this class.
        Class Engine contains an Ignite Engine that controls running process over a dataset.
        Method _update is a function receiving the running Ignite Engine and the current batch in each iteration and returns data to be stored in the Ignite Engine's state.
        Parameters:
            dataset_name (str): dataset which engine run over.
            device (str): device on which model and tensor is allocated.
            max_epochs (int): number of epochs training process runs.
    '''

    def __init__(self, dataset, device, max_epochs=1):
        super(Engine, self).__init__()
        self.dataset = dataset
        self.device = device
        self.max_epochs = max_epochs
        self.engine = e.Engine(self._update)

    def run(self):
        return self.engine.run(self.dataset, self.max_epochs)

    @abstractmethod
    def _update(self, engine, batch):
        pass


class Trainer(Engine):
    '''
        Engine controls training process.
        See Engine documentation for more details about parameters.
    '''

    def __init__(self, dataset, device, max_epochs=1):
        super(Trainer, self).__init__(dataset, device, max_epochs)

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'      # check mode, optimizer and loss functions in frame is exist or not
        assert 'optim' in self.frame, 'The frame does not have optim.'
        assert 'loss' in self.frame, 'The frame does not have loss.'
        self.model = self.frame['model'].to(self.device)
        self.optimizer = self.frame['optim']
        self.loss = self.frame['loss']

    def _update(self, engine, batch):
        self.model.train()          # convert to mode train
        self.optimizer.zero_grad()      # convert grad to 0 
        params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]        # check params is tensor => to device
        samples = torch.stack([sample.to(self.device) for sample in params[0]], dim=0)                  # stack all sample in batch to shape [batch, 3, h, w]
        targets = [{k: v.to(self.device) for k, v in target.items() if not isinstance(v, list)} for target in params[1]]    # convert target to a list of dictionaries contain 
                                                                                                                            # labels and bboxes
        preds = self.model(samples)             # put samples into model => predicts are prediction boxes and labels

        losses = self.loss(preds, targets)      # compute loss between preds and targets

        loss = losses[13] + losses[26] + losses[52]     # total loss of 3 scales 
        loss.backward()                         # compute backward
        
        self.optimizer.step()                   # implement optimizer

        return loss.item()                      # return value of loss


class Evaluator(Engine):
    '''
        Engine controls evaluating process.
        See Engine documentation for more details about parameters.
    '''

    def __init__(self, dataset, device, max_epochs=1):
        super(Evaluator, self).__init__(dataset, device, max_epochs)

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        self.model = self.frame['model'].to(self.device)        # convert model to device 

    def _update(self, engine, batch):
        self.model.eval()           # covert to mode evaluator
        with torch.no_grad():
            params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
            samples = torch.stack([image.to(self.device) for image in params[0]], dim=0)
            targets = [{k: v.to(self.device) for k, v in target.items() if not isinstance(v, list)} for target in batch[1]]

            preds = self.model(samples)

            return preds, targets
