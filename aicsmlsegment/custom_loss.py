
from torch.autograd import Variable, Function
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

class ElementNLLLoss(torch.nn.Module):
	def __init__(self, num_class):
		super(ElementNLLLoss,self).__init__()
		self.num_class = num_class
	
	def forward(self, input, target, weight):

		target_np = target.cpu().data.numpy()
		target_np = target_np.astype(int)

		row_num = target_np.shape[0]
		mask = np.zeros((row_num,self.num_class )) 
		mask[np.arange(row_num), target_np]=1
		class_x = torch.masked_select(input, Variable(torch.from_numpy(mask).cuda().byte()))

		out = torch.mul(class_x,weight)
		loss = torch.mean(torch.neg(out),0)

		return loss

class MultiAuxillaryElementNLLLoss(torch.nn.Module):
	def __init__(self,num_task, weight, num_class):
		super(MultiAuxillaryElementNLLLoss,self).__init__()
		self.num_task = num_task
		self.weight = weight

		self.criteria_list = [] 
		for nn in range(self.num_task):
			self.criteria_list.append(ElementNLLLoss(num_class[nn]))
	
	def forward(self, input, target, cmap):

		total_loss = self.weight[0]*self.criteria_list[0](input[0], target.view(target.numel()), cmap.view(cmap.numel()) )

		for nn in np.arange(1,self.num_task):
			total_loss = total_loss + self.weight[nn]*self.criteria_list[nn](input[nn], target.view(target.numel()), cmap.view(cmap.numel()) )

		return total_loss

class MultiTaskElementNLLLoss(torch.nn.Module):
	def __init__(self, weight, num_class):
		super(MultiTaskElementNLLLoss,self).__init__()
		self.num_task = len(num_class)
		self.weight = weight

		self.criteria_list = [] 
		for nn in range(self.num_task):
			self.criteria_list.append(ElementNLLLoss(num_class[nn]))
	
	def forward(self, input, target, cmap):

		assert len(target) == self.num_task and len(input) == self.num_task

		total_loss = self.weight[0]*self.criteria_list[0](input[0], target[0].view(target[0].numel()), cmap.view(cmap.numel()) )

		for nn in np.arange(1,self.num_task):
			total_loss = total_loss + self.weight[nn]*self.criteria_list[nn](input[nn], target[nn].view(target[nn].numel()), cmap.view(cmap.numel()) )

		return total_loss

class ElementAngularMSELoss(torch.nn.Module):
	def __init__(self):
		super(ElementAngularMSELoss,self).__init__()
	
	def forward(self, input, target, weight):

		#((input - target) ** 2).sum() / input.data.nelement()
	
		return torch.sum( torch.mul( torch.acos(torch.sum(torch.mul(input,target),dim=1))**2, weight) )/ torch.gt(weight,0).data.nelement()
		