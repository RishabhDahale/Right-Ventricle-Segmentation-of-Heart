import torch

def L1_regularization(model, lamda):
	params = torch.cat([param.view(-1) for param in model.parameters()])
	l1_regularization = lamda * torch.norm(params, 1)
	return l1_regularization

def L2_regularization(model, lamda):
	params = torch.cat([param.view(-1) for param in model.parameters()])
	l2_regularization = lamda * torch.norm(params, 2)
	return l2_regularization

def L1L2_regularization(model, lamda1, lamda2):
	params = torch.cat([param.view(-1) for param in model.parameters()])
	l1_regularization = lamda1 * torch.norm(params, 1)
	l2_regularization = lamda2 * torch.norm(params, 2)
	return l1_regularization + l2_regularization