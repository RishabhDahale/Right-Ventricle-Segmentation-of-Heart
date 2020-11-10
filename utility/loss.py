import torch

def SoftDiceLoss(yTrue, yPred, epsilon=1, axis=None):
	"""
	Mathematical equation is dice loss = 2*intersection/(intersection + union)
	:param yTrue: True binary mask
	:param yPred: Predicted mask
	:param epsilon: error factor for dice loss
	:return:
	"""
	numerator = torch.sum(yPred*yTrue, axis=axis) + epsilon/2
	denominator = torch.sum(yTrue, axis=axis) + torch.sum(yPred, axis=axis) + epsilon
	loss = 2*numerator/denominator
	return loss

class DiceLoss(torch.nn.Module):
	def __init__(self, epsilon=1):
		super(DiceLoss, self).__init__()
		self.epsilon=epsilon

	def forward(self, yTrue, yPred):
		batchDice = SoftDiceLoss(yTrue, yPred, self.epsilon, axis=[2, 3])
		loss = 1-batchDice
		return loss

class InverseDiceLoss(torch.nn.Module):
	def __init__(self, epsilon=1):
		super(InverseDiceLoss, self).__init__()
		self.epsilon = epsilon

	def forward(self, yTrue, yPred):
		batchInverseDice = SoftDiceLoss(1-yTrue, 1-yPred, self.epsilon, axis=[2, 3])
		loss = 1-batchInverseDice
		return loss

class BCEloss(torch.nn.Module):
	def __init__(self, reduction='none'):
		super(BCEloss, self).__init__()
		self.lossLayer = torch.nn.BCELoss(reduction=reduction)

	def forward(self, yTrue, yPred):
		loss = self.lossLayer(yPred, yTrue)
		loss = torch.sum(loss, axis=[2, 3])
		return loss

class CombinedLoss(torch.nn.Module):
	def __init__(self, lamda, loss_type='combined', reduction='none'):
		super(CombinedLoss, self).__init__()
		self.reduction = reduction
		self.lamda = lamda
		self.loss_type = loss_type

	def forward(self, yTrue, yPred):
		LD = DiceLoss()(yTrue, yPred)
		LI = InverseDiceLoss()(yTrue, yPred)
		LC = BCEloss(self.reduction)(yTrue, yPred)
		lamda = self.lamda
		if self.loss_type == 'dice':
			loss = torch.sum(LD)
		elif self.loss_type == 'idice':
			loss = torch.sum(LI)
		elif self.loss_type == 'dice_combo':
			loss = torch.sum(lamda*LD+(1-lamda)*LI)
		elif self.loss_type == 'bce':
			loss = torch.sum(LC)
		elif self.loss_type == 'combined':
			loss = torch.sum(LC + self.lamda*LD + (1-self.lamda)*LI)
		else:
			print("Loss not implemented !")
		return loss

