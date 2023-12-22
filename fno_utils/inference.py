import numpy as np
import torch
from utils.train_structure_warehouse import *

def inference_PINN_Step(model, mesh, load, field, varLOAD=False):
    use_cuda = torch.cuda.is_available()

    # prepare initial mesh - mParam

    #                     lambda,        mu,    Gc,     lc
    mParam = np.array([[4.1667e3,  6.2500e3,     1, 0.5e-3],    #fibre
                       [4.7953e3,  1.3525e3, 0.011, 0.5e-3],    #matrix
                       [    1e-3,      1e-3,  0.11, 0.5e-3]     #pore
                      ])
    nc = mParam.shape[1]

    # normalise the parameters
    for i in range(mParam.shape[1]):
        mParam[:, i] = mParam[:, i] / mParam[:, i].max() 

    nb, _, h, w = mesh.shape
    mParamMAP = torch.zeros(nc, nb, h, w)
    for ic in range(nc):
        for im in range(3):
            mParamMAP[ic, mesh[:, 0] == im + 1] = mParam[im, ic]
    mParamMAP = torch.moveaxis(mParamMAP, 1, 0)

    if use_cuda:
        mesh = mesh.cuda().float()
        load = load.cuda().float()
        field = field.cuda().float()
        mParamMAP = mParamMAP.cuda().float()

    #
    model.eval()
    with torch.no_grad():
        if varLOAD==True:
            pred = model(mParamMAP[:,:2], torch.unsqueeze(load[:, 0], 1))
        elif varLOAD==False:
            pred = model(mParamMAP[:,:2])
        
        loss = lossPhaseField_Energy(pred, field, mParamMAP)

    return pred, loss
	



def inference_mesh2stress(model, reconLoss, mesh, load, field):
    use_cuda = torch.cuda.is_available()

    # prepare initial mesh - mParam

    #                     lambda,        mu,    Gc,     lc
    mParam = np.array([[4.1667e3,  6.2500e3,     1, 0.5e-3],    #fibre
                       [4.7953e3,  1.3525e3, 0.011, 0.5e-3],    #matrix
                       [    1e-3,      1e-3,  0.11, 0.5e-3]     #pore
                      ])
    nc = mParam.shape[1]

    # normalise the parameters
    for i in range(mParam.shape[1]):
        mParam[:, i] = mParam[:, i] / mParam[:, i].max() 

    nb, _, h, w = mesh.shape
    mParamMAP = torch.zeros(nc, nb, h, w)
    for ic in range(nc):
        for im in range(3):
            mParamMAP[ic, mesh[:, 0] == im + 1] = mParam[im, ic]
    mParamMAP = torch.moveaxis(mParamMAP, 1, 0)

    if use_cuda:
        mesh = mesh.cuda().float()
        load = load.cuda().float()
        field = field.cuda().float()
        mParamMAP = mParamMAP.cuda().float()

    #
    model.eval()
    with torch.no_grad():
        pred = model(mParamMAP[:,:2])
        #loss = reconLoss(pred, torch.unsqueeze(field[:,0], 1))
        loss = reconLoss(pred, field)

    return pred, loss


def inference_mesh2stress_2(model, reconLoss, mesh, load, field, use_cuda=True):
    #use_cuda = torch.cuda.is_available()
    
    if use_cuda:
        mesh = mesh.cuda().float() / 3.
        load = load.cuda().float()
        field = field.cuda().float()
    else:
        mesh = mesh.cpu().float() / 3.
        load = load.cpu().float()
        field = field.cpu().float()
    
    #
    model.eval()
    with torch.no_grad():
        pred = model(mesh)
        #loss = reconLoss(pred, torch.unsqueeze(field[:,0], 1))
        loss = reconLoss(pred, field)
    
    return pred, loss

def inference_mesh2damage_2(model, reconLoss, mesh, load, field, use_cuda=True):
    #use_cuda = torch.cuda.is_available()
    
    if use_cuda:
        mesh = mesh.cuda().float() / 3.
        load = load.cuda().float()
        field = field.cuda().float()
    else:
        mesh = mesh.cpu().float() / 3.
        load = load.cpu().float()
        field = field.cpu().float()
    
    #
    model.eval()
    with torch.no_grad():
        pred = model(mesh)
        #loss = reconLoss(pred, torch.unsqueeze(field[:,0], 1))
        loss = reconLoss(pred, field)
    
    return pred, loss


def inference_mesh2stress_inMESH(model, reconLoss, mesh, load, field):
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        mesh = mesh.cuda().float()
        load = load.cuda().float()
        field = field.cuda().float()
    #
    model.eval()
    with torch.no_grad():
        pred = model(mesh)
        loss = reconLoss(pred, torch.unsqueeze(field[:,0], 1))

        return pred, loss
	

def inference_elas2crk(model, reconLoss, energy, load, field, use_cuda=True):
    #use_cuda = torch.cuda.is_available()

    if use_cuda:
        load = load.cuda().float()
        field = field.cuda().float()
        energy = energy.cuda().float()
    else:
        load = load.cpu().float()
        field = field.cpu().float()
        energy = energy.cpu().float()

    #
    model.eval()
    with torch.no_grad():
        pred = model(energy)
        loss = reconLoss(pred, field)

    return pred, loss
	

def inference_mesh2curve(model, mesh, curve):
    use_cuda = torch.cuda.is_available()

    nb, _, seq_len = curve.size()
    incrs = torch.unsqueeze(torch.linspace(0., 1., seq_len), 0).repeat(nb, 1)

    if use_cuda:
        mesh = mesh.cuda().float()
        incrs = incrs.cuda().float()
        model.cuda()

    #
    model.eval()
    with torch.no_grad():
        pred = model(mesh, incrs)

    return pred


