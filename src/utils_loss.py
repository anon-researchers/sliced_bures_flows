#utils_loss.py
from utils_slicing import slicing
from utils_distances import sliced_distance
from utils import rand_projections
import torch

from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize_scalar
# from torch import autograd
class loss:
    
    def __init__(self, ftype, stype, dtype, weighted):
        self.ftype = ftype
        self.stype = stype
        self.dtype = dtype
        self.weighted = weighted
        self.slice = slicing(self.stype)
        self.distance = sliced_distance(self.dtype, self.weighted)
    
    def compute(self, X, Y, weights=None, projections=None, num_projections=100, r=1, f=None, f_op=None, lam=1, iter=100, device='cuda', proj_out=False):
        if self.ftype == 'sliced':
            d = self.compute_sliced_distance(X, Y, weights=weights, projections=projections, num_projections=num_projections, r=r, device=device, proj_out=proj_out)
        elif self.ftype == 'max-sliced':
            d = self.compute_max_sliced_distance(X, Y, weights=weights, projections=projections, r=r, iter=iter, device=device, proj_out=proj_out)
        elif self.ftype == 'max-sliced-eig':
            d = self.compute_max_sliced_distance_eig(X, Y, weights=weights, projections=projections, r=r, iter=iter, device=device, proj_out=proj_out)
        elif self.ftype == 'distributional-sliced':
            d = self.compute_distributional_sliced_distance(X, Y, weights=weights, num_projections=num_projections, r=r, f=f, f_op=f_op, lam=lam, iter=iter, device=device, proj_out=proj_out)
        else:
            raise Exception("undefined function type")
        # if self.ftype == 'distributional':
            # X_projections, Y_projections = slicing(self.stype).get_slice(X,Y)
        return d

    def compute_sliced_distance(self, X, Y, weights=None, projections=None, num_projections=1000, r=1, device='cuda', proj_out=False):
        if proj_out:
            X_projections, Y_projections, directions = self.slice.get_slice(X, Y, projections=projections, num_projections=num_projections, r=r, device=device, proj_out=proj_out)
        else:
            X_projections, Y_projections = self.slice.get_slice(X, Y, projections=projections, num_projections=num_projections, r=r, device=device, proj_out=proj_out)
        # print("X_proj={}".format(X_projections[:10]))
        # print('y_proj={}'.format(Y_projections[:10]))
        # print('proj={}'.format(projections[:10]))
        if self.weighted :
            assert(weights is not None)
            d = self.distance.compute(X_projections, Y_projections, weights, 2)
        else:
            assert(weights is None)
            d = self.distance.compute(X_projections, Y_projections, 2)
        return (d, X_projections, Y_projections, directions) if proj_out else d 

    def compute_max_sliced_distance(self, X, Y, weights=None, projections=None, r=1, iter=100, device='cuda', proj_out=False):
        if projections is None:
            theta = torch.randn((1, X.shape[1]), device=device, requires_grad=True)
            theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
        else:
            theta = torch.tensor(projections, device=device, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=1e-3)
        for _ in range(iter):
            # X_projections, Y_projections = slicing(self.stype).get_slice(X,Y,theta) 
            # d = sliced_distance(self.dtype, self.weighted).compute(X_projections, Y_projections, weights, 2)
            d = self.compute_sliced_distance(X, Y, weights=weights, projections=theta, r=r, device=device)
            l = -d
            # print(l) 
            opt.zero_grad()
            l.backward(retain_graph=True)
            opt.step()
            theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
            # X_projections, Y_projections = slicing(self.stype).get_slice(X,Y,theta) 
            # d = sliced_distance(self.dtype, self.weighted).compute(X_projections, Y_projections, weights, 2)
        d = self.compute_sliced_distance(X, Y, weights=weights, projections=theta, r=r, device=device, proj_out=proj_out)
        return d
    
    def compute_max_sliced_distance_eig(self, X, Y, weights=None, projections=None, r=1, iter=100, device='cuda', proj_out=False):
        rho_X = (X.t()@X).detach().cpu().numpy()
        rho_Y = (Y.t()@Y).detach().cpu().numpy()

        w_bures_eig = max_sliced_bures_eig(rho_X, rho_Y)
        w_bures = torch.from_numpy(w_bures_eig).t().float()
        d = self.compute_sliced_distance(X, Y, weights=weights, projections=w_bures, r=r, device=device, proj_out=proj_out)
        return d
        
    
    def compute_distributional_sliced_distance(self, X, Y, weights=None, num_projections=1000, r=1, f=None, f_op=None, lam=1, iter=10, device='cuda', proj_out=False):
        dim = X.size(1)
        pro = rand_projections(dim, num_projections).to(device)
#         X_detach = X.detach()
#         Y_detach = Y.detach()
        for _ in range(iter):
            # with autograd.detect_anomaly():
            projections = f(pro)
            cos = cosine_distance_torch(projections, projections)
            reg = lam * cos
            d = self.compute_sliced_distance(X, Y, weights=weights, projections=projections, r=r, device=device)
            # X_projections, Y_projections = slicing(self.stype).get_slice(X_detach, Y_detach, projections)
            # d = sliced_distance(self.dtype, self.weighted).compute(X_projections, Y_projections, weights, 2)
            loss = reg - d
            f_op.zero_grad()
            loss.backward(retain_graph=True)
            f_op.step()
        projections = f(pro)
        d = self.compute_sliced_distance(X, Y, weights=weights, projections=projections, r=r, device=device, proj_out=proj_out)
        # d = self.compute_sliced_distance(X_detach, Y_detach, weights=weights, projections=projections, r=r, device=device)
        return d
    
def one_side_bures_obj(w,rho_x,rho_y):
    wXw = np.sqrt(np.max([0,w.T@rho_x@w]))
    wYw = np.sqrt(np.max([0,w.T@rho_y@w]))
    return (wXw - wYw)/np.sqrt(w.T@w)

def one_sided_max_sliced_bures_eig(rho_x, rho_y):
    sigmaI = 1e-7*np.identity(rho_x.shape[0])
    eig_obj = lambda w : one_side_bures_obj(w, rho_x, rho_y)
    get_eig = lambda gamma: eigsh(gamma*rho_x - rho_y - sigmaI , k=1, which='LA', maxiter=100)[1]
    f = lambda gamma : -eig_obj(get_eig(gamma))
    res = minimize_scalar(f, bounds=(1e-6, 1), method='bounded') 
    gamma_star = res.x
    w_star = get_eig(gamma_star)
    w_star /= np.sqrt(w_star.T@w_star)
    div = eig_obj(w_star)
    return w_star

def max_sliced_bures_eig(rho_x, rho_y):
    w12 = one_sided_max_sliced_bures_eig(rho_x, rho_y)
    w21 = one_sided_max_sliced_bures_eig(rho_y, rho_x)
    obj12 =  one_side_bures_obj(w12,rho_x,rho_y)
    obj21 =  one_side_bures_obj(w21,rho_x,rho_y)

    if np.abs(obj12) < np.abs(obj21):
        w12 = w21
    return w12

def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(torch.abs(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)))
