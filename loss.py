import torch


class LPLoss(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, y, y_target):
        return torch.norm((y - y_target).flatten(1), p=self.p, dim=1).sum() / torch.numel(y)


# https://github.com/jiupinjia/stylized-neural-painting
def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(2)
    y_lin = y.unsqueeze(1)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    return c


def sinkhorn_loss(x, y, epsilon, niter, mass_x=None, mass_y=None):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """
    # The Sinkhorn algorithm takes as input three variables :
    C = cost_matrix(x, y)  # Wasserstein cost function

    nx = x.shape[1]
    ny = y.shape[1]
    batch_size = x.shape[0]

    if mass_x is None:
        # assign marginal to fixed with equal weights
        mu = torch.full([batch_size, nx], 1. / nx, device=device)
    else:  # normalize
        mass_x.data = torch.clamp(mass_x.data, min=0, max=1e9)
        mass_x = mass_x + 1e-9
        mu = (mass_x / mass_x.sum(dim=-1, keepdim=True))

    if mass_y is None:
        # assign marginal to fixed with equal weights
        nu = torch.full([batch_size, ny], 1. / ny, device=device)
    else:  # normalize
        mass_y.data = torch.clamp(mass_y.data, min=0, max=1e9)
        mass_y = mass_y + 1e-9
        nu = (mass_y / mass_y.sum(dim=-1, keepdim=True))

    def M(u, v):
        """
        Modified cost for logarithmic updates
        $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$
        """
        return (-C + u.unsqueeze(2) + v.unsqueeze(1)) / epsilon

    def lse(A):
        """
        log-sum-exp
        """
        return torch.log(torch.exp(A).sum(2, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.

    for i in range(niter):
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).transpose(dim0=1, dim1=2)).squeeze()) + v

    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C, dim=[1, 2])  # Sinkhorn cost

    return torch.mean(cost)


def sinkhorn_normalized(x, y, epsilon, niter, mass_x=None, mass_y=None):
    Wxy = sinkhorn_loss(x, y, epsilon, niter, mass_x, mass_y)
    Wxx = sinkhorn_loss(x, x, epsilon, niter, mass_x, mass_x)
    Wyy = sinkhorn_loss(y, y, epsilon, niter, mass_y, mass_y)
    return 2 * Wxy - Wxx - Wyy


class SinkhornLoss(torch.nn.Module):
    def __init__(self, epsilon=0.01, niter=5, normalize=False, resize=True):
        super(SinkhornLoss, self).__init__()
        self.epsilon = epsilon
        self.niter = niter
        self.normalize = normalize
        self.resize = resize

    def _mesh_grids(self, batch_size, h, w):
        a = torch.linspace(0.0, h - 1.0, h)
        b = torch.linspace(0.0, w - 1.0, w)
        y_grid = a.view(-1, 1).repeat(batch_size, 1, w) / h
        x_grid = b.view(1, -1).repeat(batch_size, h, 1) / w
        grids = torch.cat([y_grid.view(batch_size, -1, 1), x_grid.view(batch_size, -1, 1)], dim=-1)
        return grids

    def forward(self, y, y_target):
        batch_size, c, h, w = y_target.shape
        if h > 24 and self.resize:
            y = torch.nn.functional.interpolate(y, [24, 24], mode='area')
            y_target = torch.nn.functional.interpolate(y_target, [24, 24], mode='area')
            batch_size, c, h, w = y_target.shape

        canvas_grids = self._mesh_grids(batch_size, h, w).to(y.device)
        gt_grids = torch.clone(canvas_grids)

        mass_x = y.reshape(batch_size, -1)
        mass_y = y_target.reshape(batch_size, -1)
        if self.normalize:
            loss = sinkhorn_normalized(
                canvas_grids, gt_grids, epsilon=self.epsilon, niter=self.niter,
                mass_x=mass_x, mass_y=mass_y)
        else:
            loss = sinkhorn_loss(
                canvas_grids, gt_grids, epsilon=self.epsilon, niter=self.niter,
                mass_x=mass_x, mass_y=mass_y)

        return loss
