import torch
import pytorch_lightning as pl
import numpy as np
import math

class LitDiffusionModel(pl.LightningModule):
    def __init__(self, n_dim=3, n_steps=200, lbeta=1e-5, ubeta=1e-2,embeddim=4,lr=0.0001,cos='lin'):
        super().__init__()
        """
        If you include more hyperparams (e.g. `n_layers`), be sure to add that to `argparse` from `train.py`.
        Also, manually make sure that this new hyperparameter is being saved in `hparams.yaml`.
        """
        self.save_hyperparameters()

        """
        Your model implementation starts here. We have separate learnable modules for `time_embed` and `model`.
        You may choose a different architecture altogether. Feel free to explore what works best for you.
        If your architecture is just a sequence of `torch.nn.XXX` layers, using `torch.nn.Sequential` will be easier.
        
        `time_embed` can be learned or a fixed function based on the insights you get from visualizing the data.
        If your `model` is different for different datasets, you can use a hyperparameter to switch between them.
        Make sure that your hyperparameter behaves as expecte and is being saved correctly in `hparams.yaml`.
        """
        t = torch.zeros(n_steps,embeddim,dtype=torch.float)
        for i in range(n_steps):
            for j in range(embeddim//2):
                t[i][2*j]= torch.sin(i/(torch.pow(10000,torch.tensor(2*j/embeddim))))
                t[i][2*j+1]= torch.cos(i/(torch.pow(10000,torch.tensor(2*j/embeddim))))
        self.time_embed = t
        self.model = torch.nn.Sequential(
          torch.nn.Linear(3+embeddim,100),
          torch.nn.ReLU(),
          torch.nn.Linear(100,100),
          torch.nn.ReLU(),
          torch.nn.Linear(100,100),
          torch.nn.ReLU(),
          torch.nn.Linear(100,100),
          torch.nn.ReLU(),
          torch.nn.Linear(100,100),
          torch.nn.ReLU(),
          torch.nn.Linear(100,3),
        )

        """
        Be sure to save at least these 2 parameters in the model instance.
        """
        self.n_steps = n_steps
        self.n_dim = n_dim
        self.lr = lr
        self.cos = cos
        """
        Sets up variables for noise schedule as given in the paper
        """
        self.betas = self.init_alpha_beta_schedule(lbeta, ubeta)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas,axis=0)
        self.alpha_bar_minus = torch.cat((torch.tensor([1]),self.alpha_bar[:-1]),axis=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_alpha_bar_minus = torch.sqrt(self.alpha_bar_minus)
        self.sigma = torch.sqrt(((1-self.alpha_bar_minus)/(1-self.alpha_bar))*self.betas)

    def forward(self, x, t):
        """
        Similar to `forward` function in `nn.Module`. 
        Notice here that `x` and `t` are passed separately. If you are using an architecture that combines
        `x` and `t` in a different way, modify this function appropriately.
        """
        if not isinstance(t, torch.Tensor):
            t = torch.LongTensor([t]).expand(x.size(0))
        t_embed = self.time_embed[t.cpu()].to('cuda')
        return self.model(torch.cat((x, t_embed), dim=1).float())

    def init_alpha_beta_schedule(self, lbeta, ubeta):
        """
        Set up your noise schedule. You can perhaps have an additional hyperparameter that allows you to
        switch between various schedules for answering q4 in depth. Make sure that this hyperparameter 
        is included correctly while saving and loading your checkpoints.
        """
        s = 0.008 # paramter value set inside the paper improved denoising diffusion model
        if self.cos=='cos':
            t =(((torch.linspace(lbeta,ubeta,self.n_steps)/self.n_steps)+s)/(1+s))*(math.pi/2)
            t = torch.cos(t)
            t = t**2/(torch.cos(torch.tensor(((1+s)/s)*(math.pi/2))))**2
            t1 = torch.cat((torch.tensor([1]),t[:-1]),axis=0)
            t_embed = torch.clamp(1-(t/t1),max=0.999)
        elif self.cos=='sqrt':
            t= 1 - torch.sqrt(s+(torch.linspace(lbeta,ubeta,self.n_steps)/self.n_steps))
            t1 = torch.cat((torch.tensor([1]),t[:-1]),axis=0)
            t_embed = torch.clamp(1-(t/t1),max=0.999)
        elif self.cos=='quad':
            t_embed= torch.linspace(lbeta**0.5,ubeta**0.5,self.n_steps)**2
        elif self.cos=='cubic':
            t_embed = torch.linspace(lbeta**0.33,ubeta**0.33,self.n_steps)**3
        elif self.cos=='reci':
            t_embed = 1/(torch.linspace(self.n_steps,2,self.n_steps))
        elif self.cos=='const':
            t_embed = ubeta * torch.ones(self.n_steps)
        else:
            t_embed=torch.linspace(lbeta,ubeta,self.n_steps)
        
        return t_embed
       
        
        
        # return torch.linspace(lbeta,ubeta,self.n_steps)
    def index(self,t,i,x): # helper function to get the tensors corresponding to given dimension
        y = torch.tensor(np.take_along_axis(t.detach().numpy(),i.detach().cpu().numpy(),axis=0)).to('cuda')
        for i in range(len(x.shape)-1):
            y = y.unsqueeze(-1)
        return y

    def q_sample(self, x, t):
        """
        Sample from q given x_t.
        t -> (1024,1)
        """
        i = torch.randn_like(x).type(torch.float)
        return self.index(self.sqrt_alpha_bar,t,x)*x + self.index(1-self.alpha_bar,t,x)*i,i
        

    def p_sample(self, x, t):
        """
        Sample from p given x_t.
        
        """
        z = torch.randn_like(x).type(torch.float)
        t1=self.time_embed[t.cpu()]
        # print("x ",x.is_cuda)
        # print("t ",t.is_cuda)
        # print("t1 ",t1.is_cuda)
        # print("z ",z.is_cuda)
        x = x.to("cuda")
        t1 = t1.to("cuda")
        z = z.to("cuda")
        # print("index ",(self.index(self.alphas,t,x)).is_cuda)
        s =  (1/(torch.sqrt(self.index(self.alphas,t,x))))*(x - (self.index(self.betas,t,x)/(torch.sqrt(1-self.index(self.alpha_bar,t,x))))*self.model(torch.cat((x,t1),axis=1).to("cuda")))
        s += self.index(self.sigma,t,x)*z
        
        return s
        
        

    def training_step(self, batch, batch_idx):
        """
        Implements one training step.
        Given a batch of samples (n_samples, n_dim) from the distribution you must calculate the loss
        for this batch. Simply return this loss from this function so that PyTorch Lightning will 
        automatically do the backprop for you. 
        Refer to the DDPM paper [1] for more details about equations that you need to implement for
        calculating loss. Make sure that all the operations preserve gradients for proper backprop.
        Refer to PyTorch Lightning documentation [2,3] for more details about how the automatic backprop 
        will update the parameters based on the loss you return from this function.

        References:
        [1]: https://arxiv.org/abs/2006.11239
        [2]: https://pytorch-lightning.readthedocs.io/en/stable/
        [3]: https://www.pytorchlightning.ai/tutorials
        """
        # print(batch.shape)
        t = torch.randint(0, self.n_steps, (batch.shape[0],),device='cuda')
        batch_noise,noise = self.q_sample(batch,t)
        # t1=self.time_embed[t.cpu()].to('cuda')
        # print(t1.shape)
        e_theta = self.forward(batch_noise,t)
        loss = torch.nn.MSELoss()
        return loss(e_theta,noise)
        
        

    def sample(self, n_samples, progress=False, return_intermediate=False):
        """
        Implements inference step for the DDPM.
        `progress` is an optional flag to implement -- it should just show the current step in diffusion
        reverse process.
        If `return_intermediate` is `False`,
            the function returns a `n_samples` sampled from the learned DDPM
            i.e. a Tensor of size (n_samples, n_dim).
            Return: (n_samples, n_dim)(final result from diffusion)
        Else
            the function returns all the intermediate steps in the diffusion process as well 
            i.e. a Tensor of size (n_samples, n_dim) and a list of `self.n_steps` Tensors of size (n_samples, n_dim) each.
            Return: (n_samples, n_dim)(final result), [(n_samples, n_dim)(intermediate) x n_steps]
        """
        points = []
        self.model = self.model.to('cuda')
        if return_intermediate is False:
            x = torch.randn((n_samples,self.n_dim))
            for i in range(self.n_steps-1,-1,-1):
                t = torch.full((n_samples,), i, device='cuda', dtype=torch.long)
                x_t = self.p_sample(x,t)
                x=x_t
            return x
        else:
            x = torch.randn((n_samples,self.n_dim))
            points.append(x.detach().cpu().numpy())
            for i in range(self.n_steps-1,-1,-1):
                t = torch.full((n_samples,), i, device='cuda', dtype=torch.long)
                x_t = self.p_sample(x,t)
                x=x_t
                points.append(x.detach().cpu().numpy())
            # print(torch.tensor(points).shape)
            point = torch.tensor(points)
            return x.detach().cpu(),point
            

    def configure_optimizers(self):
        """
        Sets up the optimizer to be used for backprop.
        Must return a `torch.optim.XXX` instance.
        You may choose to add certain hyperparameters of the optimizers to the `train.py` as well.
        In our experiments, we chose one good value of optimizer hyperparameters for all experiments.
        """
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        return optimizer