import torch
import pytorch_lightning as pl

class LitDiffusionModel(pl.LightningModule):
    def __init__(self, n_dim=3, n_steps=200, lbeta=1e-5, ubeta=1e-2):
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
        t = torch.zeros(200,4)
        for i in range(200):
            for j in range(2):
                t[i][2*j]= torch.sin(i/(torch.pow(10000,torch.tensor(2*j/4))))
                t[i][2*j+1]= torch.cos(i/(torch.pow(10000,torch.tensor(2*j/4))))
        self.time_embed = t
        self.model = torch.nn.Sequential(
          nn.Linear(7,10),
          nn.ReLU(),
          nn.Linear(10,3)
        )

        """
        Be sure to save at least these 2 parameters in the model instance.
        """
        self.n_steps = n_steps
        self.n_dim = n_dim

        """
        Sets up variables for noise schedule
        """
        self.betas = self.init_alpha_beta_schedule(lbeta, ubeta)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas)
        self.alpha_bar_minus = torch.cat((torch.tensor([1]),self.alpha_bar[:-1]))
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
        t_embed = self.time_embed(t)
        return self.model(torch.cat((x, t_embed), dim=1).float())

    def init_alpha_beta_schedule(self, lbeta, ubeta):
        """
        Set up your noise schedule. You can perhaps have an additional hyperparameter that allows you to
        switch between various schedules for answering q4 in depth. Make sure that this hyperparameter 
        is included correctly while saving and loading your checkpoints.
        """
        return torch.linspace(lbeta,ubeta,self.n_steps)
    def index(self,t,i,x): # helper function to get the tensors corresponding to given dimension
        
        x1 = torch.gather(t,0,i)
        return x1.reshape(len(x.shape[0]),[1]*(len(x.shape)-1))

    def q_sample(self, x, t):
        """
        Sample from q given x_t.
        """
        i = torch.randn_like(x)
        return self.index(self.sqrt_alpha_bar,t,x)*x + self.index(1-self.alpha_bar,t,x)*i,i
        

    def p_sample(self, x, t):
        """
        Sample from p given x_t.
        
        """
        z = torch.randn_like(x)
        s =  1/(self.index(self.alpha,t,x))*(x - (self.index(self.betas,t,x)/(torch.sqrt(1-self.index(self.alpha_bar,t,x))))*model(x,t))
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
        t = torch.randint(0, self.n_steps, (len(batch_idx),))
        batch_noise,noise = self.q_sample(batch[batch_idx],t)
        e_theta = model(torch.cat(batch_noise,t))
        loss = nn.MSELoss()
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
        

    def configure_optimizers(self):
        """
        Sets up the optimizer to be used for backprop.
        Must return a `torch.optim.XXX` instance.
        You may choose to add certain hyperparameters of the optimizers to the `train.py` as well.
        In our experiments, we chose one good value of optimizer hyperparameters for all experiments.
        """
        pass
