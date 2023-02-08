import torch
import pytorch_lightning as pl

class LitDiffusionModel(pl.LightningModule):
    def __init__(self, n_dim=3, n_steps=200, lbeta=1e-5, ubeta=1e-2) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.time_embed = None
        self.model = None
        self.n_steps = n_steps
        self.n_dim = n_dim
        # Sets up variables self.betas, self.alphas and self.alpha_bars
        self.init_alpha_beta_schedule(lbeta, ubeta)

    def forward(self, x, t):
        if not isinstance(t, torch.Tensor):
            t = torch.LongTensor([t]).expand(x.size(0))
        t_embed = self.time_embed(t)
        return self.model(torch.cat((x, t_embed), dim=1).float())

    def init_alpha_beta_schedule(self, lbeta, ubeta):
        pass

    def q_sample(self, x, t):
        pass

    def p_sample(self, x, t):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def sample(self, n_samples, progress=False, return_intermediate=False):
        pass

    def configure_optimizers(self):
        pass
