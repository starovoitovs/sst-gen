from torch import optim

from genhack.experiments import Experiment


class GANExperiment(Experiment):

    def configure_optimizers(self):
        d_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.discriminator.parameters()), lr=self.params['discriminator_learning_rate'])
        g_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.generator.parameters()), lr=self.params['generator_learning_rate'])
        return d_optimizer, g_optimizer

    def training_step(self, batch, batch_idx, optimizer_idx=None):

        result = self.model(batch[0])
        loss = self.model.loss(*result)
        self.log_dict({key: val.item() for key, val in loss.items()})

        if optimizer_idx == 0 and batch_idx % (self.params['n_discriminator_steps'] + self.params['n_generator_steps']) < self.params['n_discriminator_steps']:
            return list(loss.values())[0]
        elif optimizer_idx == 1 and batch_idx % (self.params['n_discriminator_steps'] + self.params['n_generator_steps']) >= self.params['n_discriminator_steps']:
            return list(loss.values())[1]
