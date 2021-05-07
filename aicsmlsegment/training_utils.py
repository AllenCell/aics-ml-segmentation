def _log_lr(self):
    lr = self.optimizer.param_groups[0]["lr"]
    self.writer.add_scalar("learning_rate", lr, self.num_iterations)


def _log_stats(self, phase, loss_avg, eval_score_avg):
    tag_value = {
        f"{phase}_loss_avg": loss_avg,
        f"{phase}_eval_score_avg": eval_score_avg,
    }

    for tag, value in tag_value.items():
        self.writer.add_scalar(tag, value, self.num_iterations)


def _log_params(self):
    self.logger.info("Logging model parameters and gradients")
    for name, value in self.model.named_parameters():
        self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
        self.writer.add_histogram(
            name + "/grad", value.grad.data.cpu().numpy(), self.num_iterations
        )


def _log_images(self, input, target, prediction):
    sources = {
        "inputs": input.data.cpu().numpy(),
        "targets": target.data.cpu().numpy(),
        "predictions": prediction.data.cpu().numpy(),
    }
    for name, batch in sources.items():
        for tag, image in self._images_from_batch(name, batch):
            self.writer.add_image(tag, image, self.num_iterations, dataformats="HW")
