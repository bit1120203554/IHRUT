import utils


# 继承utils的config
class Config(utils.Config):
    def __init__(self, config_path):
        super().__init__(config_path)

    def set_model_params(
        self,
        model_type="INV",
        model_name=None,
        loss_type="L1",
        num_stages=0,
        sharing=True,
        prior_type="DnCNN",
        prior_nc=128,
        prior_nb=17,
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.loss_type = loss_type
        self.num_stages = num_stages
        self.sharing = sharing
        self.prior_type = prior_type
        self.prior_nc = prior_nc
        self.prior_nb = prior_nb
        pass
