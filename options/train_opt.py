from .base_opt import BaseOptions


class TrainOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs with the initial learning rate')
	parser.add_argument('--eps_adam', type=float, default=1.e-8, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=1.e-5, help='initial learning rate for adam')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True
        return parser
