from .base_opt import BaseOptions


class TrainOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_start', type=int, default=1, help='the starting epoch')
        parser.add_argument('--num_warmup_steps', type=int, default=0, help='warmpup steps for the scheduler')
        parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
        parser.add_argument('--eps_adam', type=float, default=1e-8, help='eps term of adam')
        parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate for adam')


        self.isTrain = True
        return parser
