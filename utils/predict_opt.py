from .base_opt import BaseOptions

class PredictOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='predictions', help='saves predictions here.')
        self.isTrain = False
        return parser
