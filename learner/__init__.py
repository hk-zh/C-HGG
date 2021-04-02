from .normal import NormalLearner
from .hgg import HGGLearner
from .her_goalGAN import HERGoalGANLearner
from .normal_goalGAN import NormalGoalGANLearner

learner_collection = {
	'normal': NormalLearner,
	'hgg': HGGLearner,
	'her+goalGAN': HERGoalGANLearner,
	'normal+goalGAN': NormalGoalGANLearner
}

def create_learner(args):
	return learner_collection[args.learn](args)