TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE = 150
LR = 1e-3
MODEL_NAME = 'meowfinder-{}-{}'.format(LR, 'basic')

animals = ['cat', 'dog', 'crab', 'rabbit', 'monkey', 'spider', 'wasp', 'shark', 'scorpion', 'elephant', 'snake',
           'turtle']

num_animals = [0 for i in range(11)]
len_animals = len(animals)
