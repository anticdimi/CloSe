import warnings

from lib.utils.types import EasierDict

from lib import factory
import yaml
import sys

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    cfg = EasierDict(yaml.load(open(sys.argv[1], 'r'), Loader=yaml.FullLoader))

    train_data, val_data, test_data = factory.get_dataset_split(cfg)

    model = factory.get_model(cfg)

    trainer = factory.get_trainer(model, train_data, val_data, test_data, cfg)

    trainer.train_model()
