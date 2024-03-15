from .utils.types import EasierDict
from .closed import TorchCloSeDataset
from .closenet import CloSeNet
from .closenet.trainers import CloSeNetTrainer


def get_dataset(cfg: EasierDict, mode: str) -> TorchCloSeDataset:
    return TorchCloSeDataset(
        mode=mode,
        data_path=cfg.data.data_path,
        split_file=cfg.data.split_file,
        pointcloud_samples=cfg.data.pointcloud_samples,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        n_classes=cfg.data.n_classes,
        input=cfg.data.input,
    )


def get_dataset_split(
    cfg: EasierDict,
) -> tuple[TorchCloSeDataset, TorchCloSeDataset, TorchCloSeDataset]:
    train = get_dataset(cfg, 'train')
    val = get_dataset(cfg, 'val')
    test = get_dataset(cfg, 'test')

    return train, val, test


def get_model(cfg: EasierDict) -> CloSeNet:
    return CloSeNet(
        cfg=cfg,
    )


def get_trainer(
    model: CloSeNet,
    train_dataset: TorchCloSeDataset,
    val_dataset: TorchCloSeDataset,
    test_dataset: TorchCloSeDataset,
    cfg: EasierDict,
) -> CloSeNetTrainer:
    return CloSeNetTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        cfg=cfg,
    )
