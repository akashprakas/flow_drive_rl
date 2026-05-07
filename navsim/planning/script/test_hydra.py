import hydra
from omegaconf import DictConfig

@hydra.main(config_path='config/metric_caching', config_name='default_metric_caching', version_base=None)
def main(cfg: DictConfig):
    print('cfg loaded')
    print(cfg)

if __name__ == '__main__':
    main()
