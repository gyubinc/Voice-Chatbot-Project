import argparse

def get_args():
    # 기본 설정값만 남기고, hyperparameter는 wandb에게 맡김.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_epoch', default=15, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='./data/train.csv')
    parser.add_argument('--dev_path', default='./data/dev.csv')
    parser.add_argument('--test_path', default='./data/dev.csv')
    parser.add_argument('--predict_path', default='./data/test.csv')
    parser.add_argument('--ckpt_path', default='./checkpoints')
    args = parser.parse_args(args=[])
    return args