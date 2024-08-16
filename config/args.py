import argparse


def test_options():
    parser = argparse.ArgumentParser(description="Testing script.")
    parser.add_argument(
        "-exp",
        "--experiment",
        default="test",
        type=str,
        required=False,
        help="Experiment name"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="/home/npr/dataset/",
        type=str,
        required=False,
        help="Training dataset"
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=1,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="mse",
        help="Optimized for (default: %(default)s)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=3,
        help="GPU ID"
    )
    parser.add_argument(
        "--cuda",
        default=True,
        help="Use cuda"
    )
    parser.add_argument(
        "--save",
        default=True,
        help="Save model to disk"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        type=str,
        help="pretrained model path"
    )
    args = parser.parse_args()
    return args
