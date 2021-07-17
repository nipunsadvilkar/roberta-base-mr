import argparse

from transformers import RobertaForMaskedLM


def convert_flax_model_to_torch(flax_model_path: str, torch_model_path: str = "./"):
    """
    Converts Flax model weights to PyTorch weights.
    """
    model = RobertaForMaskedLM.from_pretrained(flax_model_path, from_flax=True)
    model.save_pretrained(torch_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flax to Pytorch model coversion")
    parser.add_argument(
        "--flax_model_path", type=str, default="flax-community/roberta-base-mr", help="Flax model path"
    )
    parser.add_argument("--torch_model_path", type=str, default="./", help="PyTorch model path")
    args = parser.parse_args()
    convert_flax_model_to_torch(args.flax_model_path, args.torch_model_path)
