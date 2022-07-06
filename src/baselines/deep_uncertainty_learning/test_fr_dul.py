# test face recognition performance of dul model
import torch
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from config import parse_args, Backbone_Dict, Test_FR_Data_Dict
from util.utils import get_data_pair, evaluate_model


class DUL_FR_Tester:
    def __init__(self, dul_args) -> None:
        self.dul_args = dul_args
        self.dul_args.multi_gpu = False

    def face_recog(self):
        BACKBONE = Backbone_Dict[self.dul_args.dataset]
        if os.path.isfile(self.dul_args.model_for_test):
            print("=" * 60, flush=True)
            print(
                "Model for testing Face Recognition performance is:\n '{}' ".format(
                    self.dul_args.model_for_test
                ),
                flush=True,
            )
            BACKBONE.load_state_dict(torch.load(self.dul_args.model_for_test))
        else:
            print("=" * 60, flush=True)
            print("No model found for testing!", flush=True)
            print("=" * 60, flush=True)
            return
        print("=" * 60, flush=True)
        print(
            "Face Recognition Performance on different dataset is as shown below:",
            flush=True,
        )
        print("=" * 60, flush=True)

        for name in Test_FR_Data_Dict.values():

            if name == "MNIST":
                dataset_test = datasets.MNIST(
                    root=self.dul_args.data_dir,
                    train=False,
                    transform=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ]
                    ),
                    download=True,
                )

            else:
                raise NotImplementedError(f"{name} is not implemented yet!")

            dataloader = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=self.dul_args.batch_size,
                pin_memory=self.dul_args.pin_memory,
                num_workers=self.dul_args.num_workers,
                drop_last=self.dul_args.drop_last,
            )

            evaluate_model(
                dataloader,
                self.dul_args.multi_gpu,
                BACKBONE,
            )

            # print(value.upper(), ": ", accuracy, flush=True)
        print("=" * 60, flush=True)
        print("Testing finished!", flush=True)
        print("=" * 60, flush=True)


if __name__ == "__main__":
    dul_fr_test = DUL_FR_Tester(parse_args())
    dul_fr_test.face_recog()
