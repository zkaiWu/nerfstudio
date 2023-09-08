
import torch


def main():
    ckpt = torch.load('/data5/wuzhongkai/proj/nerfstudio/outputs/llff/horns/eg3d/2023-08-28_232533/nerfstudio_models/step-000030000.ckpt')
    import pdb; pdb.set_trace()
    model_state_dict = ckpt['model_state_dict']

    print(model_state_dict)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()