
import torch


def main():
    ckpt = torch.load('/data5/wuzhongkai/proj/nerfstudio/outputs/mimic3d_ckpts/blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn.ckpt')
    import pdb; pdb.set_trace()
    model_state_dict = ckpt['model_state_dict']

    print(model_state_dict)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()