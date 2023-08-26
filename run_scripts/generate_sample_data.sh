

CIDX=$1


list=(
# "/data5/wuzhongkai/proj/nerfstudio/outputs/blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn_meanReduce_zeropadding/drums/eg3d/2023-08-24_183609"
# /data5/wuzhongkai/proj/nerfstudio/outputs/blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn_meanReduce_zeropadding/chair/eg3d/2023-08-24_171306
# "/data5/wuzhongkai/proj/nerfstudio/outputs/blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn_meanReduce_zeropadding/ficus/eg3d/2023-08-24_200048"
# "/data5/wuzhongkai/proj/nerfstudio/outputs/blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn_meanReduce_zeropadding/hotdog/eg3d/2023-08-24_212007"
# "/data5/wuzhongkai/proj/nerfstudio/outputs/blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn_meanReduce_zeropadding/materials/eg3d/2023-08-24_224115"
# "/data5/wuzhongkai/proj/nerfstudio/outputs/blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn_meanReduce_zeropadding/mic/eg3d/2023-08-25_000227"
# "/data5/wuzhongkai/proj/nerfstudio/outputs/blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn_meanReduce_zeropadding/lego/eg3d/2023-08-23_172254"
/data5/wuzhongkai/proj/nerfstudio/outputs/blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn_meanReduce_zeropadding/ship/eg3d/2023-08-25_012555
)


for ((i=0; i<${#list[*]}; i++))
do
echo $dir
CUDA_VISIBLE_DEVICES=$CIDX ns-render camera-path \
    --load-config ${list[i]}/config.yml \
    --output-path ${list[i]}/images \
    --image-format png \
    --rendered-output-names rgb \
    --camera-path-filename ./dataset_blender_r4.03.json \
    --output-format images
done