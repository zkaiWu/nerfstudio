

CIDX=$1


list=(
"/data5/wuzhongkai/proj/nerfstudio/outputs/blender_256bicubic_mipnerf/chair/mipnerf/2023-09-04_133800"
"/data5/wuzhongkai/proj/nerfstudio/outputs/blender_256bicubic_mipnerf/drums/mipnerf/2023-09-04_133800"
"/data5/wuzhongkai/proj/nerfstudio/outputs/blender_256bicubic_mipnerf/ficus/mipnerf/2023-09-04_133800"
"/data5/wuzhongkai/proj/nerfstudio/outputs/blender_256bicubic_mipnerf/hotdog/mipnerf/2023-09-04_133800"
"/data5/wuzhongkai/proj/nerfstudio/outputs/blender_256bicubic_mipnerf/lego/mipnerf/2023-09-04_133800"
"/data5/wuzhongkai/proj/nerfstudio/outputs/blender_256bicubic_mipnerf/materials/mipnerf/2023-09-04_133800"
"/data5/wuzhongkai/proj/nerfstudio/outputs/blender_256bicubic_mipnerf/mic/mipnerf/2023-09-04_133800"
"/data5/wuzhongkai/proj/nerfstudio/outputs/blender_256bicubic_mipnerf/ship/mipnerf/2023-09-04_133759"
)


obj_name_list=(
"chair"
"drums"
"ficus"
"hotdog"
"lego"
"materials"
"mic"
"ship"
)


for ((i=0; i<${#obj_name_list[*]}; i++))
do
echo $dir
CUDA_VISIBLE_DEVICES=$CIDX ns-render camera-path \
    --load-config ${list[i]}/config.yml \
    --output-path ./blender_256bicubic_sampling/${obj_name_list[i]}/images \
    --image-format png \
    --rendered-output-names rgb_fine \
    --camera-path-filename ./dataset_blender_r4.03_256bicubic.json \
    --output-format images
done
# CUDA_VISIBLE_DEVICES=$CIDX ns-render camera-path \
#     --load-config ${list[i]}/config.yml \
#     --output-path ${list[i]}/images \
#     --image-format png \
#     --rendered-output-names rgb \
#     --camera-path-filename ./dataset_llff_64/${obj_name_list[i]}.json \
#     --output-format images
# done