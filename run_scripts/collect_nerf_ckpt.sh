
#!/bin/bash
OD=$1

list=(
/data5/wuzhongkai/proj/nerfstudio/outputs/llff_512_bicubic_tv1/fern/eg3d/2023-09-08_013836
/data5/wuzhongkai/proj/nerfstudio/outputs/llff_512_bicubic_tv1/flower/eg3d/2023-09-08_013836
/data5/wuzhongkai/proj/nerfstudio/outputs/llff_512_bicubic_tv1/fortress/eg3d/2023-09-08_013836
/data5/wuzhongkai/proj/nerfstudio/outputs/llff_512_bicubic_tv1/horns/eg3d/2023-09-08_013836
/data5/wuzhongkai/proj/nerfstudio/outputs/llff_512_bicubic_tv1/leaves/eg3d/2023-09-08_013837
/data5/wuzhongkai/proj/nerfstudio/outputs/llff_512_bicubic_tv1/orchids/eg3d/2023-09-08_013836
/data5/wuzhongkai/proj/nerfstudio/outputs/llff_512_bicubic_tv1/room/eg3d/2023-09-08_013836
/data5/wuzhongkai/proj/nerfstudio/outputs/llff_512_bicubic_tv1/trex/eg3d/2023-09-08_013836
)


obj_list=(
"fern"
"flower"
"fortress"
"horns"
"leavs"
"orchids"
"room"
"trex"
)


# mkdir OD
for ((i=0; i<${#list[*]}; i++))
do
echo $dir
# CUDA_VISIBLE_DEVICES=$CIDX ns-render camera-path \
#     --load-config ${list[i]}/config.yml \
#     --output-path ./outputs/llff_64nerf_eg3d_tv1_sampling/${obj_name_list[i]}/images \
#     --image-format png \
#     --rendered-output-names rgb \
#     --camera-path-filename ./sampling_paths/dataset_llff_64/${obj_name_list[i]}.json \
#     --output-format images
cp ${list[i]}/nerfstudio_models/step-000030000.ckpt /data5/wuzhongkai/proj/nerfstudio/outputs/nerfstudio_llff_eg3d_512bic_tv1
mv /data5/wuzhongkai/proj/nerfstudio/outputs/nerfstudio_llff_eg3d_512bic_tv1/step-000030000.ckpt /data5/wuzhongkai/proj/nerfstudio/outputs/nerfstudio_llff_eg3d_512bic_tv1/${obj_list[i]}.ckpt
done