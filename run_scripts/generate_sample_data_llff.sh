

CIDX=$1


list=(
# /data5/wuzhongkai/proj/nerfstudio/outputs/llff_512_bicubic/fern/eg3d/2023-09-03_235659
# /data5/wuzhongkai/proj/nerfstudio/outputs/llff_512_bicubic/flower/eg3d/2023-09-03_235659
# /data5/wuzhongkai/proj/nerfstudio/outputs/llff_512_bicubic/fortress/eg3d/2023-09-03_235659
# /data5/wuzhongkai/proj/nerfstudio/outputs/llff_512_bicubic/horns/eg3d/2023-09-03_235659
# /data5/wuzhongkai/proj/nerfstudio/outputs/llff_512_bicubic/leaves/eg3d/2023-09-03_235659
# /data5/wuzhongkai/proj/nerfstudio/outputs/llff_512_bicubic/orchids/eg3d/2023-09-03_235659
# /data5/wuzhongkai/proj/nerfstudio/outputs/llff_512_bicubic/room/eg3d/2023-09-03_235659
# /data5/wuzhongkai/proj/nerfstudio/outputs/llff_512_bicubic/trex/eg3d/2023-09-03_235659
# /data5/wuzhongkai/proj/nerfstudio/outputs/llff_64_tv1/fern/eg3d/2023-09-03_213943
# /data5/wuzhongkai/proj/nerfstudio/outputs/llff_64_tv1/flower/eg3d/2023-09-03_213943
# /data5/wuzhongkai/proj/nerfstudio/outputs/llff_64_tv1/fortress/eg3d/2023-09-03_213942
# /data5/wuzhongkai/proj/nerfstudio/outputs/llff_64_tv1/horns/eg3d/2023-09-03_213942
# /data5/wuzhongkai/proj/nerfstudio/outputs/llff_64_tv1/leaves/eg3d/2023-09-03_213943
# /data5/wuzhongkai/proj/nerfstudio/outputs/llff_64_tv1/orchids/eg3d/2023-09-03_213942
# /data5/wuzhongkai/proj/nerfstudio/outputs/llff_64_tv1/room/eg3d/2023-09-03_213943
# /data5/wuzhongkai/proj/nerfstudio/outputs/llff_64_tv1/trex/eg3d/2023-09-03_213942

)


obj_name_list=(
"fern"
"flower"
"fortress"
"horns"
"leaves"
"orchids"
"room"
"trex"
)


for ((i=0; i<${#list[*]}; i++))
do
echo $dir
CUDA_VISIBLE_DEVICES=$CIDX ns-render camera-path \
    --load-config ${list[i]}/config.yml \
    --output-path ./outputs/llff_64nerf_eg3d_tv1_sampling/${obj_name_list[i]}/images \
    --image-format png \
    --rendered-output-names rgb \
    --camera-path-filename ./sampling_paths/dataset_llff_64/${obj_name_list[i]}.json \
    --output-format images
done