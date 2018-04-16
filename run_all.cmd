python compare.py -m zf_unet         -d dsb2018 -p 512 -b 4 -dd e:/datasets --gpu-fraction 0.3
REM python compare.py -m linknet         -d dsb2018 -p 512 -b 4 -dd e:/datasets --gpu-fraction 0.3
python compare.py -m dilated_unet    -d dsb2018 -p 512 -b 4 -dd e:/datasets --gpu-fraction 0.3
REM python compare.py -m dilated_resnet  -d dsb2018 -p 512 -b 4 -dd e:/datasets --gpu-fraction 0.3
