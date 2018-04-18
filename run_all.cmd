REM python compare.py -m zf_unet         -d dsb2018 -e 100 -p 512 -b 4 -dd e:/datasets
REM python compare.py -m selunet         -d dsb2018 -e 100 -p 512 -b 4 -dd e:/datasets
REM python compare.py -m linknet         -d dsb2018 -e 100 -p 512 -b 4 -dd e:/datasets
REM python compare.py -m dilated_unet    -d dsb2018 -e 100 -p 512 -b 4 -dd e:/datasets
python compare.py -m tiramisu        -d dsb2018 -e 100 -p 512 -b 1 -dd e:/datasets
python compare.py -m dilated_resnet  -d dsb2018 -e 100 -p 512 -b 4 -dd e:/datasets
