# CloSe-D Dataset

The dataset is available for download [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/HSWYzTreszKqf5Y).

As described in the paper, we release the following:

1. ***CloSe-Di***: We provide 1455 scans, segmentation labels, and SMPL registrations.
2. ***CloSe-Dc***: Due to licensing concerns, we release segmentation labels, SMPL registrations, and instructions for purchasing commercial 1732 scans. See below for more details.
3. ***CloSe-D++***: consisting of a subset of publicly available datasets ([THuman2-0](https://github.com/ytrock/THuman2.0-Dataset), [HuMMan](https://caizhongang.com/projects/HuMMan/), [3DHumans](https://cvit.iiit.ac.in/research/projects/cvit-projects/3dhumans)), we release the segmentation labels for ~1000 scans. For scans and their SMPL registrations please refer to original work.

### Dataset details

The dataset is provided in preprocessed `.npz` files. Each file represents one scan and contains the following fields:

```python
points # (N, 3) array of scan 3D points/vertices;
normals # (N, 3) array of scan per-point normals;
colors # (N, 3) array of scan RGB colors;
faces # (F, 3) array of scan faces; 
scale # (1,) scale;

labels # (N,) array of labels;
garments # (18,) binary encoding of garments present in scan; (See section 4.1.3 in the paper for details)

pose # (72,) SMPL pose parameters;
betas # (10,) SMPL shape parameters;
trans # (3,) SMPL translation parameters;

canon_pose # (N, 3) for each point in the scan, we find the closes SMPL vertex location in T-pose; (See section 4.1.2 in the paper for details)
```

The segmentation labels in our dataset are ordered as follows:

```python
 0: 'Hat',
 1: 'Body',
 2: 'Shirt',
 3: 'TShirt',
 4: 'Vest',
 5: 'Coat',
 6: 'Dress',
 7: 'Skirt',
 8: 'Pants',
 9: 'ShortPants',
 10: 'Shoes',
 11: 'Hoodies',
 12: 'Hair',
 13: 'Swimwear',
 14: 'Underwear',
 15: 'Scarf',
 16: 'Jumpsuits',
 17: 'Jacket'
```

#### Note
See the [prep_scan.py](../prep_scan.py) script to see how the data is prepared for inference.

## CloSe-Dc

CloSe-Dc constitues of scans from commercial sources. So we cannot release the scans. Please refer to these sources to obtain the scans:

1. ***Renderpeople*** : [Renderpeople scans](https://renderpeople.com/)
2. ***Twindom*** : [Twindom scans](https://web.twindom.com/)
3. ***AXYZ*** : [AXYZ scans](https://secure.axyz-design.com/en/shop/)

## Experiments on the Datasets

To run your training and evaluation experiments on the provided datasets or on your custom datasets, you need to keep a file which specifies the data partition you would like to use. 

An example split file for CloSe-Di, that should reside in `/data/split_closedi.npz`,  is provided in [the link](https://nextcloud.mpi-klsb.mpg.de/index.php/s/HSWYzTreszKqf5Y).

## Disclaimer

If you are using our dataset for your research, please cite our paper:

```bibtex
@inproceedings{antic2024close,
    title = {CloSe: A 3D Clothing Segmentation Dataset and Model},
    author = {Antic, Dimitrije and Tiwari, Garvita and Ozcomlekci, Batuhan  and Marin, Riccardo  and Pons-Moll, Gerard},
    booktitle = {International Conference on 3D Vision (3DV)},
    month = {March},
    year = {2024},
}
```

Moreover, if you are using the `CloSe-D++` subset, please also cite the original dataset papers.

[THuman2-0](https://github.com/ytrock/THuman2.0-Dataset)

```bibtex
@InProceedings{tao2021function4d,
        title={Function4D: Real-time Human Volumetric Capture from Very Sparse Consumer RGBD Sensors},
        author={Yu, Tao and Zheng, Zerong and Guo, Kaiwen and Liu, Pengpeng and Dai, Qionghai and Liu, Yebin},
        booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR2021)},
        month={June},
        year={2021},
}
```

 [HuMMan](https://caizhongang.com/projects/HuMMan/)

```bibtex
@inproceedings{cai2022humman,
    title={{HuMMan}: Multi-modal 4d human dataset for versatile sensing and modeling},
    author={Cai, Zhongang and Ren, Daxuan and Zeng, Ailing and Lin, Zhengyu and Yu, Tao and Wang, Wenjia and Fan,
            Xiangyu and Gao, Yang and Yu, Yifan and Pan, Liang and Hong, Fangzhou and Zhang, Mingyuan and
            Loy, Chen Change and Yang, Lei and Liu, Ziwei},
    booktitle={17th European Conference on Computer Vision, Tel Aviv, Israel, October 23--27, 2022,
                Proceedings, Part VII},
    pages={557--577},
    year={2022},
    organization={Springer}
}
```

[3DHumans](https://cvit.iiit.ac.in/research/projects/cvit-projects/3dhumans)

```bibtex
@article{Jinka2022,
		doi = {10.1007/s11263-022-01736-z},
		url = {https://doi.org/10.1007/s11263-022-01736-z},
		year = {2022},
		month = dec,
		publisher = {Springer Science and Business Media {LLC}},
		author = {Sai Sagar Jinka and Astitva Srivastava and Chandradeep Pokhariya and Avinash Sharma and P. J. Narayanan},
		title = {SHARP: Shape-Aware Reconstruction of People in Loose Clothing},
		journal = {International Journal of Computer Vision}
		}
```

If you have any questions or need further assistance, please contact us: [Dimitrije Antic](mailto:d.antic@uva.nl) or [Garvita Tiwari](mailto:gtiwari@mpi-inf.mpg.de).
