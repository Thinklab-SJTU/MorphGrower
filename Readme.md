<h1 align="center"><b>MorphGrower</b></h1>
<p align="center">
    <a href="https://openreview.net/forum?id=ZTN866OsGx"><img alt="Publication" src="https://img.shields.io/static/v1?label=Pub&message=ICML%2724&color=purple"></a>
    <a href="https://github.com/Thinklab-SJTU/MorphGrower/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-blue" alt="PRs"></a>
    <a href="https://github.com/Thinklab-SJTU/MorphGrower/stargazers"><img src="https://img.shields.io/github/stars/Thinklab-SJTU/MorphGrower?color=red&label=Star" alt="Stars"></a>
</p>

Official implementation for our paper:

MorphGrower: A Synchronized Layer-by-layer Growing Approach for Plausible Neuronal Morphology Generation

Nianzu Yang, Kaipeng Zeng, Haotian Lu, Yexin Wu, Zexin Yuan, Danni Chen, Shengdian Jiang, Jiaxiang Wu, Yimin Wang, Junchi Yan

*Forty-first International Conference on Machine Learning* (**ICML 2024**, **Oral**)

## Codes for MorphGrower

### Folder Specification

- ```model/```: File used to define the MorphGrower model.
- ```pretrain/```: Folder for storing pre-trained models.
- ```scripts/```: Scripts used for training, metric calculation, and other tasks.
- ```utils/```: Scripts for neuron output processing and log file handling.
- ```generate.py```: Utilizing the MorphGrower model for neuron generation during the inference phase.
- ```measure.py```: Evaluating metrics on generated neuron data.
- ```train.py```: Training a MorphGrower model on a specified dataset.
- `pretrain.py`: pretrain on artificial dataset. 

### Package Dependency

```
torch: 2.0.0
numpy: 1.21.2
scikit-learn: 1.2.2
scipy: 1.12.0
pyg: 2.3.1
morphpy: 0.7.2
pandas: 1.3.5
```

### Prepare Data
We use four datasets，you can download them:

- ```VPM```: 
- ```M1-EXC and M1-INH```: https://download.brainimagelibrary.org/3a/88/3a88a7687ab66069/
- ```RGC```: https://osf.io/b4qtr/

When you use RGC dataset, please use MATLAB or Python to transfor .mat file to .swc
After you download datasets, you can use the function 'smooth_swc' in utiles/utils.py to preprocess the data.

### Run the Code

Pretraining:

```
python pretrain --seed ${seed} --lr ${lr} --bs ${bs} --dropout ${dropout} --max_length 32 --teaching 0.5 --train_ratio 0.7 --valid_ratio 0.15 --data_dir ${data} --dim ${dim} --device ${device} --epoch ${epoch} --base_log_dir ${log} --ordered
```

Train the MorphGrower :

```
python train.py --data_dir ${data} --base_log_dir ${log} --pretrained_path ${pretrain} --device ${device} --seed ${seed} --kappa {$kappa}
```

Then, we can generate results :

```
python generate.py --model_path ${model} --data_dir ${data} --output_dir ${output} --device ${device} --kappa ${kappa} --generate_layers -1 --only_swc
```

If you wish to evaluate metrics for the output neuron data, you can run :
```
python measure.py --data_path ${output}
```

## Citation

```bibtex
@inproceedings{
    yang2024morphgrower,
    title={MorphGrower: A Synchronized Layer-by-layer Growing Approach for Plausible Neuronal Morphology Generation},
    author={Nianzu Yang and Kaipeng Zeng and Haotian Lu and Yexin Wu and Zexin Yuan and Danni Chen and Shengdian Jiang and Jiaxiang Wu and Yimin Wang and Junchi Yan},
    booktitle={Forty-first International Conference on Machine Learning},
    year={2024},
    url={https://openreview.net/forum?id=ZTN866OsGx}
}
```


Welcome to contact us [yangnianzu@sjtu.edu.cn](mailto:yangnianzu@sjtu.edu.cn) or [zengkaipeng@sjtu.edu.cn](mailto:zengkaipeng@sjtu.edu.cn) for any question.