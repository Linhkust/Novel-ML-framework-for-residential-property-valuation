



<h2 align="center">A novel machine learning framework for residential property valuation with distributed computing and multi-source image data fusion
</h2>



## Abstract

The automated valuation model (AVM) has been widely used by the real estate industry and financial institutions for automatic residential property valuation. Different machine learning (ML) models have been applied to develop the AVM due to their superior performances compared to traditional linear and spatial regression models. However, research on comprehensive machine learning pipelines considering different types of feature extractors and generators for AVM development from scratch is still lacking. Current AVM mainly uses numerical property and locational variables for property valuation, and how images are utilized for AVM development requires further exploration. Therefore, this paper proposes a novel machine-learning pipeline considering different types of feature extractors and generators, and multi-source image fusion including exterior estate photos, street view images, and remote sensing images. Our proposed method includes three stages: data collection, tree-based ML pipeline creation, pipeline configuration, and application. Distributed computing is applied to execute the tree-based ML pipelines, followed by optimal pipeline selection using statistical tests, and pipeline application. The results show that image-based features contribute significantly to housing price predictions and should be well considered in the AVM development. Features extracted from street view images and remote sensing images have greater importance than those of exterior estate photos. We benchmark the proposed method with two open-source automated machine learning (AutoML) frameworks: H2O and TPOT. The results show that our proposed machine learning framework outperforms H2O and TPOT in terms of prediction accuracy and execution time.

* We examined the effects of exterior housing photos, Google street view images, and remote sensing images on the housing price prediction accuracy; 
* We improved the classical semantic segmentation model performance with computer vision foundation model for more accurate feature extraction
* We improved the ML model performance by combining feature selection and hyperparameter optimization with a hybrid method of Bayesian optimization and hyperband (BOHB) and recursive feature elimination with cross-validation (RFECV)
* We proposed a novel machine learning pipeline to determine the best features and ML model for AVM deployment and application



## Research Methodology

![Methodology](https://github.com/Linhkust/Novel-ML-framework-for-residential-property-valuation/blob/main/paper%20images/image-Research%20Framework.png)



## Main Components of the Source Code

### 1. Data Preparation

This part mainly deals with tabular data cleaning and image retrieval (Google Street view images and estate photos). You can download our collected Google street view images here, and estate photos [here](https://pan.baidu.com/s/14Ki5E8FDu3HdqKUSosJXqw?pwd=irhh ) via Baidu Cloud. Remote sensing images are imported into ArcGIS Pro to calculate NDVI, NDWI, and NDBI. We have collected the Landsat-8 GeoTIFF files and stored them in the collected data/Landsat8. 



Procedures of Google street view image retrieving:

![SVI](https://github.com/Linhkust/Novel-ML-framework-for-residential-property-valuation/blob/main/paper%20images/image-SVI%20collection.png)



If you find some python packages are not installed, you can use the commond line to install:

```
pip install <package_name>
```

or

```
conda install <package_name>
```

or

Find out the source codes of the package and copy the scripts to the path where you store your Python packages. For example, it seems not workable you install HypersearchCV using pip or conda.

### 2. Feature Engineering

Our study uses the pretrained Deeplabv3+ from [VainF](https://github.com/VainF/DeepLabV3Plus-Pytorch). Detailed codes of how to implement the semnatic segmentation can be found his Github page (truely thanks). The segmentation results of panoramas, gsvs and validation datasets can be downloaded [here]() via Baidu Cloud.

#### How to get the image masks using Segment Anything Model?

 Following the procedures of [SAM](https://github.com/facebookresearch/segment-anything), we can use command line to run amg.py in main/feature engineering/deeplabv3+:

```
python amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>
```

* **checkpoint:** The path of pretrained model (sam_vit_b_01ec64.pth, sam_vit_h_4b8939.pth, sam_vit_l_0b3195.pth)
* **model_type:** 'vit_h', 'vit_l', 'vit_b'
* **input:** directory of the images you want to create masks
* **output:** directory of the generated masks



#### Validation sets of the Cityscapes Datasets

The [Cityscapes Dataset](https://www.cityscapes-dataset.com/) is intended for

* assessing the performance of vision algorithms for major tasks of semantic urban scene understanding: pixel-level, instance-level, and panoptic semantic labeling;
* supporting research that aims to exploit large volumes of (weakly) annotated data, e.g. for training deep neural networks.

The validation data sets include the ground truth and segmentation results of three cities: frankfurt, lindau, and munster. All Cityscapes dataset can be download [here]().



### 3. Model Generation

#### Machine learning pipeline generation workflow

Our novel machine learning framework use different types of feature extractors and generators. Different combined features are integrated with four base models: Random forest, Extra Tree, XGBoost and LightGBM. The machine learning pipeline is generated with following workflow:

![](https://github.com/Linhkust/Novel-ML-framework-for-residential-property-valuation/blob/main/paper%20images/image-preprocessor.png)

#### Hybrid method of BOHB and RFECV

Inspried by the work of Martinez-de-Pison et al. (2019)[^1], we propose a hybrid method of BOHB and RFECV to integrate feature selection and hyperparameter optimization. The flowchart is shown as below:

![](https://github.com/Linhkust/Novel-ML-framework-for-residential-property-valuation/blob/main/paper%20images/BOHB_RFECV.png)

### 4. Pipeline Execution and Application

#### Distributed computing configuration

Two python files are created for distributed computing: `server.py` and `client.py`. You should run the `server.py` in server node and `client.py` in client nodes. You need to identify the Server IP address in `server.py`:

```python
# IP address, port and authentication key
manager = BaseManager(address=('Your Server IP Address', 5000), authkey=b'password')
```

and `client.py`:

```python
# Identify the Server IP address
server_address = 'Your Server IP Address'
```

When using distributed computing, the following function is applied for each client node:

```python
def pipelines_fit_single(self, pipeline):
    result = self.fit(self.data[pipeline[0]], pipeline[1], verbose=1)
    return result
```

If distributed computing is not available, the following function is applied for one PC:

```python
def pipelines_fit_stream(self, pipelines):
    results = []
    for num, pipeline in tqdm(enumerate(pipelines), total=len(pipelines)):
        print('Pipeline_{}/{} Training:'.format(num + 1, len(pipelines)))

        result = self.fit(self.data[pipeline[0]], pipeline[1], verbose=1)
        result.insert(0, 'Pipeline_{}'.format(num + 1))
        results.append(result)

    pred_results = pd.DataFrame(results,
                                columns=['pipeline_id', 'model_type', 'features', 		                                          'hyperparameters', 'Time',
                                         'before_r2', 'before_mae', 'before_rmse',
                                         'after_r2', 'after_mae', 'after_rmse',
                                         'improvement'])

    pred_results.to_csv('./collected data/results/Pipelines.csv', index=False)
```



#### Benchmark with TOPT and H2O

* [TPOT](https://github.com/EpistasisLab/tpot) is based on genetic programming and Pareto optimization to flexibly automate the process of designing and optimizing machine learning pipelines. It generates tree-based pipelines and use genetic programming (GP) to output the best pipeline. 
* [H2O](https://github.com//h2oai/h2o-3/blob/master/h2o-docs/src/product/automl.rst) is a distributed machine learning platform which automates the process of training a large selection of candidate models and stacked ensembles.



We send our 64 datasets to TPOT, H2O and our proposed machine learning framework using one PC. The distribution of R squared and execution time for one dataset are shown as follwoings. The results show that our proposed machine learning framework outperforms H2O and TPOT in terms of prediction accuracy and execution time.

![](https://github.com/Linhkust/Novel-ML-framework-for-residential-property-valuation/blob/main/paper%20images/performance.png)

![](https://github.com/Linhkust/Novel-ML-framework-for-residential-property-valuation/blob/main/paper%20images/benchmark_time.png)

![](https://github.com/Linhkust/Novel-ML-framework-for-residential-property-valuation/blob/main/paper%20images/r2_set.png)

![](https://github.com/Linhkust/Novel-ML-framework-for-residential-property-valuation/blob/main/paper%20images/time_set.png)

[^1]: Martinez-de-Pison, F. J., Gonzalez-Sendino, R., Aldama, A., Ferreiro-Cabello, J., & Fraile-Garcia, E. (2019). Hybrid methodology based on Bayesian optimization and GA-PARSIMONY to search for parsimony models by combining hyperparameter optimization and feature selection. Neurocomputing, 354, 20–26. https://doi.org/10.1016/j.neucom.2018.05.136
