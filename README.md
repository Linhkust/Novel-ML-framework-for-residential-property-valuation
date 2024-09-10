



<h2 align="center">Real Estate Valuation with Multi-Source Image Fusion and Enhanced Machine Learning Pipeline
</h2>



## Abstract

Different machine learning (ML) models have been developed for real estate valuation, achieving superior performance compared to traditional models. These ML models usually use structured tabular data, overlooking the roles of multi-source unstructured data such as images. Most pre-vious studies use single feature configuration space for model training without considering the model performance sensitivity brought by various feature configuration parameters. To fill the gaps, this paper fuses multi-source image data, explores different feature configuration parameters to enrich feature configuration space, adopts four tree-based ML models including Random Forest (RF), Extremely Randomized Trees (Extra Tree), Extreme Gradient Boosting (XGBoost), and Light Gradient Boosting Machine (LightGBM), applies distributed computing techniques for ML pipeline training, and utilizes explainable artificial intelligence (XAI) methods for global and local model interpretability analysis. Results show that model performances with different feature combina-tions are significantly different, and it is necessary to test various feature configuration parameters for model training. Performances of RF and Extra Tree are significantly better than XGBoost and LightGBM. The best model pipeline is formulated based on Extra Tree. Incorporating multi-source image features can improve the model prediction accuracy. The image features show significant nonlinear effects on the housing prices, which facilitate public authorities, urban planners and real estate developers in the process of urban planning and design and project site selection.

- [x] fusing the multi-source images of exterior estate photos, street view images, and remote sensing images
- [x] utilizing multiple image feature extraction networks and circular dis-tance ranges to formulate a series of ML pipelines
- [x] using the server-client distributed computing technique to speed up the training process of ML pipelines
- [x] evaluating the ML pipelines’ performances against a set of metrics to identify the best one
- [x] enhancing the interpretability of the ML-based approach by analyzing the model-based global feature importance and the SHAP-based local feature importance.


## Research Methodology

![Methodology](https://github.com/Linhkust/Novel-ML-framework-for-residential-property-valuation/blob/main/paper%20images/image-new%20method_V1.png)



## Main Components of the Source Code

### 1. Data Preparation

This part mainly deals with tabular data cleaning and image retrieval (Google Street View images and estate photos). You can download our collected Google street view images [here](https://pan.quark.cn/s/b7befcbb258d), and estate photos [here](https://pan.baidu.com/s/14Ki5E8FDu3HdqKUSosJXqw?pwd=irhh ). Remote sensing images are imported into ArcGIS Pro to calculate NDVI, NDWI, and NDBI. We have collected the Landsat-8 GeoTIFF files and you can download [here](https://pan.quark.cn/s/ec04f161b51b). 



Procedures of Google Street View image retrieval:

![SVI](https://github.com/Linhkust/Novel-ML-framework-for-residential-property-valuation/blob/main/paper%20images/image-SVI%20collection.png)



If you find some Python packages are not installed, you can use the command line to install them:

```
pip install <package_name>
```

or

```
conda install <package_name>
```

or

Find out the source codes of the package and copy the scripts to the path where you store your Python packages. For example, it seems not workable you install HpBandSterSearchCV using pip or conda.

### 2. Feature Engineering

Our study uses the pre-trained Deeplabv3+ from [VainF](https://github.com/VainF/DeepLabV3Plus-Pytorch). Detailed codes of how to implement the semantic segmentation can be found on his Github page (truly thanks). The segmentation results of panoramas, GSVs, and validation datasets can be downloaded [here](https://drive.google.com/file/d/1kRfEm9HWoQXxguyI13AsyQwEFI_i2QR4/view?usp=sharing).

#### Validation sets of the Cityscapes Datasets

The [Cityscapes Dataset](https://www.cityscapes-dataset.com/) is intended for

* assessing the performance of vision algorithms for major tasks of semantic urban scene understanding: pixel-level, instance-level, and panoptic semantic labeling;
* supporting research that aims to exploit large volumes of (weakly) annotated data, e.g. for training deep neural networks.

The validation data sets include the ground truth and segmentation results of three cities: frankfurt, lindau, and munster. All Cityscapes datasets can be downloaded [here](https://drive.google.com/file/d/1J1B3Jc80RqGpHR4SO-WzXK5uVOwz0PZQ/view?usp=sharing).

### 3. Model Generation

#### Machine learning pipeline generation workflow

Our novel machine-learning framework uses different types of feature extractors and generators. Different combined features are integrated with four base models: Random Forest, Extra Tree, XGBoost, and LightGBM. The machine learning pipeline is generated with the following workflow:

![](https://github.com/Linhkust/Novel-ML-framework-for-residential-property-valuation/blob/main/paper%20images/image-ML%20pipeline%20generation.png)

### 4. Pipeline Execution and Application

#### Distributed computing configuration

Two Python files are created for distributed computing: `server.py` and `client.py`. You should run the `server.py` in the server node and `client.py` in the client nodes. You need to identify the Server IP address in `server.py`:

```python
# IP address, port, and authentication key
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

[^1]: Martinez-de-Pison, F. J., Gonzalez-Sendino, R., Aldama, A., Ferreiro-Cabello, J., & Fraile-Garcia, E. (2019). Hybrid methodology based on Bayesian optimization and GA-PARSIMONY to search for parsimony models by combining hyperparameter optimization and feature selection. Neurocomputing, 354, 20–26. https://doi.org/10.1016/j.neucom.2018.05.136
