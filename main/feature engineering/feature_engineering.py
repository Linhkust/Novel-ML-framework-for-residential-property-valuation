import shutil
from io import BytesIO
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import time
from pyproj import Transformer
from geopy.distance import geodesic
from scipy.stats import entropy
import warnings
import multiprocessing as mp
import datetime
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import pyplot as plt
import os
import sys
import cv2 as cv
from PIL import Image
from PIL import ImageFile
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import hdbscan

np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Feature(object):
    def __init__(self, data, view, GSV, facility, wifi, estate_img, n_jobs):
        self.data = data
        self.view = view
        self.GSV = GSV
        self.facility = facility
        self.wifi = wifi
        self.estate_img = estate_img
        self.n_jobs = n_jobs

    '''Feature Construction'''

    # Facility related variables
    def spatial_temporal_features(self, row):
        # Initial rectangular selection for reducing computation time
        facility_selection = self.facility[(self.facility['EASTING'] <= self.data.loc[row, 'x'] + 1000)
                                           & (self.facility['EASTING'] >=self.data.loc[row, 'x'] - 1000)
                                           & (self.facility['NORTHING'] >= self.data.loc[row, 'y'] - 1000)
                                           & (self.facility['NORTHING'] <= self.data.loc[row, 'y'] + 1000)].reset_index(
            drop=True)

        facility_selection['distance'] = facility_selection.apply(lambda x: geodesic((self.data.loc[row, 'Latitude'],
                                                                                      self.data.loc[row, 'Longitude']),
                                                                                     (x['Latitude'],
                                                                                      x['Longitude'])).m, axis=1)

        try:
            wifi_selection = self.wifi[(self.wifi['Easting'] <= self.data.loc[row, 'x'] + 1000)
                                       & (self.wifi['Easting'] >= self.data.loc[row, 'x'] - 1000)
                                       & (self.wifi['Northing'] >= self.data.loc[row, 'y'] - 1000)
                                       & (self.wifi['Northing'] <= self.data.loc[row, 'y'] + 1000)].reset_index(drop=True)

            wifi_selection['distance'] = wifi_selection.apply(lambda x: geodesic((self.data.loc[row, 'Latitude'],
                                                                                  self.data.loc[row, 'Longitude']),
                                                                                 (x['Latitude'],
                                                                                  x['Longitude'])).m,
                                                              axis=1)
            wifi_1km = wifi_selection[wifi_selection['distance'] <= 1000].reset_index(drop=True)
            wifi_density = len(wifi_1km)

        except (KeyError, ValueError):
            wifi_density = 0

        facilities_1km = facility_selection[facility_selection['distance'] <= 1000][
            ['GEONAMEID', 'CLASS', 'TYPE', 'distance']].reset_index(drop=True)

        # Generated Features
        variables = {}

        # POI density
        poi_density = len(facilities_1km)
        variables['wifi_hk'] = wifi_density
        variables['POI_density'] = poi_density

        # POI diversity
        # Number of CLASS and TYPE
        num_class = len(facilities_1km['CLASS'].unique())
        num_type = len(facilities_1km['TYPE'].unique())
        variables['Num_class'] = num_class
        variables['Num_type'] = num_type

        # Entropy-based CLASS diversity
        class_unique_num = facilities_1km['CLASS'].value_counts()
        class_unique_percentage = class_unique_num / class_unique_num.sum()
        class_unique_percentage = class_unique_percentage.tolist()
        class_entropy = entropy(class_unique_percentage, base=2)  # CLASS_Entropy
        variables['Class_diversity'] = class_entropy

        # Entropy-based TYPE diversity
        type_unique_num = facilities_1km['TYPE'].value_counts()
        type_unique_percentage = type_unique_num / type_unique_num.sum()
        type_unique_percentage = type_unique_percentage.tolist()
        type_entropy = entropy(type_unique_percentage, base=2)  # TYPE_Entropy
        variables['Type_diversity'] = type_entropy

        # Distance to the nearest unique TYPE of facility
        facility_type = facilities_1km['TYPE'].unique()
        for j in range(len(facility_type)):
            distance = facilities_1km[facilities_1km['TYPE'] == facility_type[j]]['distance'].min()
            variables[facility_type[j]] = distance
        return variables

    # Using parallel computing to speed up the process
    def parallel_feature_generation(self):
        # Testing
        data_count = len(self.data)
        pbar = tqdm(total=data_count, file=sys.stdout, colour='white')
        pbar.set_description('Feature Generation')
        update = lambda *args: pbar.update()

        # VERY IMPORTANT: check how many cores in your PC
        cpu_num = range(1, mp.cpu_count() + 1)
        processor_num = self.n_jobs if self.n_jobs > 0 else cpu_num[self.n_jobs]

        pool = mp.Pool(processes=processor_num)

        # 定义一个列表来存储每次循环的结果
        results = []

        # 并行运行for循环
        for num in range(data_count):
            # 将任务提交给进程池
            result = pool.apply_async(Feature.spatial_temporal_features,
                                      args=(self, num,),
                                      callback=update)
            results.append(result)

        # 等待所有进程完成
        pool.close()
        pool.join()

        pred_results = []
        # 打印每次循环的结果
        for result in results:
            pred_results.append(result.get())

        pred_results = pd.DataFrame(pred_results)
        # Combine property information and facility information
        final_data = pd.concat([self.data, pred_results], axis=1)
        final_data.to_csv('collected data/clean data/data_features.csv', index=False)

    # Building view, vegetation view, and sky view
    def gsv_view_index(self, img_path, img):
        # rgb values of sky, building and vegetation
        sky = [70, 130, 180]
        building = [70, 70, 70]
        vegetation = [107, 142, 35]

        img = cv.imread(img_path + img)
        height = img.shape[0]
        width = img.shape[1]
        max_pixels = height * width
        sky_count = 0
        building_count = 0
        vegetation_count = 0

        for i in range(height):
            for j in range(width):
                if img[i, j][0] == sky[2] and img[i, j][1] == sky[1] and img[i, j][2] == sky[0]:
                    sky_count += 1
                elif img[i, j][0] == building[2] and img[i, j][1] == building[1] and img[i, j][2] == building[0]:
                    building_count += 1
                elif img[i, j][0] == vegetation[2] and img[i, j][1] == vegetation[1] and img[i, j][2] == vegetation[0]:
                    vegetation_count += 1

        svi = "%.3f" % (sky_count / max_pixels)
        bvi = "%.3f" % (building_count / max_pixels)
        gvi = "%.3f" % (vegetation_count / max_pixels)
        return svi, bvi, gvi

    # Generate nearby gsv points (500m, 1000m)
    def nearby_gsv(self, distances):
        '''View index within distance threshold'''
        view = pd.read_csv('./data/view_index.csv')
        gsv_points = pd.read_csv('./data/GSV.csv')
        data = pd.read_csv('./data/Paper_data.csv')

        for i, distance in enumerate(distances):
            for row in tqdm(range(len(data))):
                gsv_selected = gsv_points[(gsv_points['POINT_X'] <= data.loc[row, 'x'] + distance)
                                          & (gsv_points['POINT_X'] >= data.loc[row, 'x'] - distance)
                                          & (gsv_points['POINT_Y'] >= data.loc[row, 'y'] - distance)
                                          & (gsv_points['POINT_Y'] <= data.loc[row, 'y'] + distance)].reset_index(
                    drop=True)

                gsv_points_distance = [x - 1 for x in gsv_selected['OBJECTID1'].values.tolist()]

                view_distance = view[view['img'].isin(gsv_points_distance)]

                data.loc[row, 'sky{}'.format(distance)] = view_distance['sky'].mean()
                data.loc[row, 'building{}'.format(distance)] = view_distance['building'].mean()
                data.loc[row, 'vegetation{}'.format(distance)] = view_distance['vegetation'].mean()

        data.to_csv('./data/gsv_view_index.csv', index=False)

    # GoogleNet, AlexNet, VGG-16, ResNet-101 for feature extraction
    def estate_photo_feature(self, cnn_type):
        img_names = os.listdir(self.estate_img)  # 获取目录下所有图片名称列表

        # AlexNet
        if cnn_type == 'AlexNet':
            data = np.zeros((len(img_names), 9216))  # 初始化一个np.array数组用于存数据
            alexnet = models.alexnet(pretrained=True)
            alexnet = torch.nn.Sequential(*list(alexnet.children())[:-1])

            alexnet.eval()

            for i, name in tqdm(enumerate(img_names), total=len(img_names)):
                image_path = self.estate_img + '/' + name
                img = Image.open(image_path).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((256, int(256 * img.height / img.width))),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

                img_tensor = transform(img)
                img_tensor = torch.unsqueeze(img_tensor, 0)

                features = alexnet(img_tensor)

                flatten_features = torch.flatten(features)
                data[i] = flatten_features.detach().numpy()

            # t-SNE into 2D and 3D
            scaler = MinMaxScaler()
            tsne_2d = TSNE(n_components=2, init='pca', random_state=0)
            tsne_3d = TSNE(n_components=3, init='pca', random_state=0)
            embedded_features_2d = tsne_2d.fit_transform(data)
            embedded_features_2d = scaler.fit_transform(embedded_features_2d)
            embedded_features_3d = tsne_3d.fit_transform(data)
            embedded_features_3d = scaler.fit_transform(embedded_features_3d)
            return np.hstack((embedded_features_2d, embedded_features_3d))

        # GoogleNet
        elif cnn_type == 'GoogleNet':
            data = np.zeros((len(img_names), 1000))  # 初始化一个np.array数组用于存数据
            model = models.googlenet(pretrained=True)

            for i, name in tqdm(enumerate(img_names), total=len(img_names)):
                image_path = self.estate_img + '/' + name
                img = Image.open(image_path).convert('RGB')
                preprocess = transforms.Compose([
                    transforms.Resize((256, int(256 * img.height / img.width))),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                input_tensor = preprocess(img)
                input_tensor = input_tensor.unsqueeze(0)

                outputs = model(input_tensor)
                data[i] = outputs[0].detach().numpy()

            # t-SNE into 2D and 3D
            scaler = MinMaxScaler()
            tsne_2d = TSNE(n_components=2, init='pca', random_state=0)
            tsne_3d = TSNE(n_components=3, init='pca', random_state=0)
            embedded_features_2d = tsne_2d.fit_transform(data)
            embedded_features_2d = scaler.fit_transform(embedded_features_2d)
            embedded_features_3d = tsne_3d.fit_transform(data)
            embedded_features_3d = scaler.fit_transform(embedded_features_3d)

            return np.hstack((embedded_features_2d, embedded_features_3d))

        # VGG-16
        elif cnn_type == 'VGG16':
            data = np.zeros((len(img_names), 4096))  # 初始化一个np.array数组用于存数据
            vgg16 = models.vgg16(pretrained=True)
            vgg16.eval()

            vgg16_features = torch.nn.Sequential(*list(vgg16.features.children()))
            vgg16_fc6 = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])

            for i, name in tqdm(enumerate(img_names), total=len(img_names)):
                image_path = self.estate_img + '/' + name
                img = Image.open(image_path).convert('RGB')
                preprocess = transforms.Compose([
                    transforms.Resize((256, int(256 * img.height / img.width))),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                input_tensor = preprocess(img).unsqueeze(0)

                # 获取特征
                features = vgg16_fc6(vgg16_features(input_tensor).view(input_tensor.size(0), -1))
                data[i] = features.detach().numpy().flatten()

            # t-SNE into 2D and 3D
            scaler = MinMaxScaler()
            tsne_2d = TSNE(n_components=2, init='pca', random_state=0)
            tsne_3d = TSNE(n_components=3, init='pca', random_state=0)
            embedded_features_2d = tsne_2d.fit_transform(data)
            embedded_features_2d = scaler.fit_transform(embedded_features_2d)
            embedded_features_3d = tsne_3d.fit_transform(data)
            embedded_features_3d = scaler.fit_transform(embedded_features_3d)

            return np.hstack((embedded_features_2d, embedded_features_3d))

        # ResNet101
        elif cnn_type == 'ResNet101':
            data = np.zeros((len(img_names), 2048))  # 初始化一个np.array数组用于存数据
            model = models.resnet101(pretrained=True)
            model.eval()

            for i, name in tqdm(enumerate(img_names), total=len(img_names)):
                image_path = self.estate_img + '/' + name
                img = Image.open(image_path).convert('RGB')
                preprocess = transforms.Compose([
                    transforms.Resize((256, int(256 * img.height / img.width))),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                input_tensor = preprocess(img)
                input_batch = input_tensor.unsqueeze(0)

                # 使用ResNet101模型进行前向传递
                with torch.no_grad():
                    features = model.conv1(input_batch)
                    features = model.bn1(features)
                    features = model.relu(features)
                    features = model.maxpool(features)

                    features = model.layer1(features)
                    features = model.layer2(features)
                    features = model.layer3(features)
                    features = model.layer4(features)

                    features = model.avgpool(features)
                    features = torch.flatten(features, 1)

                # 提取全连接层的特征
                fc_features = model.fc.in_features
                extracted_features = features.reshape(-1, fc_features)
                data[i] = extracted_features.detach().numpy()

            # t-SNE into 2D and 3D
            scaler = MinMaxScaler()
            tsne_2d = TSNE(n_components=2, init='pca', random_state=0)
            tsne_3d = TSNE(n_components=3, init='pca', random_state=0)
            embedded_features_2d = tsne_2d.fit_transform(data)
            embedded_features_2d = scaler.fit_transform(embedded_features_2d)
            embedded_features_3d = tsne_3d.fit_transform(data)
            embedded_features_3d = scaler.fit_transform(embedded_features_3d)

            return np.hstack((embedded_features_2d, embedded_features_3d))

    # POI_walk (within specific distance range, 300m, 500m)
    # View index from Google Street View Images: building view, vegetation view, sky view (500m, 1000m)
    # NDVI, NDBI, NDWI (within specific distance range, 500m, 1000m)
    # Dimension reduction from Estate Photos: 2D and 3D from GoogleNet, AlexNet, VGG-16, ResNet-101

    def t_sne_dbscan(n_components):
        title_index = ['a', 'b', 'c', 'd']
        if n_components == 2:
            fig = plt.figure(figsize=(10, 10))
            for i, type in enumerate(['AlexNet', 'GoogleNet', 'ResNet101', 'VGG16']):
                data = pd.read_csv('./data/t-sne/{}.csv'.format(type))
                hdbscan_cluster_labels = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5,
                                                         gen_min_span_tree=True).fit(
                    np.array(data.iloc[:, 0:2])).labels_
                print(np.unique(hdbscan_cluster_labels))
                ax = fig.add_subplot(2, 2, i + 1)
                ax.scatter(
                    data.iloc[:, 0],
                    data.iloc[:, 1],
                    c=hdbscan_cluster_labels,  # 数据标签
                    cmap='viridis', s=3)
                ax.set_xlabel('({}) {}'.format(title_index[i], type), fontdict={'weight': 'normal', 'size': 12}, y=-0.1)
            plt.savefig('t_sne_2d.png', dpi=300)
            plt.show()

        if n_components == 3:
            fig = plt.figure(figsize=(10, 10))
            for i, type in enumerate(['AlexNet', 'GoogleNet', 'ResNet101', 'VGG16']):
                data = pd.read_csv('./data/t-sne/{}.csv'.format(type))
                hdbscan_cluster_labels = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5,
                                                         gen_min_span_tree=True).fit(
                    np.array(data.iloc[:, 2:5])).labels_
                print(np.unique(hdbscan_cluster_labels))
                ax = fig.add_subplot(2, 2, i + 1, projection='3d')
                ax.scatter(
                    data.iloc[:, 2],
                    data.iloc[:, 3],
                    data.iloc[:, 4],  # 三维数据
                    c=hdbscan_cluster_labels,  # 数据标签
                    cmap='viridis', s=3)

                ax.set_title('({}) {}'.format(title_index[i], type), fontdict={'weight': 'normal', 'size': 12}, y=-0.1)
            plt.savefig('t_sne_3d.png', dpi=300)
            plt.show()

    def feature_integration(self):
        print('Processing walking threshold data...')
        # POIs that significantly affect the housing prices (domain knowledge)
        selected_poi = ['MAL', 'SMK', 'KDG', 'PRS', 'SES', 'PAR', 'PLG', 'RGD', 'BUS', 'MIN', 'CPO', 'MTA']
        walk_thresholds = [300, 500]
        x_variables = ['CCL',
                       'Floor',
                       'Area',
                       'x',
                       'y',
                       'Longitude',
                       'Latitude',
                       'wifi_hk',
                       'POI_density',
                       'Num_class',
                       'Num_type',
                       'Class_diversity',
                       'Type_diversity'] + selected_poi

        final_data_samples = self.data[x_variables]

        for i, poi in enumerate(selected_poi):
            for walk_threshold in walk_thresholds:
                final_data_samples[poi + '_Walk{}'.format(walk_threshold)] = self.data.apply(
                    lambda x: 1 if x[poi] <= walk_threshold else 0, axis=1)

        xx_variables = ['CCL',
                        'Floor',
                        'Area',
                        'x',
                        'y',
                        'Longitude',
                        'Latitude',
                        'wifi_hk',
                        'POI_density',
                        'Num_class',
                        'Num_type',
                        'Class_diversity',
                        'Type_diversity']

        for poi in selected_poi:
            for walk_threshold in walk_thresholds:
                xx_variables.append(poi + '_Walk{}'.format(walk_threshold))

        final_data_samples = final_data_samples[xx_variables]

        print('Walking threshold data integration completed!')
        print('Processing google street view index data...')

        # Google street view index within distance thresholds
        view_distances = [500, 1000]
        for i, distance in enumerate(view_distances):
            print('View index data within {}m:'.format(distance))
            for row in tqdm(range(len(final_data_samples))):
                gsv_selected = self.GSV[(self.GSV['POINT_X'] <= final_data_samples.loc[row, 'x'] + distance) &
                                        (self.GSV['POINT_X'] >= final_data_samples.loc[row, 'x'] - distance) &
                                        (self.GSV['POINT_Y'] >= final_data_samples.loc[row, 'y'] - distance) &
                                        (self.GSV['POINT_Y'] <= final_data_samples.loc[
                                            row, 'y'] + distance)].reset_index(
                    drop=True)

                # gsv_selected['distance'] = gsv_selected.apply(lambda x: geodesic((final_data_samples.loc[row, 'Latitude'],
                #                                                                   final_data_samples.loc[row, 'Longitude']),
                #                                                                  (x['Latitude'],
                #                                                                   x['Longitude'])).m,
                #                                               axis=1)
                # gsv_points_distance = gsv_selected[gsv_selected['distance'] <= distance].reset_index(drop=True)

                gsv_points_distance = [x - 1 for x in gsv_selected['OBJECTID1'].values.tolist()]

                view_distance = self.view[self.view['img'].isin(gsv_points_distance)]

                final_data_samples.loc[row, 'sky{}'.format(distance)] = view_distance['sky'].mean()
                final_data_samples.loc[row, 'building{}'.format(distance)] = view_distance['building'].mean()
                final_data_samples.loc[row, 'vegetation{}'.format(distance)] = view_distance['vegetation'].mean()

        print('Google street view index data preparation completed!')

        # NDVI, NDBI, NDWI (within specific distance range, 200m, 500m, 1000m)
        print('Processing NDVI, NDBI, NDWI data...')
        for rs_distance in [500, 1000]:
            for rs_criteria in ['NDVI', 'NDWI', 'NDBI']:
                rs_data = pd.read_excel('collected data/image data/landsat/{}_{}.xls'.format(rs_criteria, rs_distance))
                final_data_samples[rs_criteria + str(rs_distance)] = rs_data['MEAN']

        print('NDVI, NDBI, NDWI data preparation completed!')

        # Dimension reduction from Estate Photos: 2D and 3D from GoogleNet, AlexNet, VGG-16, ResNet-101
        print('Processing estate photo data...')

        final_data_samples = pd.concat([final_data_samples, self.data['photo_id']], axis=1)

        for cnn_type in ['GoogleNet', 'AlexNet', 'VGG16', 'ResNet101']:
            result = Feature.estate_photo_feature(self, cnn_type=cnn_type)
            df = pd.DataFrame(result, columns=['{}_2d1'.format(cnn_type),
                                               '{}_2d2'.format(cnn_type),
                                               '{}_3d1'.format(cnn_type),
                                               '{}_3d2'.format(cnn_type),
                                               '{}_3d3'.format(cnn_type)])
            df.to_csv('./collected data/image data/estate/tnse/{}.csv'.format(cnn_type), index=False)

        GoogleNet = pd.read_csv('./collected data/image data/estate/tnse/{}.csv'.format('GoogleNet'))
        AlexNet = pd.read_csv('./collected data/image data/estate/tnse/{}.csv'.format('AlexNet'))
        VGG16 = pd.read_csv('./collected data/image data/estate/tnse/{}.csv'.format('VGG16'))
        ResNet101 = pd.read_csv('./collected data/image data/estate/tnse/{}.csv'.format('ResNet101'))

        for row in tqdm(range(len(final_data_samples))):
            for cnn_type in ['GoogleNet', 'AlexNet', 'VGG16', 'ResNet101']:
                if cnn_type == 'GoogleNet':
                    features = GoogleNet
                elif cnn_type == 'AlexNet':
                    features = AlexNet
                elif cnn_type == 'VGG16':
                    features = VGG16
                elif cnn_type == 'ResNet101':
                    features = ResNet101

                for dimension in ['2d1', '2d2', '3d1', '3d2', '3d3']:
                    photo_id = final_data_samples.loc[row, 'photo_id']
                    final_data_samples.loc[row, cnn_type + '_' + dimension] = features.loc[
                        photo_id, cnn_type + '_' + dimension]
        final_data_samples.drop(['photo_id'], axis=1, inplace=True)
        print('Estate photo data preparation completed!')

        # Add price data as y variable
        print('Add price data...')
        y_variable = ['Price']
        final_data_samples = pd.concat([final_data_samples, self.data[y_variable]], axis=1)

        # Delete row with Null value (view index column due to missing data)
        final_data_samples.dropna(inplace=True)

        print('Completed!')
        final_data_samples.to_csv('./collected data/clean data/paper.csv', index=False)


def main():
    data = pd.read_csv('./collected data/clean data/data_after_clean_xy_ccl.csv')
    facility = pd.read_csv('./collected data/GeoCom4.0_202203.csv')
    wifi = pd.read_csv('./collected data//WIFI_HK.csv')
    view = pd.read_csv('./paper data/data/surrounding/view_index.csv')
    gsv_points = pd.read_csv('./paper data/data/surrounding/GSV.csv')
    estate_img = './collected data/image data/estate/estate photos'

    res = Feature(data, view, gsv_points, facility, wifi, estate_img, n_jobs=-1)

    res.feature_integration()


if __name__ == "__main__":
    main()