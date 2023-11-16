from io import BytesIO
import pandas as pd
import requests
from tqdm import tqdm
import time
import threading
from fake_useragent import UserAgent
from bs4 import BeautifulSoup as bs
import json
from functools import partial
from PIL import ImageChops
import itertools
import glob
import os
from PIL import Image


class SVI(object):

    def __init__(self, data, api_key, date):
        self.data = data
        self.api_key = api_key
        self.date = date

    "Google Street View Images"

    def gsv(self, row):
        headings = [0, 90, 180, 270]
        headings_name = ['front', 'right', 'rear', 'left']
        for i, heading in enumerate(headings):
            url = 'https://maps.googleapis.com/maps/api/streetview?' \
                  'size=640x360' \
                  '&location={},{}' \
                  '&heading={}' \
                  '&fov=90' \
                  '&pitch=0' \
                  'date={}' \
                  '&key={}'.format(self.data.loc[row, 'Latitude'],
                                   self.data.loc[row, 'Longitude'],
                                   heading,
                                   self.date,
                                   self.api_key)

            req = requests.get(url)
            with open('./paper data/gsv images/gsv/{}_{}.jpg'.format(row, headings_name[i]), 'wb') as f:
                f.write(req.content)

    def gsv_panorama(self, row):
        url = 'https://maps.googleapis.com/maps/api/streetview/metadata?' \
              'size=640x360' \
              '&location={},{}' \
              '&key={}'.format(self.data.loc[row, 'Latitude'],
                               self.data.loc[row, 'Longitude'],
                               self.api_key)

        js = json.loads(requests.get(url).text)
        return js['pano_id']

    def tiles_save(self, x, y, pano_id):
        # REAL URL FOR TILE DOWNLOAD
        image_url = "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={}&x={}&y={}&zoom=5".format(
            pano_id, x, y)
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            img.save('./paper data/gsv images/tiles/{}_{}.png'.format(x, y))
        except requests.exceptions.RequestException as e:
            print('Error downloading tile ({}, {}): {}'.format(x, y, e))

    # Use threading to speed up image collection process (I/O)
    def multi_thread_tiles_save(self, pano_id):
        # The tiles positions
        coords = list(itertools.product(range(26), range(13)))

        partial_fun = partial(self.tiles_save, pano_id=pano_id)

        threads = []
        i = 0
        while i < 26:

            for j in range(13):
                t = threading.Thread(target=partial_fun, args=(coords[13 * i + j][0], coords[13 * i + j][1]))
                threads.append(t)

            for t in threads:
                time.sleep(0.01)
                t.start()

            for t in threads:
                t.join()

            threads.clear()

            i += 1

    def tiles_stitch(self):
        coords = list(itertools.product(range(26), range(13)))
        panorama = Image.new('RGB', size=(13312, 6656))

        for x, y in coords:
            img = Image.open('./collected data/image data/tiles/{}_{}.png'.format(x, y))
            panorama.paste(img, box=(x * 512, y * 512))

        # Resize the image to reduce the resolution, save disk space
        panorama = panorama.resize((int(1664), int(832)), Image.ANTIALIAS)
        return panorama

    def gsv_panorama_download(self, row):
        try:
            pano_id = SVI.gsv_panorama(self, row)

            # Tile save
            SVI.multi_thread_tiles_save(self, pano_id)

            # Tiles stitch
            pano = SVI.tiles_stitch(self)

            pano.save('./collected data/image data/panorama/{}.png'.format(row))

        except Exception:
            print('Server Connection Failed')
            time.sleep(2)

    # Compare the panorama images to detect repeated images
    # Find out the index of missing and repeated panoramas. Download the corresponding GSV single-view images
    def compare_panorama_download(self):
        for row in tqdm(range(len(self.data))):
            try:
                file1 = Image.open('./collected data/image data/panorama/{}.png'.format(row))
                file2 = Image.open('./collected data/image data/panorama/{}.png'.format(row+1))
                diff = ImageChops.difference(file1, file2).getbbox()
                if diff is None:
                    SVI.gsv(self, row+1)
            except FileNotFoundError:
                continue


"Estate photos from 28Hse website"


class Estate(object):
    def estate_photo_retrieval(self):
        # total pages of https://www.28hse.com/estate/
        for page in range(1, 624):
            path = 'https://www.28hse.com/en/estate/dosearch'
            payload = {"form_data": "page={}&searchText=" \
                                    "&myfav=" \
                                    "&myvisited=" \
                                    "&item_ids=" \
                                    "&sortBy=popular" \
                                    "&is_grid_mode=" \
                                    "&district=" \
                                    "&district_by_text=0" \
                                    "&type=&type_by_text=0" \
                                    "&landreg_sqprice=" \
                                    "&landreg_sqprice_by_text=0" \
                                    "&estate_age=" \
                                    "&estate_age_by_text=0".format(page)}  # what action you want the server to perform

            headers = {
                'connection': 'close',
                'Accept': 'application/json, text/javascript, */*; q=0.01',  # accept json and text response only
                'Accept-Encoding': 'gzip, deflate, br',  # changed to gzip only as others return me garbled codes
                'Referer': 'https://www.28hse.com/en/estate/',
                'User-Agent': UserAgent().random  # use random UA to obfuscate the server
            }
            try:
                r = requests.post(path, headers=headers, data=payload,
                                  verify=False)  # request the website to update the table
                js = json.loads(r.text)
                data = js['data']['results']['html']  # return the html part of the data received
                soup = bs(data, 'html.parser')  # use html.parser to parse the website response
                links = soup.find_all('div', attrs={'class': 'item enter_detailpage cursor_pointer'})

            except Exception:
                time.sleep(30)
                r = requests.post(path, headers=headers, data=payload,
                                  verify=False)  # request the website to update the table
                js = json.loads(r.text)
                data = js['data']['results']['html']  # return the html part of the data received
                soup = bs(data, 'html.parser')  # use html.parser to parse the website response
                links = soup.find_all('div', attrs={'class': 'item enter_detailpage cursor_pointer'})

            for j, link in enumerate(links):
                estate_name = link.find('a').text

                # Detail_link get the info
                detail_link = link['href']
                try:
                    detail_response = requests.get(detail_link, verify=False)
                    estate_url = bs(detail_response.text, 'html.parser')
                    community_photo_links = []
                    community_photos = estate_url.find_all('a', attrs={
                        'class': 'ui centered bordered large image estate_large_image'})

                    # Only need the building photos by setting category_id == 1
                    for community_photo in community_photos:
                        if community_photo.get('category_id') == '1':
                            building_photo = community_photo.get('href')
                            community_photo_links.append(building_photo)
                        else:
                            break
                    estate_info_photos = {'estate name': estate_name, 'detail link': detail_link,
                                          'photo links': community_photo_links}
                    estate_info_photos = pd.DataFrame(estate_info_photos)
                    estate_info_photos.to_csv(
                        '..\\collected data\\image data\\estate\\estate_photos_link\\page_{}_{}_{}.csv'.format(page, j, estate_name), index=False)
                except Exception:
                    continue

                time.sleep(5)

            time.sleep(5)

    # website photos combine
    def combine_csv(self):
        path = '../collected data/image data/estate/estate_photos_link'  # directory
        csv_files = glob.glob(os.path.join(path, "*.csv"))

        # real all csv files and combine
        temp = pd.DataFrame()
        for f in csv_files:
            df = pd.read_csv(f, header=0, encoding='utf-8')
            if df.empty is False:
                temp = pd.concat([temp, df])
            else:
                pass

        # 将DataFrame写入到一个CSV文件中
        temp.to_csv("./collected data/image data/estate/estate photos.csv", index=False, encoding='utf-8')

    # find the unique property in our dataset and find corresponding estate link
    def estate_photo_link(self):
        paper_data = pd.read_csv('./paper data/data/data/data_features.csv', encoding='unicode_escape')
        estate_photo = pd.read_csv('./paper data/data/estate photos.csv')
        unique_values = paper_data[['Estate', 'Estate_html']].drop_duplicates().reset_index()
        estate_photo_link = []
        for i in tqdm(range(len(unique_values))):
            estate_name = unique_values.loc[i, 'Estate']
            property_link = unique_values.loc[i, 'Estate_html']
            estate_id = property_link[(property_link.rfind('_') + 1):]  # 831_convention-plaza
            new_estate_id = estate_id[estate_id.find('_') + 1:]  # convention-plaza-831
            try:
                search_results = estate_photo[estate_photo['detail link'].str.contains(new_estate_id)].reset_index()
                estate_photo_link.append([estate_name, property_link, search_results.loc[0, 'photo links']])
            except Exception:
                estate_photo_link.append([estate_name, property_link, ''])

        estate_photo_link = pd.DataFrame(estate_photo_link, columns=['Name', 'Estate', 'Photo'])
        estate_photo_link.to_csv('./collected data/image data/estate/estate_photo_links.csv', index=False)

    def estate_photo_download(self, row):
        estate_photo_link = pd.read_csv('./collected data/image data/estate/estate_photo_links.csv')
        name = estate_photo_link.iloc[row, 0]
        url = estate_photo_link.iloc[row, 2]
        headers = {
            'connection': 'close',
            'Accept': 'application/json, text/javascript, */*; q=0.01',  # accept json and text response only
            'Accept-Encoding': 'gzip, deflate, br',  # changed to gzip only as others return me garbled codes
            'Referer': 'https://www.28hse.com/en/estate/',
            'User-Agent': UserAgent().random  # use random UA to obfuscate the server
        }
        response = requests.get(url, headers=headers)
        # 确认响应状态码为200

        if response.status_code == 200:
            # 用二进制打开文件并写入响应内容
            with open("./collected data/image data/estate/estate photos/{}_{}.jpg".format(row, name), "wb") as f:
                f.write(response.content)
                print("图片下载成功！")
        else:
            print("下载失败：HTTP状态码为", response.status_code)
        time.sleep(3)

    # Detect unsuccessfully downloaded estate photos and manually down the photos from other real estate websites
    def estate_photo_download_update(self):
        data = pd.read_csv('./collected data/image data/estate/estate_photo_links.csv')
        list = []
        for row in range(len(data)):
            print(row)
            name = data.loc[row, 'Name']
            if os.path.exists('./paper data/estate photos/{}_{}.jpg'.format(row, name)):
                continue
            else:
                try:
                    Estate.estate_photo_download(self, row)
                    time.sleep(3)
                except Exception:
                    list.append([row, name])
        print(list)


# main function
def main():
    sampling_points = pd.read_csv('./collected data/image data/GSV.csv')
    api_key = 'Your APIKey'
    date = '2021-07'
    res = SVI(data=sampling_points, api_key=api_key, date=date)

    for i in range(len(sampling_points)):
        res.gsv_panorama_download(i)


if __name__ == "__main__":
    main()