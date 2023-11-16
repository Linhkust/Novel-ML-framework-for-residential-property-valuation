import pandas as pd
import datetime
from tqdm import tqdm
import requests
import time
from pyproj import Transformer


def data_clean():
    # read the data
    df = pd.read_csv('./collected data/raw data.csv')

    # Discard unwanted column
    unwanted_columns = ['Source.Name', 'District_html', 'Contract', 'Cat_ID', 'Arearaw', 'Deal_ID']
    df = df.drop(unwanted_columns, axis=1)

    # Delete data rows with empty values
    df.dropna(axis=0, how='any', inplace=True)
    df = df.drop(df[(df['GFA'] == '--') & (df['SFA'] == '--')].index)

    # Identify transaction date column for transformation
    df['Date'] = pd.to_datetime(df['Date'])

    # Inconsistent values: Delete the unit of GFA and SFA
    df['GFA'] = df['GFA'].apply(lambda x: x.replace('ft²', ''))
    df['SFA'] = df['SFA'].apply(lambda x: x.replace('ft²', ''))

    df['Area'] = df.apply(lambda x: x.GFA if x.GFA != '--' else x.SFA, axis=1)
    # df.dropna(subset=['Floor'], inplace=True)

    # Outliers in price: different units of price (HK$ and a million HK$)
    sorted_price = df['Price'].sort_values()  # sort the values as ascending order
    diff = sorted_price.diff()  # calculate the difference
    gap_point = sorted_price.loc[diff.idxmax()]  # find the data gap point
    df['Price'] = df['Price'].apply(lambda x: x / 1e6 if x >= gap_point else x)  # HK$ to Million HK$

    df = df.drop(['GFA', 'SFA', 'Change', 'GFA Price', 'SFA Price'], axis=1)
    df.to_csv('./collected data/clean data/data_after_clean.csv', index=False, encoding='utf_8_sig')


# Search the geographic coordinates and assign them to each record
def assign_geographic_coordinates():
    data = pd.read_csv('./collected data/clean data/data_after_clean.csv')

    for i in tqdm(range(len(data))):
        if data.loc[i, 'Block'] is None:
            address = data.loc[i, 'Estate']
        else:
            address = data.loc[i, 'Estate'] + ' ' + data.loc[i, 'Block']

        # Build the URL for retrieving Easting and Northing
        location_url = "https://geodata.gov.hk/gs/api/v1.0.0/locationSearch?q={}".format(address)
        response = requests.get(location_url)
        response = response.json()

        # Retrieve the x and y information
        x = response[0]['x']
        y = response[0]['y']

        # Add x and y to the dataset
        data.loc[i, 'x'] = x
        data.loc[i, 'y'] = y

        # Server rest
        time.sleep(2)

    # HKGrid1980 to WGS84
    tf = Transformer.from_crs('epsg:2326', 'epsg:4326', always_xy=True)
    # Add geographic coordinates
    data['Longitude'] = data.apply(lambda x: tf.transform(x['x'], x['y'])[0], axis=1)
    data['Latitude'] = data.apply(lambda x: tf.transform(x['x'], x['y'])[1], axis=1)
    data.to_csv('./collected data/clean data/data_after_clean_xy.csv', index=False)


# Assign Centa-leading city index to
def assign_CCL_index():
    data = pd.read_csv('./collected data/clean data/data_after_clean_xy.csv')

    hk = pd.read_excel('./collected data/ccl/CCL_HK.xlsx', engine='openpyxl')
    kl = pd.read_excel('./collected data/ccl/CCL_KL.xlsx', engine='openpyxl')
    ntw = pd.read_excel('./collected data/ccl/CCL_NTW.xlsx', engine='openpyxl')
    nte = pd.read_excel('./collected data/ccl/CCL_NTE.xlsx', engine='openpyxl')

    # Attach ccl index information to the data
    for row in range(len(data)):
        transaction_data = datetime.datetime.strptime(data.loc[row, 'Date'], '%d/%m/%Y')

        for j in range(len(hk)):
            start = datetime.datetime.strptime(hk.iloc[j, 0].split('-')[0].replace(' ', ''), '%Y/%m/%d')
            end = datetime.datetime.strptime(hk.iloc[j, 0].split('-')[1].replace(' ', ''), '%Y/%m/%d')
            if (transaction_data > start) & (transaction_data < end):
                index = j + 1
                break
        if data.loc[row, 'REGION'] == 'HK':
            data['CCL'] = hk.iloc[index, 1]
        elif data.loc[row, 'REGION'] == 'KL':
            data['CCL'] = kl.iloc[index, 1]
        elif data.loc[row, 'REGION'] == 'NTW':
            data['CCL'] = ntw.iloc[index, 1]
        else:
            data['CCL'] = nte.iloc[index, 1]
    data.to_csv('./collected data/clean data/data_after_clean_xy_ccl.csv', index=False)


'''
Street View Images
'''


# Main function
def main():
    data_clean()
    assign_geographic_coordinates()
    assign_CCL_index()


if __name__ == "__main__":
    main()