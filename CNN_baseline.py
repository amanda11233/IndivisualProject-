import numpy as np
import matplotlib.pyplot as plt
import glob
import math
from PIL import Image
import statsmodels.api as sm
import pandas as pd
from yahoo_fin import stock_info as si

def r_squared(y_true, y_hat):
    ssr = 0
    sst = 0
    e = np.subtract(y_true, y_hat)
    y_mean = np.mean(y_true)
    for item in e:
        ssr += item**2
    for item in y_true:
        sst += (item - y_mean)**2
    r2 = 1 - ssr / sst
    return r2


def data_process(data):
    processed_data = []
    for item in data:
        m = np.mean(item)
        s = np.std(item)
        normal_item = [(float(i)-m)/s for i in item]
        normal_item.insert(0, 1)
        processed_data.append(normal_item)
    return processed_data


def get_pixel_values():
    file_name = r'./figures'
    pixels = []
    for filename in glob.glob(file_name + '/*.png'):
        im = Image.open(filename)
        temp_pixels = list(im.getdata())
        pixels.append(temp_pixels)
    return pixels


def find_returns(data):
    returns = []
    for group in data:
        count = 30
        while count <= (len(group)-5):
            current_data = group[count-1]
            future_data = group[count+4]
            p1 = np.mean(current_data)
            p2 = np.mean(future_data)
            returns.append(math.log(p2/p1))
            count += 1
    return returns


def convert_image():
    size = 54, 32
    file_name = r'./figures'
    for filename in glob.glob(file_name + '/*.png'):
        img = Image.open(filename)
        img.thumbnail(size)
        img = img.convert('L')
        img.save(filename)


def plot_data(data):
    t = np.arange(0, 29, 1)
    file_name_number = 0
    fig = plt.figure(frameon=False)
    for group in data:
        count = 30
        while count <= (len(group)-5):
            high = []
            low = []
            for item in group[count-30:count]:
                high.append(item[0])
                low.append(item[1])
            file_name = r'/fig_' + str(file_name_number)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.plot(t, high[0:-1], 'b', t, low[0:-1], 'g')
            fig.savefig(r'./figures' + file_name)
            fig.clf()
            file_name_number += 1
            count += 1
    print('Created %d files!' % file_name_number)


def extract(ticker):
    if (isinstance(ticker, str)):
        data = si.get_data(ticker)

    elif isinstance(ticker, pd.DataFrame):
        data = ticker

    else:
        raise TypeError("ticker can be either a str or a `pd.Dataframe`")
    df = pd.DataFrame(data)
    temp_buffer = []
    groups = []
    for high, low in zip(df["high"].values, df["low"].values):
        temp = [high, low]
        temp = [float(i) for i in temp]
        temp_buffer.append(temp)

    groups.append(temp_buffer)
    return groups


def main():
    ticker = "TSLA"
    # plot_data(extract(ticker))
    # convert_image()
    returns = np.asarray(find_returns(extract(ticker)))
    training_data = np.asarray(get_pixel_values())
    training_data = sm.add_constant(training_data, has_constant='add')
    results = sm.OLS(returns[0:1340], training_data[0:1340]).fit()
    y_in_sample = results.predict(training_data[0:1340])
    r2 = r_squared(returns[0:1340], y_in_sample)
    print(r2)



if __name__ == "__main__":
    main()