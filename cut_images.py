import pandas as pd
import numpy as np
import spectral.io.envi as envi
import skimage.draw
import time as time
from skimage import filters
import spectral as sp
import inspect
import matplotlib.pyplot as plt

plt.show(block=True)

def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno


def load_roi_file(roi):
    data = []
    sheet_names = []
    with open(roi) as f:
        content = f.readlines()
        for i in content:
            if "ROI name:" in i:
                sheet_names.append(i.split(":")[1].strip())
            else:
                data.append(i.split())

    return data


def read_roi(roi):
    data = load_roi_file(roi)

    # this list must comply with the same roi features in the file
    # group_lines = ['spectralon', 'v1', 'v2', 'v3', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10',
    #                'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'Black1', 'Black2']
    group_lines = ['spectralon', 'v1', 'v2', 'v3', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'Black1']

    # turn the ROI file to a DF, filter only to relevant rows
    df = pd.DataFrame(data, dtype=int)
    first_row = df.loc[df[1] == 'ID'].index[0]
    df.columns = df.loc[first_row]
    df = df.loc[first_row + 1:, :]
    df.fillna(value=np.nan, inplace=True)

    # keep only relevant columns
    df = df.iloc[:, 1:-5]

    # rename columns
    df.columns = df.columns.tolist()[1:3] + ['B1', 'B2', 'B3']
    df['group_id'] = df.isnull().all(axis=1).cumsum()  # give id to a group, a group is seperated by "nan" row
    df = df.dropna()

    df['X'] = df['X'].astype('int32')
    df['Y'] = df['Y'].astype('int32')
    df['coor'] = list(zip(df.X, df.Y))

    # start = time.time()
    df['group_id'] = df['group_id'].map(lambda x: group_lines[x])
    # end = time.time()
    # print(end - start)

    return df


def cut_images_in_fldr(D_image, roi, image_path, cal_path):
    print(D_image)
    print("line number", lineno())
    print(roi)
    print(image_path)

    # read the TXT ROI file
    df = read_roi(roi)

    """
    cut the images - slice the  image to individual plants
    """

    first_column = {}
    second_column = {}
    third_column = {}
    forth_column = {}

    # hor_lines = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15',
    #              'h16', 'h17', 'h18']
    hor_lines = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']
    ver_lines = ['v1', 'v2', 'v3']

    def create_poly_1st_col(curr_hor_line):
        top_L_x = df[df['group_id'] == hor_lines[curr_hor_line]]['X'].min()  # upper line first x
        top_L_y = df[df['group_id'] == hor_lines[curr_hor_line]][df['X'] == top_L_x]['Y'].values[0]  # upper line first y
        bot_L_x = df[df['group_id'] == hor_lines[curr_hor_line + 1]]['X'].min()  # lower line first x
        bot_L_y = df[df['group_id'] == hor_lines[curr_hor_line + 1]][df['X'] == bot_L_x]['Y'].values[0]  # lower line first y
        #left_vertical = np.array([[0, i] for i in range(top_L_y, bot_L_y + 1)])[1:-1]
        left_vertical = np.array([[0, i] for i in range(top_L_y + 1, bot_L_y)])

        a = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line]]['coor'], df[df['group_id'] == ver_lines[0]]['coor'])[0][0]  # intersection of upperline and v1 x
        c = df[df['group_id'] == hor_lines[curr_hor_line]]['coor'][::-1]  # upper line
        top_horizontal = np.array([[i, c.iloc[i][1]] for i in range(a + 1)])

        b = df[df['group_id'] == hor_lines[curr_hor_line + 1]]['coor'][::-1]  # lower line
        c = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line + 1]]['coor'],
                           df[df['group_id'] == ver_lines[0]]['coor'])  # intersection of lowerlnie and v1
        bottom_horizontal = np.array([[i, b.iloc[i][1] - 1] for i in range(c[0][0] + 1)])

        a = df[df['group_id'] == ver_lines[0]]  # vertical line1
        b = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line]]['coor'], df[df['group_id'] == ver_lines[0]]['coor'])  #
        c = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line + 1]]['coor'],
                           df[df['group_id'] == ver_lines[0]]['coor'])  #
        right_vertical = np.array([[a[a['Y'] == i]['X'].values[0], i] for i in range(b[0][1], c[0][1])])[1:]  #

        poly = np.concatenate((top_horizontal, left_vertical, bottom_horizontal, right_vertical))
        poly = poly[poly[:, 0].argsort()]
        poly1 = skimage.draw.polygon_perimeter(poly[:, 0], poly[:, 1])
        return poly1

    def create_poly_2nd_col(curr_hor_line):
        a = df[df['group_id'] == ver_lines[0]]
        b = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line]]['coor'], df[df['group_id'] == ver_lines[0]]['coor'])
        c = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line + 1]]['coor'], df[df['group_id'] == ver_lines[0]]['coor'])
        left_vertical = np.array([[a[a['Y'] == i]['X'].values[0], i] for i in range(b[0][1], c[0][1])])[1:]

        a = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line]]['coor'], df[df['group_id'] == ver_lines[0]]['coor'])[0][0]
        b = df[df['group_id'] == hor_lines[curr_hor_line]]
        d = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line]]['coor'], df[df['group_id'] == ver_lines[1]]['coor'])[0][0]
        top_horizontal = np.array([[i, b[b['X'] == i]['Y'].values[0]] for i in range(a, d)])

        a = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line + 1]]['coor'], df[df['group_id'] == ver_lines[0]]['coor'])[0][0]
        b = df[df['group_id'] == hor_lines[curr_hor_line + 1]]
        d = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line + 1]]['coor'], df[df['group_id'] == ver_lines[1]]['coor'])[0][0]
        bottom_horizontal = np.array([[i, b[b['X'] == i]['Y'].values[0]] for i in range(a, d)])

        a = df[df['group_id'] == ver_lines[1]]
        b = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line]]['coor'], df[df['group_id'] == ver_lines[1]]['coor'])
        c = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line + 1]]['coor'], df[df['group_id'] == ver_lines[1]]['coor'])
        right_vertical = np.array([[a[a['Y'] == i]['X'].values[0], i] for i in range(b[0][1], c[0][1])])[1:]

        poly = np.concatenate((top_horizontal, left_vertical, bottom_horizontal, right_vertical))
        poly = poly[poly[:, 0].argsort()]
        poly1 = skimage.draw.polygon_perimeter(poly[:, 0], poly[:, 1])
        return poly1

    def create_poly_3rd_col(curr_hor_line):
        a = df[df['group_id'] == ver_lines[1]]
        b = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line]]['coor'], df[df['group_id'] == ver_lines[1]]['coor'])
        c = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line + 1]]['coor'], df[df['group_id'] == ver_lines[1]]['coor'])
        left_vertical = np.array([[a[a['Y'] == i]['X'].values[0], i - 1] for i in range(b[0][1], c[0][1])])[1:]

        a = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line]]['coor'], df[df['group_id'] == ver_lines[1]]['coor'])[0][0]
        b = df[df['group_id'] == hor_lines[curr_hor_line]]
        d = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line]]['coor'], df[df['group_id'] == ver_lines[2]]['coor'])[0][0]
        top_horizontal = np.array([[i, b[b['X'] == i]['Y'].values[0]] for i in range(a, d)])

        a = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line + 1]]['coor'], df[df['group_id'] == ver_lines[1]]['coor'])[0][0]
        b = df[df['group_id'] == hor_lines[curr_hor_line + 1]]
        d = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line + 1]]['coor'], df[df['group_id'] == ver_lines[2]]['coor'])[0][0]
        bottom_horizontal = np.array([[i, b[b['X'] == i]['Y'].values[0] - 1] for i in range(a, d)])

        a = df[df['group_id'] == ver_lines[2]]
        b = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line]]['coor'], df[df['group_id'] == ver_lines[2]]['coor'])
        c = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line + 1]]['coor'], df[df['group_id'] == ver_lines[2]]['coor'])
        right_vertical = np.array([[a[a['Y'] == i]['X'].values[0], i - 1] for i in range(b[0][1], c[0][1])])[1:]
        poly = np.concatenate((top_horizontal, left_vertical, bottom_horizontal, right_vertical))
        poly = poly[poly[:, 0].argsort()]
        poly1 = skimage.draw.polygon_perimeter(poly[:, 0], poly[:, 1])
        return poly1

    def create_poly_4th_col(curr_hor_line):
        x_mina = df[df['group_id'] == hor_lines[curr_hor_line]]['X'].max()
        x_mina2 = df[df['group_id'] == hor_lines[curr_hor_line]][df['X'] == x_mina]['Y'].values[0]
        X_minb = df[df['group_id'] == hor_lines[curr_hor_line + 1]]['X'].max()
        x_minb2 = df[df['group_id'] == hor_lines[curr_hor_line + 1]][df['X'] == X_minb]['Y'].values[0]
        right_vertical = np.array([[1023, i] for i in range(x_mina2, x_minb2)])[1:-1]

        a = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line]]['coor'], df[df['group_id'] == ver_lines[2]]['coor'])[0][0]
        b = df[df['group_id'] == hor_lines[curr_hor_line]]['X'].max()
        c = df[df['group_id'] == hor_lines[curr_hor_line]][::-1]
        top_horizontal = np.array([[i, c[c['X'] == i]['Y'].values[0]] for i in range(a, b)] +
                                  [[i, c[c['X'] == b]['Y'].values[0]] for i in range(b, 1024)])

        a = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line + 1]]['coor'], df[df['group_id'] == ver_lines[2]]['coor'])[0][0]
        b = df[df['group_id'] == hor_lines[curr_hor_line + 1]]['X'].max()
        c = df[df['group_id'] == hor_lines[curr_hor_line + 1]][::-1]
        bottom_horizontal = np.array([[i, c[c['X'] == i]['Y'].values[0] - 1] for i in range(a, b)] +
                                     [[i, c[c['X'] == b]['Y'].values[0] - 1] for i in range(b, 1024)])

        a = df[df['group_id'] == ver_lines[2]]
        b = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line]]['coor'], df[df['group_id'] == ver_lines[2]]['coor'])
        c = np.intersect1d(df[df['group_id'] == hor_lines[curr_hor_line + 1]]['coor'], df[df['group_id'] == ver_lines[2]]['coor'])
        left_vertical = np.array([[a[a['Y'] == i]['X'].values[0], i] for i in range(b[0][1], c[0][1])])[1:]
        poly = np.concatenate((top_horizontal, left_vertical, bottom_horizontal, right_vertical))
        poly = poly[poly[:, 0].argsort()]
        poly1 = skimage.draw.polygon_perimeter(poly[:, 0], poly[:, 1])
        return poly1


    # create the polygones around each plant
    for curr_h_line in range(len(hor_lines) - 1):
        print(curr_h_line)

        first_column[curr_h_line] = create_poly_1st_col(curr_h_line)
        second_column[curr_h_line] = create_poly_2nd_col(curr_h_line)
        third_column[curr_h_line] = create_poly_3rd_col(curr_h_line)
        forth_column[curr_h_line] = create_poly_4th_col(curr_h_line)

    # names
    first_column_names = ['1A', '2A', '3A', '4A', '5A', '6A', '7A', '8A', '9A', '10A', '11A', '12A', '13A', '14A',
                          '15A',
                          '16A', '17A', '18A']
    forth_column_names = ['1D', '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', '10D', '11D', '12D', '13D', '14D',
                          '15D',
                          '16D', '17D', '18D']
    second_column_names = ['1B', '2B', '3B', '4B', '5B', '6B', '7B', '8B', '9B', '10B', '11B', '12B', '13B', '14B',
                           '15B',
                           '16B', '17B', '18B']
    third_column_names = ['1C', '2C', '3C', '4C', '5C', '6C', '7C', '8C', '9C', '10C', '11C', '12C', '13C', '14C',
                          '15C',
                          '16C', '17C', '18C']

    # get the keys of the columns
    first_keys = list(first_column.keys())
    second_keys = list(second_column.keys())
    third_keys = list(third_column.keys())
    forth_keys = list(forth_column.keys())

        # calibrate the image
    image = envi.open(image_path).load().load()
    cal_img = envi.open(cal_path + '.hdr', image=cal_path + '.cal').load()
    dark_img = envi.open(D_image).load().mean(axis=0)
    print("loaded images line number", lineno())
    # get the integration time
    time_integration = float(image.metadata['tint'][:2])
    rad_image = np.empty_like(image)
    for i in range(image.shape[0]):
        rad_image[i] = np.divide(np.multiply(np.subtract(image[i], dark_img), cal_img), time_integration)

    # calculate reflectance
    # white panel position
    ref_x_start = df[df['group_id'] == 'spectralon']['X'].min()
    ref_x_stop = df[df['group_id'] == 'spectralon']['X'].max()
    ref_y_start = df[df['group_id'] == 'spectralon']['Y'].min()
    ref_y_stop = df[df['group_id'] == 'spectralon']['Y'].max()
    # white panel mean spectrum
    spectralon = rad_image[ref_y_start:ref_y_stop, ref_x_start:ref_x_stop]
    spectralon_avg = spectralon.mean(axis=(0, 1))
    # black panel position
    black_x_start = df[df['group_id'] == 'Black1']['X'].min()
    black_x_stop = df[df['group_id'] == 'Black1']['X'].max()
    black_y_start = df[df['group_id'] == 'Black1']['Y'].min()
    black_y_stop = df[df['group_id'] == 'Black1']['Y'].max()

    # black panel mean spectrum
    black = rad_image[black_y_start:black_y_stop, black_x_start:black_x_stop]
    black_avg = black.mean(axis=(0, 1))

    refle_img = np.divide((np.subtract(rad_image, black_avg)), (np.subtract(spectralon_avg, black_avg)))

    RED_CHANNEL = 210
    NIR_CHANNEL = 266
    red = refle_img[:, :, RED_CHANNEL]
    red_wave = float(image.metadata['wavelength'][RED_CHANNEL])
    nir = refle_img[:, :, NIR_CHANNEL]
    nir_wave = float(image.metadata['wavelength'][NIR_CHANNEL])
    diff = np.subtract(nir, red) / (nir_wave - red_wave)

    print("calculated diff image", lineno())

    # """new addition"""
    # sp.imshow(diff[:, :], stretch=(0.2, 0.98))
    # for key1 in first_column.keys():
    #     plt.plot(first_column[key1][0], first_column[key1][1], linewidth=4, alpha=0.3)
    # for key1 in second_column.keys():
    #     plt.plot(second_column[key1][0], second_column[key1][1], linewidth=4, alpha=0.3)
    # for key1 in third_column.keys():
    #     plt.plot(third_column[key1][0], third_column[key1][1], linewidth=4, alpha=0.3)
    # for key1 in forth_column.keys():
    #     plt.plot(forth_column[key1][0], forth_column[key1][1], linewidth=4, alpha=0.3)
    # plt.show()
    # """finish new addition"""

    def col_imgs2sql(rowcol, names, segmented_dict):  # rowcol = keys of segmented images, names = name of plant
        df_collector = {}

        for key1, name in zip(rowcol, names):
            print(key1, name)
            print('begining img2sql', time.strftime("%H:%M:%S"))
            loop_DIFF = diff[segmented_dict[key1][1], segmented_dict[key1][0]] # only diff pixels in inside the polygon
            thresh = filters.threshold_otsu(loop_DIFF) # find first threshold (background vs plant)
            loop_DIFF[np.where(loop_DIFF < thresh)] = thresh #insert threshold value into all none plant pixels
            thresh = filters.threshold_otsu(loop_DIFF) # find second threshold (plant vs edges)
            plant_idx = np.where(loop_DIFF > thresh) #local plant index
            print('finished both otsu thresholding', time.strftime("%H:%M:%S"))
            # find out original index instead of a and b
            # get time and date from file name
            date = image.metadata['acquisition date'].split(' ')[1]
            time_aquision = image.metadata['start time'].split(' ')[2]
            print('extracted date and time', time.strftime("%H:%M:%S"))

            plant_info = {'name': name, 'date': date, 'time': time_aquision, 'coordX': segmented_dict[key1][0][plant_idx[0]], 'coordY': segmented_dict[key1][1][plant_idx[0]] }
            plant_info_df = pd.DataFrame.from_dict(plant_info)
            print('converted to original index', time.strftime("%H:%M:%S"))

            plant_info_df['type'] = 'raw'
            plant_pixels = image[segmented_dict[key1][1], segmented_dict[key1][0]][plant_idx[0]]
            plant_pixels_spectra = pd.DataFrame(columns=image.metadata['wavelength'], data=plant_pixels.tolist())
            spectra_type_df = pd.concat([plant_info_df, plant_pixels_spectra], axis=1)
            df_collector[name] = spectra_type_df
            print('created raw df', time.strftime("%H:%M:%S"))

            plant_pixels = rad_image[segmented_dict[key1][1], segmented_dict[key1][0]][plant_idx[0]]
            plant_info_df['type'] = 'rad'
            plant_pixels_spectra = pd.DataFrame(columns=image.metadata['wavelength'], data=plant_pixels.tolist())
            spectra_type_df = pd.concat([plant_info_df, plant_pixels_spectra], axis=1)
            df_collector[name] = df_collector[name].append(spectra_type_df)
            print('created rad df', time.strftime("%H:%M:%S"))

            plant_pixels = refle_img[segmented_dict[key1][1], segmented_dict[key1][0]][plant_idx[0]]
            plant_info_df['type'] = 'ref'
            plant_pixels_spectra = pd.DataFrame(columns=image.metadata['wavelength'], data=plant_pixels.tolist())
            spectra_type_df = pd.concat([plant_info_df, plant_pixels_spectra], axis=1)
            df_collector[name] = df_collector[name].append(spectra_type_df)
            print('created ref df', time.strftime("%H:%M:%S"))

        col_dfx = pd.concat([df for df in df_collector.values()], ignore_index=True)

        return col_dfx

    def calibration_sql(name, data):
        date = image.metadata['acquisition date'].split(' ')[1]
        time_aquision = image.metadata['start time'].split(' ')[2]
        plant_info = {'name': name, 'date': date, 'time': time_aquision, 'type': 'rad'}
        temp_dict2 = pd.DataFrame(plant_info, index=[0])
        temp_dict3 = pd.DataFrame(data.tolist(), image.metadata['wavelength'])
        column_df = pd.concat([temp_dict2, temp_dict3.T], axis=1, sort=False)
        return column_df

    first_column_df = col_imgs2sql(first_keys, first_column_names, first_column)
    second_column_df = col_imgs2sql(second_keys, second_column_names, second_column)
    third_column_df = col_imgs2sql(third_keys, third_column_names, third_column)
    forth_column_df = col_imgs2sql(forth_keys, forth_column_names, forth_column)
    spectralon_df = calibration_sql('spectralon', spectralon_avg)
    black_df = calibration_sql('black', black_avg)
    # stack everything
    frames = [first_column_df, second_column_df, third_column_df, forth_column_df, spectralon_df, black_df]
    all_df = pd.concat(frames, sort=False)

    df_float = all_df.select_dtypes(include=['float'])
    df_float_converted = df_float.apply(pd.to_numeric, downcast='float')
    type_df = all_df['type'].astype('category')
    name_df = all_df['name'].astype('category')
    date_df = pd.to_datetime(all_df['date'])

    df2 = pd.concat((name_df, date_df, type_df, all_df['time'], all_df['coordX'], all_df['coordY'], df_float_converted), axis=1,
                    sort=False)

    # pickle uses lots of memory - so lets delete some vars
    del df_float
    del df_float_converted
    del type_df
    del name_df
    del date_df
    del all_df
    del frames
    del first_column_df
    del second_column_df
    del third_column_df
    del forth_column_df

    df2.to_pickle(roi[:-7] + 'pickle',protocol = 4)


#    df2.to_csv(ROI[:-7] + 'csv')
# all_df.to_csv(r'N:\Shahar\test automatic method\pickle4.csv', sep=',')
# all_df.to_hdf(r'N:\Shahar\test automatic method\pickle3.h5', key='all_df', mode='w')


# The files - either recrusive or per image 
# recrusive
# for i in range (22,23,1):
#    path = r'\\remotesensinglabfs.tau.ac.il\REMOTESENSINGLAB$\remotesensinglab3\Shahar\new chamama\data2\2018-09-' + str(i)
#    DARKS = [file for file in glob.glob(path + '\*\*\DARKREF' +'*.hdr', recursive=True)]
#    ROIS = [file for file in glob.glob(path + '\*\ROI.txt', recursive=True)]
#    IMAGES = [file for file in glob.glob(path + '\*\*\*' + 'emptyname' +'*.hdr', recursive=True) if not os.path.basename(file).startswith(('DARKREF','RADIANCE'))]
# for DARK,ROI,IMAGE1 in zip(DARKS,ROIS,IMAGES):
#    cut_images_in_fldr(DARK,ROI,IMAGE1)

# per image
TEST_ROOT = r'C:\Learning from benny'
CAL_FILE = r'C:\Learning from benny\Radiometric_1x1'
# TEST_DIR = TEST_ROOT + r'\emptyname_2018-09-20_07-26-11'
IMAGE = TEST_ROOT + r'\subset.hdr'
DARK_IMAGE = TEST_ROOT + r'\DARKREF_emptyname_2018-09-13_07-05-57.hdr'
ROI = TEST_ROOT + r'\ROI2.txt'

overall_start = time.time()
cut_images_in_fldr(DARK_IMAGE, ROI, IMAGE, CAL_FILE)
print('overall duration:', time.time()-overall_start, 's')
