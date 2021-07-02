import os

def classdef(name):
    if name == '480x270':
        return '0'
    if name == '480x540':
        return '1'
    if name == '480x810':
        return '2'
    if name == '960x270':
        return '3'
    if name == '960x540':
        return '4'
    if name == '960x810':
        return '5'
    if name == '1440x270':
        return '6'
    if name == '1440x540':
        return '7'
    if name == '1440x810':
        return '8'


# label the images
def labelling(src, dst, name):
    # read the file name under the folder
    img_list = os.listdir(src)
    # txt name
    txt_path = dst + '/' + name + '.txt'
    for img in img_list:
        srcPath = src + "/" + img
        with open(txt_path, 'a', encoding='utf-8') as dstFile:
            # src = '../DataSet/300x150'
            src_list = src.split('/')
            class_name = src_list[-1]
            class_type = classdef(class_name)
            text = srcPath
            dstFile.write(text + ' ' + class_type)
            dstFile.write('\n')
            dstFile.close()


# main function
if __name__ == '__main__':
    Path = "F:/Gaze_Estimation/GazeNet_end2end/Dataset/second_dataset/test"
    dstPath = "../label/second_dataset"

    # create a new one if it's not existing
    try:
        os.mkdir(dstPath)
        print(dstPath + ' build success!')
    except:
        print("The folder already exists, no need to create!")

    # read all kinds of folders in the data set folder
    dir_list = os.listdir(Path)
    # stitch the names of each folder
    for dir_name in dir_list:
        # find the folders of each category
        srcPath = Path + "/" + dir_name
        # multi-process, while reading the content of the file in the first folder, it is also reading the content of the files in other folders
        labelling(srcPath, dstPath, dir_name)
