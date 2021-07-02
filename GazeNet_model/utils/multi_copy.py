import  os
import cv2
import  multiprocessing

# copy function
def copy(src, dst, name):
    # first, you need to find the target file, so you need to splice
    src_file = src + "/" + name
    dst_file = dst + "/" + name
    # read the image list
    img_list = os.listdir(src_file)
    for img in img_list:
        src_Path = src_file + '/' + img
        dst_Path = dst_file + '/' + img
        src_img = cv2.imread(src_Path)
        cv2.imwrite(dst_Path, src_img)



# main
if __name__ == '__main__':
    srcPath = "../expand"
    dstPath = "../Dataset/first_dataset/train"

    # create a new if not exists
    try:
        os.mkdir(dstPath)
    except:
        print("The folder already exists, no need to create!")

    # read files in source folder
    file_list = os.listdir(srcPath)
    # call copy function
    for file_name in file_list:
        # use multiple processes, while copying the first one, other files are also being copied
        copy_process = multiprocessing.Process(target = copy, args = (srcPath, dstPath, file_name))
        copy_process.start()
