
import numpy as np
import cv2
import os
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def image_energy(image):
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    energy_img = np.abs(dx) + np.abs(dy)
    return energy_img

def v_cum_min_energy(energy_img):
    r,c = energy_img.shape
    energy_sum = energy_img.copy()
    for i in range(1,r):
        for j in range(c):
            if j==0:
                energy_sum[i,j] = energy_sum[i,j]+min(energy_sum[i-1,j], energy_sum[i-1,j+1])
            elif j== c-1:
                energy_sum[i,j] = energy_sum[i,j]+min(energy_sum[i-1,j-1], energy_sum[i-1,j])
            else:
                energy_sum[i, j] = energy_sum[i, j] + min(energy_sum[i - 1, j - 1], energy_sum[i - 1, j], energy_sum[i - 1, j + 1])
    return energy_sum

def find_seam(energy_sum):
    r, c = energy_sum.shape
    seam = []
    min_idx = np.argmin(energy_sum[r-1,:])
    seam.append([r-1,min_idx])
    for i in range(r-2,-1,-1):
        if min_idx ==0:
            min_idx = np.argmin(energy_sum[i,min_idx:min_idx+2])+min_idx
        elif min_idx==c:
            min_idx = np.argmin(energy_sum[i, min_idx-1:min_idx +1])+1+min_idx
        else:
            min_idx = np.argmin(energy_sum[i, min_idx-1:min_idx+2])-1+min_idx
        seam.append([i,min_idx])
    return seam

def remove_seam(image, seam):
    m, n, b = image.shape
    reduced_img = np.zeros((m,n-1,b), np.uint8)
    for r, c in reversed(seam):
        reduced_img[r,:c] = image[r,:c]
        reduced_img[r,c:] = image[r,c+1:]
    return reduced_img

def seam_carve(image, height, width):
    r,c,b = image.shape
    out_image = image
    if c>width:
        for i in range(c-width):
            img_energy = image_energy(out_image)
            vert_cum_eng = v_cum_min_energy(img_energy)
            seam = find_seam(vert_cum_eng)
            out_image = remove_seam(out_image, seam)
    if r>height:
        out_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        for j in range(r-height):
            img_energy = image_energy(out_image)
            vert_cum_eng = v_cum_min_energy(img_energy)
            seam = find_seam(vert_cum_eng)
            out_image = remove_seam(out_image,seam)
        out_image = cv2.rotate(out_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return out_image

def add_seam(image, seam, averaging=False):
    m, n, b = image.shape
    expanded_img = np.zeros((m,n+1,b), np.uint8)
    for r, c in reversed(seam):
        if averaging==True:
            if c==0:
                averaged_seam = np.mean([image[r, c], image[r, c + 1]])
            elif c==n-1:
                averaged_seam = np.mean([image[r, c], image[r, c - 1]])
            else:
                averaged_seam = np.mean([image[r,c],image[r,c-1],image[r,c+1]])
            expanded_img[r,:c] = image[r,:c]
            expanded_img[r,c] = averaged_seam
            expanded_img[r,c+1:] = image[r,c:]
        elif averaging==False:
            expanded_img[r,:c] = image[r,:c]
            expanded_img[r,c] = image[r,c]#averaged_seam #[0,0,255]
            expanded_img[r,c+1:] = image[r,c:]
    return expanded_img

def seam_insertion(image, height, width, averaging=False):
    r,c,b = image.shape
    out_img = image
    carve_img = image
    all_seams = []
    if (c<width):
        for i in range(width-c): #outputs list of seams in order they were removed
            img_energy = image_energy(carve_img)
            vert_cum_eng = v_cum_min_energy(img_energy)
            seam = find_seam(vert_cum_eng)
            all_seams.append(seam)
            carve_img = remove_seam(carve_img, seam)
        for i in range(len(all_seams)): #for each seam, take it, add it, and update the rest of the indices
            curr_seam = all_seams.pop(0)
            out_img = add_seam(out_img,curr_seam,averaging=averaging)
            #update seams
            flag=0
            for each_seam in (all_seams):
                for j in range(len(each_seam)):
                    if each_seam[j][1] >= curr_seam[j][1]:
                        flag = 1
                        break
                if flag==1:
                    for j in range(len(each_seam)):
                        each_seam[j][1] +=2
                    flag=0
    return out_img

def tmap(image, f_img_r, f_img_c):
    r,c,b = image.shape
    # m_p = r-height
    # n_p  = c-width
    t_map = np.zeros((r,c))
    cache = [None]*c
    bitmap = np.zeros((r,c))
    out_img = None
    out_dim_r = r-f_img_r
    out_dim_c = c-f_img_c
#i = horiz seam removal, j= vert seam removal
    for i in range(r):
        for j in range(c):
            if i==0 and j==0:
                cache[j] = image
            elif i==0 and j!=0:
                image_left = cache[j-1]
                energy = image_energy(image_left)
                v_cum_eng = v_cum_min_energy(energy)
                seam = find_seam(v_cum_eng)
                t_map[i,j] = v_cum_eng[-1,seam[0][1]]+t_map[i,j-1]
                cache[j] = remove_seam(image_left,seam)
            elif i!=0 and j==0:
                image_top = cv2.rotate(cache[j], cv2.ROTATE_90_CLOCKWISE)
                energy = image_energy(image_top)
                h_cum_eng = v_cum_min_energy(energy)
                seam = find_seam(h_cum_eng)
                t_map[i,j] = h_cum_eng[-1,seam[0][1]]+t_map[i-1,j]
                image_top = remove_seam(image_top,seam)
                cache[j] = cv2.rotate(image_top,cv2.ROTATE_90_COUNTERCLOCKWISE)
                bitmap[i,j] = 1
            else:
                image_left = cache[j-1] #need to do vertical removal
                img_left_eng = image_energy(image_left)
                v_cum_eng = v_cum_min_energy(img_left_eng)
                seam_left = find_seam(v_cum_eng)
                tmap_eng_left = v_cum_eng[-1,seam_left[0][1]]+t_map[i,j-1]

                image_top = cv2.rotate(cache[j], cv2.ROTATE_90_CLOCKWISE) #need to do horiz removal
                img_top_eng = image_energy(image_top)
                h_cum_eng = v_cum_min_energy(img_top_eng)
                seam_top = find_seam(h_cum_eng)
                tmap_eng_top = h_cum_eng[-1, seam_top[0][1]] + t_map[i-1,j]

                if tmap_eng_left >= tmap_eng_top:
                    t_map[i,j] = tmap_eng_left
                    cache[j] = remove_seam(image_left,seam_left)
                else:
                    t_map[i,j] = tmap_eng_top
                    image_top = remove_seam(image_top,seam_top)
                    cache[j] = cv2.rotate(image_top,cv2.ROTATE_90_COUNTERCLOCKWISE)
                    bitmap[i,j] = 1
            if i==out_dim_r and j==out_dim_c:
                out_img = cache[j]
    return t_map, out_img, bitmap

def traverse_tmap(tmap,last_r, last_c):
    # tmap = tmap[:last_r+1,:last_c+1]
    r ,c = tmap.shape
    path_map = np.zeros((r,c))
    i, j = last_r-1, last_c-1
    path_map[i,j] = 1
    bool=True
    while bool:
        if i==0 and j==0:
            bool=False
        elif j==0:
            path_map[i-1,j] = 1
            i-=1
        elif i==0:
            path_map[i,j-1] = 1
            j-=1
        elif tmap[i-1,j]<tmap[i,j-1]:
            path_map[i-1,j] = 1
            i-=1
        else:
            path_map[i,j-1] = 1
            j-=1
    return path_map



if __name__ == "__main__":
    all_images = load_images_from_folder("images")
    fig5 = all_images[0] #coast (466,7000,3)
    fig7 = all_images[1] #butterfly (254,350,3)
    fig8 = all_images[2] #dolphin (200,239,3)
    # print fig5.shape
    # print fig7.shape
    # print fig8.shape

    coast_carve = seam_carve(fig5,466,350)
    cv2.imwrite("carved_coast.png", coast_carve)

    dolphin_insertion = seam_insertion(fig8,200, 359, averaging=False)
    dolphin_insertion_2 = seam_insertion(dolphin_insertion,200,478, averaging=False)
    cv2.imwrite("insertion_dolphin_no_avg.png", dolphin_insertion)
    cv2.imwrite("insertion_dolphin2_no_avg.png", dolphin_insertion_2)

    dolphin_insertion = seam_insertion(fig8,200, 359, averaging=True)
    dolphin_insertion_2 = seam_insertion(dolphin_insertion,200,478, averaging=True)
    cv2.imwrite("insertion_dolphin_avg.png", dolphin_insertion)
    cv2.imwrite("insertion_dolphin2_avg.png", dolphin_insertion_2)

    transport_map, final_img, bitmap = tmap(fig7,102, 185)

    cv2.imwrite("t_map_img_output.png", final_img)
    plt.imshow(transport_map,cmap="jet")
    plt.savefig("t_map.png")
    plt.clf()

    plt.imshow(bitmap,cmap="binary")
    plt.savefig("binary_tmap.png")
    plt.clf()

    pathmap = traverse_tmap(transport_map,102,185)
    plt.imshow(pathmap,cmap="binary")
    plt.savefig("path_map.png")

